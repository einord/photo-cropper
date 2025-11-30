use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use opencv::core::{self, AlgorithmHint, Mat, Point, Point2f, Scalar, Size, Vector};
use opencv::imgcodecs;
use opencv::imgproc::{self, InterpolationFlags};
use opencv::prelude::*;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Extracts individual photos from scanned sheets and saves them straightened."
)]
struct Args {
    /// Directory containing input images
    input_dir: PathBuf,
    /// Directory where cropped images will be written
    output_dir: PathBuf,
    /// Minimum contour area to consider as a photo (in pixels)
    #[arg(long, default_value_t = 20_000.0)]
    min_area: f64,
}

struct DetectedPhoto {
    warped: Mat,
}

fn main() -> Result<()> {
    let args = Args::parse();

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output dir {:?}", args.output_dir))?;

    for entry in WalkDir::new(&args.input_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if !is_image_file(path) {
            continue;
        }

        println!("Processing {}...", path.display());
        match process_image(path, &args.output_dir, args.min_area) {
            Ok(count) => {
                if count == 0 {
                    println!("  No photos found");
                } else {
                    println!("  Saved {count} cropped photos");
                }
            }
            Err(err) => {
                eprintln!("  Failed: {err:?}");
            }
        }
    }

    Ok(())
}

fn process_image(path: &Path, output_dir: &Path, min_area: f64) -> Result<usize> {
    let img = imgcodecs::imread(path.to_str().unwrap_or_default(), imgcodecs::IMREAD_COLOR)
        .with_context(|| format!("Could not read image {}", path.display()))?;

    let crops = detect_photos(&img, min_area)
        .with_context(|| format!("Failed to analyze {}", path.display()))?;

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");

    for (idx, crop) in crops.iter().enumerate() {
        let filename = format!("{}_{}.jpg", stem, idx + 1);
        let mut out_path = output_dir.to_path_buf();
        out_path.push(filename);

        imgcodecs::imwrite(
            out_path.to_str().unwrap_or_default(),
            &crop.warped,
            &Vector::new(),
        )
        .with_context(|| format!("Failed to save cropped photo to {}", out_path.display()))?;
    }

    Ok(crops.len())
}

fn detect_photos(image: &Mat, min_area: f64) -> Result<Vec<DetectedPhoto>> {
    let mut gray = Mat::default();
    imgproc::cvt_color(
        image,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        Size::new(5, 5),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Adaptive threshold handles both light and dark backgrounds.
    let mut binary = Mat::default();
    imgproc::adaptive_threshold(
        &blurred,
        &mut binary,
        255.0,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY,
        25,
        10.0,
    )?;

    let mut inverted = Mat::default();
    core::bitwise_not(&binary, &mut inverted, &core::no_array())?;

    let mut edges = Mat::default();
    imgproc::canny(&inverted, &mut edges, 50.0, 150.0, 3, false)?;

    let kernel =
        imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(5, 5), Point::new(-1, -1))?;
    let mut dilated = Mat::default();
    imgproc::dilate(
        &edges,
        &mut dilated,
        &kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;
    edges = dilated;

    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &edges,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut photos = Vec::new();

    for contour in contours {
        let area = imgproc::contour_area(&contour, false)?;
        if area < min_area {
            continue;
        }

        let rect = imgproc::min_area_rect(&contour)?;
        let size = rect.size;
        if size.width <= 1.0 || size.height <= 1.0 {
            continue;
        }

        let warped = warp_photo(image, &rect)?;
        photos.push(DetectedPhoto { warped });
    }

    photos.sort_by(|a, b| b.warped.total().cmp(&a.warped.total()));
    Ok(photos)
}

fn warp_photo(image: &Mat, rect: &core::RotatedRect) -> Result<Mat> {
    let mut box_points = [Point2f::default(); 4];
    rect.points(&mut box_points)?;
    let ordered = order_points(&box_points);

    let width_top = distance(&ordered[0], &ordered[1]);
    let width_bottom = distance(&ordered[3], &ordered[2]);
    let max_width = width_top.max(width_bottom).round() as i32;

    let height_left = distance(&ordered[0], &ordered[3]);
    let height_right = distance(&ordered[1], &ordered[2]);
    let max_height = height_left.max(height_right).round() as i32;

    let max_width = max_width.max(1);
    let max_height = max_height.max(1);

    let dst: Vector<Point2f> = Vector::from(vec![
        Point2f::new(0.0, 0.0),
        Point2f::new((max_width - 1) as f32, 0.0),
        Point2f::new((max_width - 1) as f32, (max_height - 1) as f32),
        Point2f::new(0.0, (max_height - 1) as f32),
    ]);

    let src: Vector<Point2f> = Vector::from(ordered.to_vec());
    let m = imgproc::get_perspective_transform(&src, &dst, core::DECOMP_LU)?;

    let mut warped = Mat::default();
    imgproc::warp_perspective(
        image,
        &mut warped,
        &m,
        Size::new(max_width, max_height),
        InterpolationFlags::INTER_CUBIC as i32,
        core::BORDER_REPLICATE,
        Scalar::all(0.0),
    )?;

    Ok(warped)
}

fn order_points(points: &[Point2f; 4]) -> [Point2f; 4] {
    let mut ordered = [Point2f::default(); 4];

    // top-left has smallest sum, bottom-right largest sum
    ordered[0] = *points
        .iter()
        .min_by(|a, b| (a.x + a.y).partial_cmp(&(b.x + b.y)).unwrap())
        .unwrap();
    ordered[2] = *points
        .iter()
        .max_by(|a, b| (a.x + a.y).partial_cmp(&(b.x + b.y)).unwrap())
        .unwrap();

    // top-right has smallest difference, bottom-left largest difference
    ordered[1] = *points
        .iter()
        .min_by(|a, b| (a.x - a.y).partial_cmp(&(b.x - b.y)).unwrap())
        .unwrap();
    ordered[3] = *points
        .iter()
        .max_by(|a, b| (a.x - a.y).partial_cmp(&(b.x - b.y)).unwrap())
        .unwrap();

    ordered
}

fn distance(a: &Point2f, b: &Point2f) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

fn is_image_file(path: &Path) -> bool {
    const EXTENSIONS: [&str; 6] = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"];
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}
