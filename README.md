## Photo Cropper

Liten CLI-app som hittar separata foton i skannade bilder, rätar upp dem och sparar dem som nya filer.

### Förkrav

- Rust toolchain (https://www.rust-lang.org/tools/install)
- OpenCV installerat lokalt (krävs av Rust-bindningen `opencv`).
  - macOS: `brew install opencv`
  - Ubuntu/Debian: `sudo apt install libopencv-dev`
  - Windows: installera OpenCV och sätt `OpenCV_DIR` till din byggmapp, se https://docs.rs/opencv/latest/opencv/#user-content-windows

### Körning

```bash
cargo run -- <input_dir> <output_dir> [--min-area 20000] [--pad 12] [--canny-low 50] [--canny-high 150]
```

- `input_dir`: mapp med källbilderna (t.ex. skannade ark med flera foton).
- `output_dir`: mapp där beskurna och rätade foton sparas.
- `--min-area`: minsta kontursyta som tolkas som ett foto (pixlar). Öka om du får brus, sänk om mindre bilder missas.
- `--pad`: antal pixlar som läggs som padding runt bilden innan detektering, bra om foton ligger ända ut i kanten.
- `--canny-low` / `--canny-high`: tröskelvärden för Canny-kantdetektering (hysteresis). Lägre värden = mer känslig; högre värden = mindre brus. Normal praxis är att börja med `high` ≈ 3×`low`. Om `high` <= `low` sätts `high` automatiskt till 3×`low`.

### Tips för Canny-trösklar

- Svag kontrast / mjuka kanter: sänk båda, t.ex. `--canny-low 20 --canny-high 60`.
- Mycket brus eller texturrika bakgrunder: höj båda, t.ex. `--canny-low 80 --canny-high 200`.
- Testa med små steg (±10–20) tills konturerna ser rimliga ut. Programmet behöver tydliga, sammanhängande konturer runt varje foto för att detektera rektanglarna.

Programmet loggar vilka filer som hittas och hur många foton som sparas per bild.

### Så funkar det (kort)

1. Läser varje bild i källmappen.
2. Gråskala + Gaussian blur för att minska brus.
3. Adaptiv tröskling (klarar ljus/mörk bakgrund) + invertering.
4. Canny-kantdetektering och dilation för att stänga glapp.
5. Letar konturer, filtrerar på area och skapar minsta omslutande roterade rektangel.
6. Perspektivtransform av rektangeln till en rak bild och sparar som `filnamn_#.jpg` i `output_dir`.

Justera `--min-area` eller kernelstorlekar i `src/main.rs` om dina bilder kräver mer finlir.
