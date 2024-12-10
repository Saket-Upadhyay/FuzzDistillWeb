# FuzzDistill Web

Web Interface for FuzzDistill project.
The app includes both a web interface and several API endpoints for interacting with the models.
> This module constitutes one-third of the FuzzDistill project. For further information on other
> modules and the project paper, please refer to the GitHub repository
> at [FuzzDistill](https://github.com/Saket-Upadhyay/FuzzDistill).

## Features

- **File Upload**: Users can upload CSV files through the web interface or request via POST API
  calls.
- **Model Selection**: Choose between two models (`dnnfn` and `dnnbb`) for analysis.
- **Results Visualization**: Visualizations of prediction results.
- **Caching**: Efficient file processing using cached results to avoid redundant computations.
- **REST APIs**: Programmatic access to model predictions and cache management.

### Models
1. `dnnfn` : Tensorflow DNN model trained on NIST Juliet1.3 Functions. 
2. `dnnbb` : Tensorflow DNN model trained on NIST Juliet1.3 Basic Blocks.
> More model details in [FuzzDistill Paper](https://github.com/Saket-Upadhyay/FuzzDistill)

## Project Structure

```text
../FuzzDistillWeb
├── README.md
├── app.py
├── includes
│   └── constants.py
├── models
│   ├── BBpredict_TF_21_32_68.keras
│   └── FNpredict_TF_30_32_86.keras
├── requirements.txt
├── templates
│   ├── index.html
│   └── results.html
├── test
│   ├── 100bbfeat.csv
│   ├── 200fndata.csv
│   └── xpdf3.02FNfeatures.csv

```

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Saket-Upadhyay/FuzzDistillWeb.git
    cd FuzzDistillWeb
    ```

2. **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Environment Variables**:
   Create a `.env` file with the following structure:
    ```env
    FLASK_APP=app.py
    FLASK_ENV=development  # Use 'production' for deployment
    SECRET_KEY=your_secret_key_here
    ```

5. **Run the Application**:
    ```bash
    flask run
    ```

The application will be accessible at `http://127.0.0.1:5000/`.

---

## API Endpoints

Below is a list of available API endpoints.

### 1. Prediction APIs

#### `POST /api/high-conf-list`

- **Description**: Returns a list of high-confidence predictions.
- **Request**: Upload a CSV file with the `file` parameter and specify `modelselect` (`dnnfn` or
  `dnnbb`).
- **Response**:
    ```json
    [
        "Function1",
        "Function2"
    ]
    ```

#### `POST /api/sure-list`

- **Description**: Returns a list of predictions with 100% confidence.
- **Request**: Upload a CSV file with the `file` parameter and specify `modelselect`.
- **Response**:
    ```json
    [
        "Function1",
        "Function2"
    ]
    ```

#### `POST /api/all-list`

- **Description**: Returns all predictions with their confidence scores.
- **Request**: Upload a CSV file with the `file` parameter and specify `modelselect`.
- **Response**:
    ```json
    [
        { "Function Name": "Function1", "confidence": 0.97 },
        { "Function Name": "Function2", "confidence": 1.0 }
    ]
    ```

### 2. Cache Management APIs

#### `GET /api/clear-cache-record`

- **Description**: Clears the cache for a specific file.
- **Query Parameter**:
    - `hash`: The hash of the file to clear.
- **Response**:
    ```json
    { "message": "Cache cleared for file hash: your_hash" }
    ```

#### `POST /api/clear-cache`

- **Description**: Clears the entire cache.
- **Response**:
    ```json
    { "message": "Entire cache cleared" }
    ```

---

## Example

1. Get json list of 100% confident predictions for a file containing function features, using dnnfn
   to use tensorflow function prediction model.

```shell
curl -L 'http://127.0.0.1:5000/api/sure-list' \
-F 'file=@"./test/xpdf3.02FNfeatures.csv"' \
-F 'modelselect="dnnfn"'
```

Example output:

```json
[
  "FAnnot::Annot(XRef*, Dict*, Dict*, Ref*)",
  "FAnnot::generateFieldAppearance(Dict*, Dict*, Dict*)",
  "FJBIG2Stream::readTextRegion(int, int, int, int, unsigned int, unsigned int, int, JBIG2HuffmanTable*, unsigned int, JBIG2Bitmap**, unsigned int, unsigned int, unsigned int, unsigned int, int, JBIG2HuffmanTable*, JBIG2HuffmanTable*, JBIG2HuffmanTable*, JBIG2HuffmanTable*, JBIG2HuffmanTable*, JBIG2HuffmanTable*, JBIG2HuffmanTable*, JBIG2HuffmanTable*, unsigned int, int*, int*)",
  "FJPXStream::inverseTransformLevel(JPXTileComp*, unsigned int, JPXResLevel*, unsigned int, unsigned int, unsigned int, unsigned int)",
  "FOutputDev::beginType3Char(GfxState*, double, double, double, double, unsigned int, unsigned int*, int)",
  "FOutputDev::beginTransparencyGroup(GfxState*, double*, GfxColorSpace*, int, int, int)",
  "FOutputDev::setSoftMask(GfxState*, double*, int, Function*, GfxColor*)",
  "FPage::displaySlice(OutputDev*, double, double, int, int, int, int, int, int, int, int, Catalog*, int (*)(void*), void*)",
  "FPage::makeBox(double, double, int, int, int, double, double, double, double, PDFRectangle*, int*)",
  "FParser::getObj(Object*, unsigned char*, CryptAlgorithm, int, int, int)",
  "FPDFDoc::PDFDoc(GString*, GString*, GString*, void*)",
  "FPDFDoc::displayPages(OutputDev*, int, int, double, double, int, int, int, int, int (*)(void*), void*)",
  "FPDFDoc::displayPageSlice(OutputDev*, int, double, double, int, int, int, int, int, int, int, int, int (*)(void*), void*)",
  "FPreScanOutputDev::drawSoftMaskedImage(GfxState*, Object*, Stream*, int, int, GfxImageColorMap*, Stream*, int, int, GfxImageColorMap*)",
  "FOutputDev::drawChar(GfxState*, double, double, double, double, double, double, unsigned int, int, unsigned int*, int)"
]
```

2. Clear Cache using cURL

```shell
curl -L 'http://127.0.0.1:5000/api/clear-cache'
```

output:

```json
{
  "message": "Entire cache cleared"
}
```

---

## Deployment

1. Configure a production WSGI server like Gunicorn:
    ```bash
    gunicorn -w 4 -b 0.0.0.0:8000 app:app
    ```

2. Use a reverse proxy like Nginx for improved performance and security.

3. Update your `.env` to use the production configuration.

---

## License

This project is licensed under the [MIT License](LICENSE).

---


