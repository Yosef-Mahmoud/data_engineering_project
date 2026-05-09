## 🚀 Getting Started

To run this project locally, you need to start the FastAPI backend and serve the HTML frontend.

### Prerequisites

* Python 3.8+
* A modern web browser

### 1. Backend Setup

1. **Navigate to the Backend directory:**
Open your terminal and navigate to the `Back/` directory of the project.
```bash
cd /Back

```


2. **Create a Virtual Environment (Optional but recommended):**
```bash
python -m venv .venv
source venv\Scripts\activate
```


3. **Install Dependencies:**
Install the required Python packages using the provided requirements file.
```bash
pip install -r requirements.txt

```


*(Note: Ensure your `requirements.txt` includes `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `imbalanced-learn`, `openpyxl`, and `python-multipart` based on the imports in your code).*
4. **Run the FastAPI Server:**
Start the backend server using Uvicorn. It will run on port `8000` by default.
```bash
uvicorn main:app --reload

```


The backend API is now running at `http://localhost:8000`.

### 2. Frontend Setup

Because the frontend consists of static HTML, CSS, and JS files, you don't strictly need a build step.

1. **Navigate to the Frontend HTML directory:**
```bash
cd ../Front/html

```


2. **Open the Application:**
You can simply double-click `index.html` to open it in your browser.
*Alternatively, for the best experience (to avoid CORS/file protocol issues), serve it using Python's built-in HTTP server:*
```bash
python -m http.server 3000

```


Then open `http://localhost:3000` in your web browser.

---

## 🛠️ How to Use the Application

Once both the backend and frontend are running, follow these steps in the web interface:

### Step 1: Upload your dataset

* Click on the upload zone or use the **Choose File** button to select your dataset.
* The application accepts **.csv** and **.xlsx** (Excel) files.
* Click **Upload**.

### Step 2: Preview Data

* The system will display the first 5 rows of your dataset so you can verify the data was parsed correctly.
* Click **Looks good, continue** to proceed.

### Step 3: Configure Task

* **Task Type:** Select the type of Machine Learning problem you want to solve:
* **Classification:** Categorize data into distinct classes (applies SMOTE automatically if imbalanced).
* **Regression:** Predict continuous numerical values.
* **Clustering:** Unsupervised grouping of data (target column is disabled).


* **Target Column:** Select the column you want the model to predict (for Classification and Regression only).
* Click **Train model**.

### Step 4: Results & Download

* The application will evaluate multiple baseline models and select the best one based on performance metrics (e.g., R² for Regression, Weighted F1 for Classification, Silhouette Score for Clustering).
* Review the metrics and confusion matrices (if applicable) displayed on the screen.
* Click **Download Pipeline** to receive a serialized `.pkl` file containing the best-performing `scikit-learn` pipeline (which includes automated preprocessing, scaling, and the trained model).

---

## 🧠 Technologies Used

* **Backend:** FastAPI, Pandas, Scikit-Learn, Imbalanced-Learn
* **Frontend:** Vanilla HTML5, CSS3, JavaScript (Fetch API)