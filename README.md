\# Smart PPE Detection – AI Safety Monitoring



An AI-powered safety monitoring system that detects Personal Protective Equipment (PPE) and identifies missing safety equipment in construction environments using YOLO object detection.



\## Project Overview



Smart PPE Detection is a computer vision project designed to support workplace safety monitoring. It analyzes images to detect workers, safety equipment, and missing PPE items.



The system combines a YOLO-based object detection model with an ONNX inference pipeline, a FastAPI backend, and a React frontend.



\## Key Features



\- Detection of workers and PPE equipment.

\- Identification of missing safety equipment.

\- Class-specific confidence thresholds.

\- Bounding-box visualization and annotated images.

\- REST API built with FastAPI.

\- Interactive React dashboard.

\- ONNX Runtime CPU inference.

\- Experiment tracking with MLflow.

\- Dataset and model versioning with DVC.

\- Automated backend tests and frontend CI checks through GitHub Actions.



\## Detection Classes



The model supports 12 classes:



| Category | Classes |

|---|---|

| PPE equipment | helmet, gloves, vest, boots, goggles |

| Person | Person |

| Missing PPE | no\_helmet, no\_goggle, no\_gloves, no\_boots, no\_vest |

| Vehicle | forklift |



\## Model and Dataset



\- Model architecture: YOLO11n

\- Dataset: Construction PPE dataset

\- Total images: 1,000

\- Training images: 800

\- Validation images: 200

\- Input image size: 960 × 960

\- Export format: ONNX

\- Inference provider: ONNX Runtime CPU



\### Validation Results



The ONNX model was evaluated on the validation dataset.



| Metric | Result |

|---|---:|

| Precision | 88.76% |

| Recall | 74.75% |

| mAP@50 | 82.41% |

| mAP@50-95 | 59.24% |



The `no\_goggle` class remains a challenging detection category and is evaluated separately during confidence-threshold experiments.



\## System Architecture



```text

&#x20;            User

&#x20;             |

&#x20;             v

&#x20;      React Frontend

&#x20;             |

&#x20;             v

&#x20;       FastAPI Backend

&#x20;             |

&#x20;             v

&#x20;      Image Preprocessing

&#x20;             |

&#x20;             v

&#x20;      ONNX Runtime CPU

&#x20;             |

&#x20;             v

&#x20;      YOLO11n Detection

&#x20;             |

&#x20;             v

&#x20;   Confidence Filtering

&#x20;      + Class Thresholds

&#x20;             |

&#x20;             v

&#x20;      Detection Results

&#x20;             |

&#x20;      +------+------+

&#x20;      |             |

&#x20;      v             v

&#x20; Bounding Boxes  JSON Results

&#x20;      |             |

&#x20;      +------+------+

&#x20;             |

&#x20;             v

&#x20;       React Dashboard

```



\## Technologies



\*\*AI \& Computer Vision\*\*

\- Python

\- YOLO11n

\- Ultralytics

\- ONNX

\- ONNX Runtime

\- OpenCV / NumPy



\*\*Backend\*\*

\- FastAPI

\- Pydantic

\- Uvicorn



\*\*Frontend\*\*

\- React

\- Vite

\- Tailwind CSS



\*\*MLOps \& DevOps\*\*

\- DVC

\- MLflow

\- Git and GitHub

\- GitHub Actions

\- Pytest



\## Project Structure



```text

Smart\_PPE\_Detection/

├── .github/

│   └── workflows/

├── .dvc/

├── backend/

│   ├── app/

│   ├── tests/

│   └── requirements.txt

├── frontend/

├── data/

│   ├── raw/

│   └── processed/

├── models/

│   └── best.onnx.dvc

├── notebooks/

├── src/

├── construction-ppe.dvc

├── .dvcignore

└── .gitignore

```



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone https://github.com/mouadhrached0125/-Smart\_PPE\_Detection.git

cd -Smart\_PPE\_Detection

```



\### 2. Set up the Python environment



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install backend dependencies:



```bash

pip install -r backend/requirements.txt

```



\### 3. Retrieve versioned files with DVC



DVC must be installed and configured with access to the project's Google Drive remote.



```bash

dvc pull

```



This retrieves the versioned dataset and model files tracked by DVC.



\### 4. Start the backend



From the project root:



```bash

python -m uvicorn backend.app.main:app --reload

```



API documentation:



http://127.0.0.1:8000/docs



\### 5. Start the frontend



Open another terminal:



```bash

cd frontend

npm install

npm run dev

```



Frontend:



http://localhost:5173/



\## Testing



\### Backend tests



From the project root:



```bash

python -m pytest backend/tests -v

```



Latest local result: \*\*6 tests passed\*\*.



\### Frontend lint



```bash

cd frontend

npm run lint

```



Latest result: \*\*0 warnings and 0 errors\*\*.



\### Production build



```bash

npm run build

```



The frontend production build completed successfully.



\## MLOps Workflow



\- \*\*DVC:\*\* tracks the construction PPE dataset and ONNX model through versioned `.dvc` pointer files.

\- \*\*MLflow:\*\* records experiments, evaluation metrics, and model registration.

\- \*\*ONNX:\*\* provides a portable model format for CPU inference.

\- \*\*GitHub Actions:\*\* automates project validation and CI checks.



\## Current Limitations



\- The model's performance varies across PPE classes, particularly for some missing-equipment categories.

\- CPU inference performance depends on the machine and image size.

\- Testing has been performed using dataset images and generated test images; physical on-site safety validation has not been conducted.

\- The system is a decision-support prototype and does not replace certified safety inspections or human supervision.



\## Contributors



Developed as an engineering project in Data Science, with contributions covering computer vision, model evaluation, backend development, frontend integration, and MLOps.



\## License



Dataset and model usage remain subject to their respective licenses and terms. Check the original dataset license before redistribution.



