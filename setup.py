from setuptools import setup, find_packages

setup(
    name="dr-diffusion-augmentation",
    version="1.0.0",
    author="Berdjouh Chemousse",
    author_email="your_email@example.com",
    description="Diabetic Retinopathy Classification using Diffusion-Based Data Augmentation",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "opencv-python",
        "Pillow",
        "tqdm",
    ],
    python_requires=">=3.9",
)
