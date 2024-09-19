from setuptools import find_packages, setup

setup(
    name="story_illustrator",
    version="0.1.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/soumik12345/diffusion-story-illustration",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={
        "fal": [
            "fal-client>=0.4.1",
            "openai>=1.42.0",
            "weave>=0.51.7",
        ],
    },
)
