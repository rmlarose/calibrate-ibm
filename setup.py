# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup


setup(
    name="q4bio",
    version="0.0.1a",
    install_requires=["qiskit"],
    packages=["q4bio"],
    include_package_data=True,
    description="Quantum hardware code for the Stanford Q4Bio team.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Stanford Q4Bio team",
    license="Apache 2.0",
    url="https://github.com/rmlarose/calibrate-ibm",
    project_urls={
        "Bug Tracker": "https://github.com/rmlarose/calibrate-ibm/issues/",
        "Documentation": "https://github.com/rmlarose/calibrate-ibm",
        "Source": "https://github.com/rmlarose/calibrate-ibm",
    },
    python_requires=">=3.9.0",
)
