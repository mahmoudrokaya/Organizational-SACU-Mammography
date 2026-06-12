# Raw Data

This directory contains original mammography datasets before any preprocessing or harmonization.

## Characteristics

Raw data may include:

* DICOM files
* PNG exports
* JPEG exports
* Original metadata files
* Vendor-specific acquisition information

## Important Rules

Do not:

* Modify files
* Rename files
* Delete files
* Alter metadata

The raw directory serves as the immutable source of truth.

## Example Structure

```text
raw/
├── patient_0001/
│   ├── LCC.dcm
│   ├── LMLO.dcm
│   ├── RCC.dcm
│   └── RMLO.dcm
├── patient_0002/
└── ...
```

## Expected Metadata

Typical metadata includes:

* Patient ID
* Study ID
* Acquisition date
* View position
* Laterality
* Manufacturer
* Breast density
* Diagnostic label

## Version Control

Large imaging datasets should not be committed directly to GitHub.

Use:

* Institutional storage
* Zenodo
* Data repositories
* Secure research storage

and document access procedures separately.
