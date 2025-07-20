# Summary

This repo contains solutions from Utkarsh Chaturvedi <utkarsh.chaturvedi@u.nus.edu> for the GovTech Take-Home case study. The questions asked to the candidate in the case study are also attached to this repo as a Word Document file.

The candidate has answered both questions in Section 1 and Scenario 1 in Section 2 (ECDA subzone demand). All resources for the solution along with the presentation are kept in folders marked as 1_1, 1_2 and 2_1 for Questions 1 and 2 in Section 1 and Question 1 in Section 2 respectively.

## Repo Structure
There are 5 key sub-directories, each following the structure specified in summary above:

* code: This subdirectory documents the code for each project along with a seperate subdirectory "utils/" for downloading raw data using API.
NOTE: Some datasets could not be found on the website and had to be downloaded manually from different sources.
NOTE 2: Since datasets are too big to be sent over as a Zipped file, a snippet of all raw data-sets used has been provided for the user to find and download.

* data: This subdirectory should be used to store raw data for each question seperately.
NOTE: 5 year-forecasts for Section 2, Scenario 1 have been placed in "/2_1/forecast/" subdirectory.

* images: Any images produced during analysis or model run are stored in this subdirectory.

* model: The saved model for Scenario 1 - Question 2 is saved here. Other scenrios did not need a saved model.

* presentations: Presentations for each scenario labelled with the same naming convention as above.

## Get Started
Create a virtual Python environment (>=3.10.11), install all packages from requirements.txt and run Python scripts.