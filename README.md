INSTRUCTIONS:

1. Download the dataset from this link: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
    - I used the Gift Cards dataset as it the smallest and fastest for training / processing.
    - On the page click here:
        ![Picture of Dataset Website](img/dataset.png)
    - Once downloaded, move to data/amazon in root. If these directories don't exist, create them. Path should be HOUSER/data/amazon/Gift_Cards.jsonl

2. For data analysis, run 
        python data_analysis 
    from root. This may take a second as it takes a while to create the images with so much data.

3. To see all model performance for link prediction or edge classification, run 
        python -m link_prediction.autorun 
    or 
        python -m link_prediction.autorun

4. To simply run the HOUSER model, run 
        python -m houser