INSTRUCTIONS:

1. Download the dataset from this link: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
    - I used the Gift Cards dataset as it the smallest and fastest for training / processing.
    - On the page click here:
        ![Picture of Dataset Website](img/dataset.png)
    - In terminal run
        mkdir data
        mkdir data/amazon
    - Once downloaded, unpack the zip file. It should be in Gift_Cards.jsonl format. Move to data/amazon. 

2. Create the weights directories by running
        mkdir link_prediction/models/weights
        mkdir edge_classification/models/weights
        mkdir eval/models/weights

3. For data analysis, run 
        python data_analysis 
    from root. This may take a second as it takes a while to create the images with so much data.

4. To see all model performance for link prediction or edge classification, run 
        python -m link_prediction.autorun 
    or 
        python -m link_prediction.autorun

5. To simply run the HOUSER model, run 
        python -m houser