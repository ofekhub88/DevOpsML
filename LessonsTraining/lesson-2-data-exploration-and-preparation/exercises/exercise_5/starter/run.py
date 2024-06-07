#!/usr/bin/env python
import argparse
import logging,os
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    ## YOUR CODE HERE
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)

    df = pd.read_parquet(artifact.file())
    # clean duplicate data 
    df = df.drop_duplicates().reset_index(drop=True)
    
    logger.info("Fixing missing values")
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    df.to_csv(args.artifact_name, index=False)
    #
    ## Log the CSV file as an artifact
    artifact = wandb.Artifact(args.artifact_name, 
                              type=args.artifact_type,
                              description = args.artifact_description)
    artifact.add_file(args.artifact_name)
    logger.info("Logging artifact")
    run.log_artifact(artifact)
    os.remove(args.artifact_name)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
