import os

import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from doc_topic_definition_data_and_models.modules.parser import edit, parse_data


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    output_dir = config["data_load"]["data_path"]
    max_links_pages_for_hub = int(config["data_load"]["max_number_of_pages_with_links_for_hub"])
    hubs = config["data_load"]["hubs"]
    os.makedirs(output_dir, exist_ok=True)

    print(hubs)

    print("Parsing and saving data from habr.ru...")
    data = parse_data(hubs, output_dir, max_links_pages_for_hub)



    print("Cleaning data from habr.ru...")
    prepared_data = edit(output_dir)


    # print("Saving data from habr.ru...")
    # train_df, test_df = train_test_split(
    #     prepared_data, test_size=0.2, random_state=42, stratify=prepared_data["target"]
    # )

    # train_path = os.path.join(output_dir, "docs_train.csv")
    # test_path = os.path.join(output_dir, "docs_test.csv")

    # train_df.to_csv(train_path, index=False, encoding="utf-8")
    # test_df.to_csv(test_path, index=False, encoding="utf-8")

    # print(f"Training and testing data saved to: {output_dir}")


if __name__ == "__main__":
    main()
