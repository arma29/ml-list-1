import processing
import training
import reporting

def main():
    raw_lst = processing.get_raw_data()
    names_lst = processing.get_raw_names()

    for idx in range(len(raw_lst)):
        dataset_name = names_lst[idx]

        X,y,target_names = processing.process_data(raw_lst[idx])

        model_dict = training.train_model(X,y,target_names,dataset_name)

        training.print_elapsed_time(model_dict)

        reporting.produce_report(model_dict)

if __name__ == "__main__":
    main()
