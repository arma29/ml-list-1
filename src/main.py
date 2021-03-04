import processing
import training
import reporting

def main():
    
    '''
    int_data = create_int_data(raw_data)
    pro_drug_features = create_pro_drug_features(int_data)
    pro_patient_features = create_pro_patient_features(int_data)
    pro_master_table = create_pro_master_table(pro_drug_features, pro_patient_features)
    model = train_model(pro_master_table)
    rpt_report = produce_report(model)
    '''
    raw_lst = processing.get_raw_data()
    names_lst = processing.get_raw_names()
    idx = 0
    dataset_name = names_lst[idx]

    X,y,target_names = processing.process_data(raw_lst[idx])

    model_dict = training.train_model(X,y,target_names,dataset_name)

    training.print_elapsed_time(model_dict)

    reporting.produce_report(model_dict)

if __name__ == "__main__":
    main()
