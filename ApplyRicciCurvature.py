import Config
import DataUtils


landmark_array_all_subjects = DataUtils.data_loader(Config.working_directory + "networkx_list_by_subject.pickle")
ricci_subjects = []

for subject_graphs in landmark_array_all_subjects:
    ricci_subjects.append(DataUtils.apply_RicciCurvature_on_list(subject_graphs))

DataUtils.data_saver(Config.working_directory + "ricci_by_subject_def.pickle", ricci_subjects)