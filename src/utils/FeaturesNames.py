def get_features_names() -> dict[str, list[str]]:
    """
    Defines the features names for each configuration, and returns it in dictionary structure.

    ### Parameters :
    None

    ### Returns :
    A dictionary structure, where keys are the configurations, and values are the features names.
    """
    features_names_no_sigmut = ["Age at advanced disease diagnosis",
                                "CD8+ T cell score",
                                "Genome mut per mb",
                                "Exome mut per mb",
                                "CD274 expression",
                                "M1M2 expression"]

    features_names_only_sigmut_sbs = ["SBS1",
                                      "SBS2",
                                      "SBS3",
                                      "SBS4",
                                      "SBS5",
                                      "SBS7a",
                                      "SBS7b",
                                      "SBS7c",
                                      "SBS7d",
                                      "SBS8",
                                      "SBS10a",
                                      "SBS10b",
                                      "SBS10c",
                                      "SBS13",
                                      "SBS15",
                                      "SBS17a",
                                      "SBS17b",
                                      "SBS18",
                                      "SBS31",
                                      "SBS35",
                                      "SBS36",
                                      "SBS37",
                                      "SBS38",
                                      "SBS40",
                                      "SBS44",
                                      "SBS4426"]

    features_names_only_sigmut_indel = ["ID1",
                                        "ID2",
                                        "ID3",
                                        "ID4",
                                        "ID5",
                                        "ID6",
                                        "ID7",
                                        "ID8",
                                        "ID9",
                                        "ID10",
                                        "ID11",
                                        "ID12",
                                        "ID13",
                                        "ID14",
                                        "ID15",
                                        "ID16",
                                        "ID17",
                                        "ID18"]

    features_names_only_sigmut_comb = ["SBS1",
                                       "SBS2",
                                       "SBS3",
                                       "SBS4",
                                       "SBS5",
                                       "SBS7a",
                                       "SBS7b",
                                       "SBS7c",
                                       "SBS7d",
                                       "SBS8",
                                       "SBS10a",
                                       "SBS10b",
                                       "SBS10c",
                                       "SBS13",
                                       "SBS15",
                                       "SBS17a",
                                       "SBS17b",
                                       "SBS18",
                                       "SBS31",
                                       "SBS35",
                                       "SBS36",
                                       "SBS37",
                                       "SBS38",
                                       "SBS40",
                                       "SBS44",
                                       "SBS4426",
                                       "ID1",
                                       "ID2",
                                       "ID3",
                                       "ID4",
                                       "ID5",
                                       "ID6",
                                       "ID7",
                                       "ID8",
                                       "ID9",
                                       "ID10",
                                       "ID11",
                                       "ID12",
                                       "ID13",
                                       "ID14",
                                       "ID15",
                                       "ID16",
                                       "ID17",
                                       "ID18"]

    features_names_comb = ["Age at advanced disease diagnosis",
                           "CD8+ T cell score",
                           "Genome mut per mb",
                           "Exome mut per mb",
                           "CD274 expression",
                           "M1M2 expression",
                           "SBS1",
                           "SBS2",
                           "SBS3",
                           "SBS4",
                           "SBS5",
                           "SBS7a",
                           "SBS7b",
                           "SBS7c",
                           "SBS7d",
                           "SBS8",
                           "SBS10a",
                           "SBS10b",
                           "SBS10c",
                           "SBS13",
                           "SBS15",
                           "SBS17a",
                           "SBS17b",
                           "SBS18",
                           "SBS31",
                           "SBS35",
                           "SBS36",
                           "SBS37",
                           "SBS38",
                           "SBS40",
                           "SBS44",
                           "SBS4426",
                           "ID1",
                           "ID2",
                           "ID3",
                           "ID4",
                           "ID5",
                           "ID6",
                           "ID7",
                           "ID8",
                           "ID9",
                           "ID10",
                           "ID11",
                           "ID12",
                           "ID13",
                           "ID14",
                           "ID15",
                           "ID16",
                           "ID17",
                           "ID18"]

    # Dictionary structure to avoid if-elif-else
    dico_features_names = {
        "no-sigmut": features_names_no_sigmut,
        "only-sigmut-sbs": features_names_only_sigmut_sbs,
        "only-sigmut-indel": features_names_only_sigmut_indel,
        "only-sigmut-comb": features_names_only_sigmut_comb,
        "comb": features_names_comb
    }

    return dico_features_names
