# Check, if in the chunk there is only 1 class
            # for i in range(concept_kwargs["n_chunks"]):
            #     X_cl, y_cl = stream.get_chunk() # tu problem
            #     print(len(np.unique(y_cl)))
            #     if len(np.unique(y_cl)) == 1:
            #         one_class = True
            
            # # Exclude SVC, because for it, the number of classes cannot be 0
            # if one_class == True:
            #     clfs = [
            #         HDWE(GaussianNB(), pred_type="hard"),
            #         HDWE(MLPClassifier(hidden_layer_sizes=(10)), pred_type="hard"),
            #         HDWE(DecisionTreeClassifier(), pred_type="hard"),
            #         HDWE(HDDT(), pred_type="hard"),
            #         HDWE(KNeighborsClassifier(), pred_type="hard"),
            #     ]
            #     clf_names = [
            #         "HDWE-GNB",
            #         "HDWE-MLP",
            #         "HDWE-CART",
            #         "HDWE-HDDT",
            #         "HDWE-KNN",
            #     ]