# # Check, if in the chunk there is only 1 class
#             for i in range(concept_kwargs["n_chunks"]):
#                 X_cl, y_cl = stream.get_chunk() # tu problem, bo potem w stream learnie - chyba w evaluator nie moze znowu wywolac wewnatrz tej metody get_chunk()
#                 print(len(np.unique(y_cl)))
#                 if len(np.unique(y_cl)) == 1:
#                     one_class = True
            
#             # Exclude SVC, because for it, the number of classes cannot be 0
#             if one_class == True:
#                 clfs = [
#                     HDWE(GaussianNB(), pred_type="hard"),
#                     HDWE(MLPClassifier(hidden_layer_sizes=(10)), pred_type="hard"),
#                     HDWE(DecisionTreeClassifier(), pred_type="hard"),
#                     HDWE(HDDT(), pred_type="hard"),
#                     HDWE(KNeighborsClassifier(), pred_type="hard"),
#                 ]
#                 clf_names = [
#                     "HDWE-GNB",
#                     "HDWE-MLP",
#                     "HDWE-CART",
#                     "HDWE-HDDT",
#                     "HDWE-KNN",
#                 ]



# import numpy as geek 
  
  
# x = geek.array([[0, 1, 2, 3], [4, 5, 6, 7]], 
#                                  order ='F') 
# print("x is: \n", x) 
  
# # copying x to y 
# y = x.copy() 
# y.append(90, 99)
# print("y is :\n", y) 
# print("\nx is copied to y") 

n_streams = 84
r = []
r[range(n_streams)] = 9

print(r)