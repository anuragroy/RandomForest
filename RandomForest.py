import csv
import numpy as np  # http://www.numpy.org
import ast 
from random import randint, random 
from collections import Counter
from scipy.stats import mode

"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of total records
and d is the number of features of each record
Also, y is assumed to be a vector of d labels

XX is similar to X, except that XX also contains the data label for
each record.
"""


class RandomForest(object):
    class __DecisionTree(object):
        #tree = {}
        def __init__(self): 
            self.tree = {} 

        def learn(self, X, y):
             
            
            X_cbind = []
            
            for row, label in zip(X,y):
                row.append(label)
                X_cbind.append(row)
                
            self.tree = self.optimal_split(X_cbind)
            self.tree_builder(self.tree, 3, 40, 1) #parameters: tree, maximum depth limit, minimum size limit, current depth
            pass

        def classify(self, record):
            
            if record[self.tree['selected_column']] < self.tree['value']:
                if isinstance(self.tree['branch1'], dict):
                    x = self.tree['branch1']
                else:
                    return self.tree['branch1']
            else:
                if isinstance(self.tree['branch2'], dict):
                    x = self.tree['branch2']
                else:
                    return self.tree['branch2']

            while isinstance(x, dict):
                if record[x['selected_column']] < x['value']:
                    x = x['branch1']
                
                else:

                    x = x['branch2']
            
            return int(x)


            
        def entropy_calculator( self, data): #calculates entropy estimate to determine homogeneity of the results using formula p(x)log(p(x))
            from math import log            

            log_fn=lambda x:log(x)/log(2)  
            
            label_counter = {}
            
            entropy = float(0)

            for row in data:

                label = row[-1]
                if label not in label_counter.keys():                                         
                    label_counter[label] = 0 
                label_counter[label] += 1

            for key in label_counter.keys():
                
                p=float(label_counter[key])/len(data)
                
                entropy = entropy - p * log_fn(p)
            
            return entropy 



        def feature_splitter(self, X, column, val): #splits records in a column based on a selected value
            
            branch1 = []
            branch2 = []
            
            for record in X:
                if record[column] >= val and (isinstance(val, int) or isinstance(val, float)):
                    branch2.append(record)
                else:
                    branch1.append(record)
            
            return branch1, branch2


        
        def optimal_split(self, data): #uses information gain (via entropy) to arrive at an optimal splitting value selection
                                           
                left = None
                right = None
                selected_column = None
                value = None                        
                score = self.entropy_calculator(data)
                top_gain = float(0)

                for j in range(len(data[1])-1):
                    
                    for row in data:
                        
                        branch1, branch2 = self.feature_splitter(data, j, row[j])

                        p = float(len(branch1)) / len(data) 

                        gain = score - p*(self.entropy_calculator(branch1)) - (1-p)*(self.entropy_calculator(branch2))
                        
                        if gain > top_gain:
                            top_gain = gain
                            left = branch1
                            right = branch2
                            selected_column = j
                            val = row[j]
                
                return {"branch1": left, "branch2": right, "selected_column":j,"value": val, "information_gain": top_gain}
            

        def tree_builder(self, tree, depth_max, size_min, current_depth): #recursively builds tree using optimal splitting values until maximum depth is reached

            #if not isinstance(tree['branch1'], list ):
            #    print type(tree['branch1'])
            #if not isinstance(tree['branch2'], list ):
            #    print type(tree['branch2'])
#                try: 


            branch1 = tree['branch1']
            branch2 = tree['branch2']
            del(tree['branch1'])
            del(tree['branch2'])

            

            if not (branch1 and branch2):
                #if not isinstance(tree['branch1'], list):
                    #pass
                tree['branch1']= tree['branch2']  = self.label_ctr( branch1+branch2)
                #tree['branch2'] = label_ctr(branch1+branch2)
                return
                
            

            if current_depth >= depth_max:
                tree['branch1'] = self.label_ctr( branch1)
                tree['branch2'] = self.label_ctr( branch2)
                return
            
            if len(branch1) <= size_min or tree['information_gain'] == float(1) or self.entropy_calculator(branch1) == float(0):
                tree['branch1'] = self.label_ctr(branch1)
            else:
                tree['branch1'] = self.optimal_split(branch1)
                self.tree_builder(tree['branch1'], depth_max, size_min, current_depth + 1)
                pass

            if len(branch2) <= size_min or tree['information_gain'] == float(1) or self.entropy_calculator(branch2) == float(0):
                tree['branch2'] = self.label_ctr( branch2)
            else:
                tree['branch2'] = self.optimal_split(branch2)
                self.tree_builder(tree['branch2'], depth_max, size_min, current_depth + 1)
                pass
            pass        
                  
#                pass   

#               except (IndexError, KeyError):
#                  print tree
#                 pass

                
                
        #Calculates maximum occurring value (mode) among the results
        def label_ctr(self, data):
            try:
                results = [row[-1] for row in data]
                return max(set(results), key=results.count)             
            except (IndexError, ValueError):
                return mode(data)[0][0]

        

    num_trees = 0
    decision_trees = []
    bootstraps_datasets = [] # the bootstrapping dataset for trees
    bootstraps_labels = []   # the true class labels,
                             # corresponding to records in the bootstrapping dataset 

    def __init__(self, num_trees):
        # TODO: do initialization here.
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree() for i in range(num_trees)]
        pass 
    
    def _bootstrapping(self, XX, n):
         
        idx = np.array([randint(0 , n-1) for i in range(n)])
        XX = np.array(XX)
        train_feature = XX[idx, 0:-1]
        train_label = XX[idx, -1]
        
        return train_feature.tolist(), train_label.tolist()



    def bootstrapping(self, XX):
         
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)
        
        pass

    def fitting(self):
        for i in range(self.num_trees):
            (self.decision_trees[i]).learn(self.bootstraps_datasets[i],self.bootstraps_labels[i])
        
        pass

    def voting(self, X):
        y = np.array([], dtype = int)

        for record in X:
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]

                #try:
                if record not in dataset:

                    OOB_tree = self.decision_trees[i] 
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)
#                except ValueError:
#                    print record
#                    pass

            counts = np.bincount(votes)
            if len(counts) == 0:
                y = np.append(y,0)

            else:
                y = np.append(y, np.argmax(counts))

        return y








def main():
    X = list()
    y = list()
    XX = list() # Contains data features and data labels 

    
    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])
            xline = [ast.literal_eval(i) for i in line]
            XX.append(xline[:])

    forest_size = 5

    # Initialize a random forest
    randomForest = RandomForest(forest_size)

    # Create the bootstrapping datasets
    randomForest.bootstrapping(XX)

    # Build trees in the forest
    randomForest.fitting()

    # Provide an unbiased error estimation of the random forest 
    # based on out-of-bag (OOB) error estimate.
    y_truth = np.array(y, dtype = int)
    X = np.array(X, dtype = float)
    y_predicted = randomForest.voting(X)

    #results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]
    results = [prediction == truth for prediction, truth in zip(y_predicted, y_truth)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print "accuracy: %.4f" % accuracy

main()
