from __future__ import division
import csv
import math
import sys
import copy
import random

class Node:
    def __init__(self):
        self.attribute = None
        self.left = None
        self.right = None
        self.parent = None
        self.leaf = False
        self.inc = ""
        self.value = None
        self.id = id
        self.order = None

    def _updateOrder(self,order):
        self.order = order

class CopyOfNode(Node):
    def __init__(self, node):
        if node is None:
            return None
        if node.leaf is not None:
            self.attribute = node.attribute
            self.parent = node.parent
            self.leaf = node.leaf
            self.inc = node.inc
        else:
            self.attribute = node.attribute
            self.parent = node.parent
            self.leaf = node.leaf
            self.inc = node.inc
            self.id = node.id
            self.order = node.order

        if node.left is not None:
            self.left = CopyOfNode(node.left)
        else:
            self.left = None

        if node.right is not None:
            self.right = CopyOfNode(node.right)
        else:
            self.right = None

# function to parse csv file, returns a list of list of values
def parse_csv_file(csv_filename):
    attribute_list = []
    with open(csv_filename,'r') as csv_file:
        has_header = csv.Sniffer().has_header(csv_file.read(1024))
        csv_file.seek(0)
        csv_reader = csv.reader(csv_file, csv.QUOTE_NONNUMERIC)
        if has_header:
            attribute_list = next(csv_reader)
        data_list = [[int(x) for x in line] for line in csv.reader(csv_file, delimiter=',')]
    return data_list, attribute_list


def entropy(data, target_attr):
    data = data[:]
    val_freq = {}
    data_entropy = 0.0
    for data_list in data:
        if val_freq.has_key(data_list[target_attr]):
            val_freq[data_list[target_attr]] += 1.0
        else:
            val_freq[data_list[target_attr]] = 1.0
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)
    return data_entropy


def calculate_gain(data_list,attribute,target_attribute):
    data_list = data_list[:]
    attribute = attribute[:]
    attribute_value_frequency = {}
    subset_entropy = 0.0
    # Calculate the frequency of each of the values in the target attribute
    for each_list in data_list:
        if each_list[attribute] in attribute_value_frequency:
        #if attribute_value_frequency.has_key(each_list[attribute]):
            attribute_value_frequency[each_list[attribute]] += 1.0
        else:
            attribute_value_frequency[each_list[attribute]] = 1.0
    #print attribute, attribute_value_frequency
    for val in attribute_value_frequency.keys():
        val_prob = attribute_value_frequency[val] / sum(attribute_value_frequency.values())
        data_subset = [each_list for each_list in data_list if each_list[attribute] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attribute)
    return (entropy(data_list, target_attribute) - subset_entropy)

def vi_s(data_list,target_attribute):
    #print data_list
    #print len(data_list)
    data_list = data_list[:]
    pos_class_count = 0
    neg_class_count = 0
    for each_list in data_list:
        if each_list[target_attribute] == 1:
            #if attribute_value_frequency.has_key(each_list[attribute]):
            pos_class_count += 1
        else:
            neg_class_count += 1
    #print pos_class_count, neg_class_count
    vi_of_s = float((pos_class_count * neg_class_count)/(len(data_list) ** 2))
    #print vi_of_s
    return vi_of_s


def calclulate_variance_impurity_gain(data_list,attribute,target_attribute):
    data_list = data_list[:]
    vi_of_s = vi_s(data_list,target_attribute)
    #print attribute
    #print vi_of_s
    count_0_0 = 0
    count_0_1 = 0
    count_1_1 = 0
    count_1_0 = 0
    for each_list in data_list:
        if each_list[attribute] == 0 and each_list[target_attribute] == 0:
            count_0_0 += 1
        elif each_list[attribute] == 0 and each_list[target_attribute] == 1:
            count_0_1 += 1
        elif each_list[attribute] == 1 and each_list[target_attribute] == 1:
            count_1_1 += 1
        elif each_list[attribute] == 1 and each_list[target_attribute] == 0:
            count_1_0 += 1

    total_0 = count_0_0 + count_0_1
    prob_matched_0 = count_0_0 / total_0 if total_0 else 0
    prob_unmatched_0 = count_0_1 / total_0 if total_0 else 0
    vi_s_0 = (total_0 / len(data_list)) * (prob_matched_0 * prob_unmatched_0)

    total_1 = count_1_0 + count_1_1
    prob_matched_1 = count_1_0 / total_1 if total_1 else 0
    prob_unmatched_1 = count_1_1 / total_1 if total_1 else 0
    vi_s_1 = (total_1 / len(data_list)) * (prob_matched_1 * prob_unmatched_1)

    #print vi_s_0, vi_s_1
    vi_s_attr = vi_s_0 + vi_s_1

    info_gain = vi_of_s - vi_s_attr

    return info_gain



def choose_best_attribute(data, attribute_list, target_attribute,heuristic):
    data = data[:]
    max_gain = 0.0
    best_attribute = None
    # Find gain for eah attribute in the attribute list and pick the one with max gain
    #print "Printing attribute and gain"
    for attribute in attribute_list:
        if heuristic == 'gain':
            gain = calculate_gain(data, attribute, target_attribute)
        elif heuristic == 'variance_impurity' and attribute != target_attribute:
            gain = calclulate_variance_impurity_gain(data, attribute, 'Class')
        #print attribute, gain
        if gain >= max_gain and attribute != target_attribute:
            max_gain = gain
            best_attribute = attribute
    return best_attribute


def get_unique_class_values_for_attribute(data, attribute):
    data = data[:]
    return unique_class_values([data_values[attribute] for data_values in data])


def unique_class_values(class_values_list):
    class_values_list = class_values_list[:]
    unique_classes_list = []
    # Cycle through the list and add each value to the unique list only once.
    for item in class_values_list:
        if unique_classes_list.count(item) <= 0:
            unique_classes_list.append(item)
    # Return the list with all redundant values removed.
    return unique_classes_list


def majority_value(data_list, target_attrribute):
    # create a list of only target attribute values i.e list of values for 'Class'
    data_list = data_list[:]
    class_values_list = []
    for each_list in data_list:
        class_values_list.append(each_list[target_attrribute])
    # find the most frequest element in the class
    highest_frequency = 0
    most_frequent_class = None
    for class_value in unique_class_values(class_values_list):
        if class_values_list.count(class_value) > highest_frequency:
            most_frequent_class = class_value
            highest_frequency = class_values_list.count(class_value)
    return most_frequent_class


def count_postive_negative_classes(data, target_attrribute):
    data_list = data[:]
    negative_class_count = 0
    positive_class_count = 0
    for each_list in data_list:
        if each_list[target_attrribute] == 0:
            negative_class_count += 1
        else:
            positive_class_count += 1
    return positive_class_count, negative_class_count


def split_data(data, attribute):
    pos_attr_val_list = []
    neg_attr_val_list = []
    for item in data:
        if item[attribute] == 1:
            pos_attr_val_list.append(item)
        else:
            neg_attr_val_list.append(item)
    return pos_attr_val_list, neg_attr_val_list


def remove_best_attr_value(data, best_attribute):
    data_list = data[:]
    for item in data_list:
        del item[best_attribute]
    return data_list


def id3(data, attribute_list, target_attribute, heuristic, inc, parent, leaf=None, id=0):
    # Creating root node for the tree
    node = Node()
    node.inc = inc
    node.leaf = leaf
    node.id = id + 1
    node.left = None
    node.right = None
    node.parent = parent
    if leaf is not None or data == []:
        node.attribute = None
        node.left = None
        node.right = None
        node.id = id
        return node

    #Calculating total postive ad negative class in the data
    pos_class_count, neg_class_count = count_postive_negative_classes(data,target_attribute)
    #print pos_class_count, neg_class_count

    #If all Examples are positive, Return the single-node tree Root, with label = +
    if pos_class_count == len(data):
        node.leaf = 1
        node.attribute = None
        return node
    #If all Examples are negative, Return the single-node tree Root, with label = -
    elif neg_class_count == len(data):
        node.leaf = 0
        node.attribute = None
        return node

    #If Attributes is empty, Return the single-node tree Root, with label = most common value of Target attribute in Examples
    if len(attribute_list) == 0:
        if neg_class_count > pos_class_count:
            node.leaf = 0
        else:
            node.leaf = 1
        node.attribute = None
        return node

    #Find the best attribute to split on
    best_attribute = choose_best_attribute(data,attribute_list,target_attribute, heuristic)
    #print "Best Attribute = %s" % best_attribute
    #Set the current nodes attr to be the best attr
    node.attribute = best_attribute
    #print best_attribute

    #Split the data into 2 sets , based on best attribute value
    #pos_class_count =  neg_class_count = 0
    data_with_attr_val_1, data_with_attr_val_0 = split_data(data, best_attribute)
    #print "data_with_attr_val_1 %s " % data_with_attr_val_1
    #print "data_with_attr_val_0 %s " % data_with_attr_val_0
    leaf_with_class_0 = None
    leaf_with_class_1 = None

    if data_with_attr_val_0 == []:
        data_with_attr_val_0 = []
    elif len(data_with_attr_val_0[0]) - 1 == 1:
        pos_class_count, neg_class_count = count_postive_negative_classes(data_with_attr_val_0,target_attribute)
        if neg_class_count > pos_class_count:
            leaf_with_class_0 = 0
        else:
            leaf_with_class_1 = 1

    target_X = data[0][target_attribute]
    if data_with_attr_val_1 == []:
        data_with_attr_val_1 = []
    elif len(data_with_attr_val_1[0]) - 1 == 1:
        pos_class_count, neg_class_count = count_postive_negative_classes(data_with_attr_val_1,target_attribute)
        if neg_class_count > pos_class_count:
            leaf_with_class_0 = 0
        else:
            leaf_with_class_1 = 1

    attribute_list.remove(best_attribute)
    data_with_attr_val_1 = remove_best_attr_value(data_with_attr_val_1,best_attribute)
    data_with_attr_val_0 = remove_best_attr_value(data_with_attr_val_0, best_attribute)

    if len(data_with_attr_val_0) == 0 or len(data_with_attr_val_1) == 0:
        if leaf_with_class_0 is not None or leaf_with_class_1 is not None:
            if leaf_with_class_1 is None:
                leaf_with_class_1 = 1
            if leaf_with_class_0 is None:
                leaf_with_class_0 = 0
            node.left = id3([], [], target_attribute, heuristic,node.inc + '-', best_attribute,leaf_with_class_0, id=node.id)
            node.right = id3([], [], target_attribute, heuristic,node.inc + '+', best_attribute, leaf_with_class_1, id=node.id)
            return

        else:
            if len(data_with_attr_val_0) == 0:
                node.left = id3([], [], target_attribute, heuristic,node.inc + '-', best_attribute, 0, id=node.id)
                node.right = id3([], [],target_attribute, heuristic, node.inc + '+', best_attribute, target_X, id=node.id)
                return
            if len(data_with_attr_val_1) == 0:
                node.left = id3([], [], target_attribute, heuristic,node.inc + '-', best_attribute, target_X, id=node.id)
                node.right = id3([], [], target_attribute, heuristic,node.inc + '+', best_attribute, 1, id=node.id)
                return

    node.left = id3(data_with_attr_val_0, list(attribute_list), target_attribute, heuristic,node.inc + '-', best_attribute, id=node.id)
    node.right = id3(data_with_attr_val_1, list(attribute_list), target_attribute, heuristic,node.inc + '+', best_attribute, id=node.id)
    return node

def calculate_accuracy(root_node, file_name):
    #print root_node.attribute
    data_list, attributes = parse_csv_file(file_name)
    #data_list.insert(0, attributes)
    number_of_correct_predictions = 0
    class_index = len(attributes) - 1
    for each_list in data_list:
        current_node = CopyOfNode(root_node)
        while current_node is not None and current_node.attribute is not None:
            #print current_node.attribute
            res = each_list[attributes.index(current_node.attribute)]
            if res == 0:
                current_node = current_node.left
            else:
                current_node = current_node.right
        if current_node is not None and current_node.leaf == each_list[class_index]:
            number_of_correct_predictions += 1
    precision = float(number_of_correct_predictions) / (len(data_list))
    return precision


def print_tree(node,indent):
    if node is not None:
        if node.leaf is not None:
            print (str(node.leaf)),
        else:
            print ("\n" + indent + str(node.attribute) + " = " + "0 : "),
            print_tree(node.left, indent + "|\t")
            print ("\n" +indent + str(node.attribute) + " = " + "1 : "),
            print_tree(node.right, indent + "|\t")


def non_leaf_node_count(node):
    if node is None or node.left is None or node.right is None:
        return 0
    else:
        return 1 + non_leaf_node_count(node.left) + non_leaf_node_count(node.right)


def count_leaf_nodes(node, entry):
    global leaf_count
    if node is None:
        return
    if entry == 1:
        entry = 0
        leaf_count = [0,0]
    if node.leaf is not None:
        if node.leaf == 1:
            leaf_count[1] += 1
        else:
            leaf_count[0] += 1
    count_leaf_nodes(node.left, entry)
    count_leaf_nodes(node.right, entry)


def tree_pruning(root, L, K, data_file):
    D = CopyOfNode(root)
    Dbest = CopyOfNode(D)
    best_accuracy = round(calculate_accuracy(root, data_file) * 100, 2)
    initial_accuracy = best_accuracy
    for i in xrange(1, L):
        Ddash = CopyOfNode(D)
        M = random.randint(1, K)
        for j in range(1, M):
            N = non_leaf_node_count(Ddash)
            #print "Number of non-leaf nodes = %s" % N
            if N <= 1:
                break
            order_nodes(Ddash, 1)
            P = random.randint(1, N)
            remove_p_order_node(Ddash, P)
        temp_accuracy = round(calculate_accuracy(Ddash, data_file) * 100, 2)
        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            Dbest = CopyOfNode(D)
    return Dbest, best_accuracy , initial_accuracy


def order_nodes(node,entry):
    global node_order
    if entry == 1:
        node_order = 0
    if node is not None and node.leaf is None:
        node_order = node_order + 1
        node._updateOrder(node_order)
        order_nodes(node.left,0)
        order_nodes(node.right,0)
    return node


def remove_p_order_node(node,p):
    if node is not None and node.leaf is None:
        #print node.attribute, node.order
        if node.order == p:
            node.attribute = None
            left = node.left
            node.left = None
            del (left)
            right = node.right
            node.right = None
            del (right)
            count = [0, 0]
            count_leaf_nodes(node, count)
            if count[0] > count[1]:
                node.leaf = 0
            elif count[1] > count[0]:
                node.leaf = 1
            else:
                if node.inc.count('+') > node.inc.count('-'):
                    node.leaf = 1
                else:
                    node.leaf = 0
        else:
            remove_p_order_node(node.left,p)
            remove_p_order_node(node.right,p)


if __name__ == "__main__":
    import sys
    try:
        L, K , training_data, validation_data, test_data, to_print = sys.argv[1:]
    except Exception:
        sys.exit("Please provide the command line parameters of the form : main.py <L> <K> <training-set> <validation-set> <test-set> <to-print> \n where \n L: integer (used in the post-pruning algorithm) \n K: integer (used in the post-pruning algorithm) \n to-print:{yes,no} ")

    data_list, attribute_list = parse_csv_file(training_data)

    L = int(L)
    K = int(L)
    target_attribute = attribute_list[-1]

    data = []
    for each_list in data_list:
        data.append(dict(zip(attribute_list, each_list)))

    data2 = copy.deepcopy(data)
    attribute_list2 = attribute_list[:]

    print("**************************************************")
    print("DECISION TREE BASED ON INFORMATION GAIN")
    print("**************************************************")
    print ""

    print("Printing Pre-Pruned Tree and Accuracy")
    print("--------------------------------------------------")
    inc2 = '0'
    unpruned_tree_for_IG = id3(data2,attribute_list2,target_attribute,'gain',inc2,'root',id=0)
    if to_print.lower() == "yes":
        print_tree(unpruned_tree_for_IG,"")
        print ""
    unpruned_accuracy_IG = round(calculate_accuracy(unpruned_tree_for_IG, test_data) * 100, 2)
    print ("Accuracy of the model on the Test dataset = " + str(unpruned_accuracy_IG) + "%")

    print ""
    print ""

    print("Printing Post-Pruned Tree and Accuracy")
    print("--------------------------------------------------")
    pruned_tree_for_IG, best_accuracy, initial_accuracy = tree_pruning(unpruned_tree_for_IG,L,K,validation_data)
    if to_print.lower() == "yes":
        print_tree(pruned_tree_for_IG,"")
        print ""
    pruned_accuracy_IG = round(calculate_accuracy(pruned_tree_for_IG, test_data) * 100, 2)
    print ("Accuracy of the model on the Test dataset = " + str(pruned_accuracy_IG) + "%")




    print ("\n\n\n")
    print("**************************************************")
    print("DECISION TREE BASED ON VARIANCE IMPURITY")
    print("**************************************************")
    print ""

    print("Printing Pre-Pruned Accuracy and Tree")
    print("--------------------------------------------------")
    inc= '0'
    unpruned_tree_for_variance_impurity = id3(data,attribute_list,target_attribute,'variance_impurity',inc,'root',id=0)
    if to_print.lower() == "yes":
        print_tree(unpruned_tree_for_variance_impurity,"")
        print ""
    unpruned_accuracy_VI = round(calculate_accuracy(unpruned_tree_for_variance_impurity, test_data) * 100, 2)
    print ("Accuracy of the model on the Test dataset = " + str(unpruned_accuracy_VI) + "%")


    print ""
    print ""

    print("Printing Post-Pruned Tree and Accuracy")
    print("--------------------------------------------------")
    pruned_tree_for_VI, best_accuracy_VI,initial_accuracy_VI = tree_pruning(unpruned_tree_for_variance_impurity,L,K,validation_data)
    if to_print.lower() == "yes":
        print_tree(pruned_tree_for_VI,"")
        print ""
    pruned_accuracy_VI = round(calculate_accuracy(pruned_tree_for_VI, test_data) * 100, 2)
    print ("Accuracy of the model on the Test dataset = " + str(pruned_accuracy_VI) + "%")
