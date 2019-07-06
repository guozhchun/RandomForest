#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <omp.h>
using namespace std;

// the tree node
struct Node
{
	int featureIndex;   // save the node feature index to split if it is not a leaf
	int label;          // the node label if it is a leaf
	bool isLeaf;        // true when it is a leaf and false when it is not a leaf
	Node* leftChild;    // the node's left child, whose all child's feature value is less than the present node feature value
	Node* rightChild;   // the node's right child, whose all child's feature value is no less than the present node feature value
	Node()
	{
		leftChild = NULL;
		rightChild = NULL;
		label = -1;
		isLeaf = false;
		featureIndex = -1;
	}
};

vector<double> allModifyFeature;  // the global variable which saves the average feature value for every feature grouop
static omp_lock_t lock;           // the lock for variable when change it parallelly

// the function reading file from the filename
// return a two dimension vector like a matrix
vector<vector<double> >  readFile(string filename)
{
	vector<vector<double> > data;
	ifstream in(filename);
	if (in.is_open())
	{
		string line, number;
		vector<double> row;
		getline(in, line);
		while(getline(in, line))
		{
			istringstream istream(line);
			int count = 0;
			while(getline(istream, number, ','))
				if (count++ > 0)   // remove the first column
					row.push_back(atof(number.c_str()));
			data.push_back(row);
			row.clear();
		}
	}
	in.close();
	return data;
}

// calculate the average feature value for every feature group
// return a vector contain the average feature value which is calculated
vector<double> modifyFeature(const vector<vector<double> >& trainData)
{
	vector<double> allModifyFeature;
	int recordCount = trainData.size();
	for (int i = 0, length = trainData[0].size() - 1; i < length; i++)
	{
		double sum = accumulate(trainData[i].begin(), trainData[i].end(), 0.0); // calculate the sum of the feature group
		double avg = sum / recordCount;    // calculate the average for this feature group
		allModifyFeature.push_back(avg);
	}
	return allModifyFeature;
}

// the function choosing some records from the data randomly
// input a data which is a two dimension vector and the variable i is the seed which to make a random not influenced by time
// return a two dimension vector contain the records choosen from the data randomly
vector<vector<double> > chooseRow(vector<vector<double> > trainData, int i)
{
	srand(rand() % 100 + clock() + i);   // initialize the seed so that the random_shuffle is not influenced by time
	random_shuffle(trainData.begin(), trainData.end());  // shuffle the data randomly
	vector<vector<double> > records;
	for (int i = 0, length = trainData.size() * 0.7; i < length; i++)
		records.push_back(trainData[i]);
	return records;
}

// find the majority label in all labels
// input a data which contains the labels
// return a label whose number is the most of all labels
int findMostLabels(const vector<vector<double> >& records)
{
	int labels[27] = {0};
	for (int i = 0, length = records.size(); i < length; i++)
		labels[int(records[i].size() - 1)]++;
	int maxCount = 1, label = rand() % 26 + 1;
	for (int i = 1; i < 27; i++)
		if (labels[i] > maxCount)
		{
			maxCount = labels[i];
			label = i;
		}
	return label;
}

// judge the records contain only one label
// return true if it contains only one and the variable label is that one label
// and return false if it contains more than one label and the variable label is the first label in the records
bool isOneLabel(const vector<vector<double> >& records, int& label)
{
	label = int(records[0][records[0].size() - 1]);
	for (int i = 0, length = records.size(); i < length; i++)
		if (label != int(records[i][records[i].size() - 1]))
			return false;
	return true;
}

// choose some feature index from all feature index randomly
vector<int> chooseFeatureIndex(vector<int> featuresIndex)
{
	random_shuffle(featuresIndex.begin(), featuresIndex.end());
	vector<int> featIndex;
	for (int i = 0; i < 300; i++)
		featIndex.push_back(featuresIndex[i]);
	return featIndex;
}

// calculate the sum of a array
// input a pointor which is the array name and a lenght which is the array size
// return the sum of this array
int calSum(int* a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		sum += a[i];
	return sum;
}

// calculate the gini index in this child node
// input a array containing all labels
// return the gini index for this child node which is a double variable
double calGini(int* labels)
{
	int total = calSum(labels, 26);
	double valueSum = 0.0;

	if (total == 0)
		return 0.0;

	for (int i = 0; i < 26; i++)
		valueSum += double(labels[i]) / total * double(labels[i]) / total;

	return 1 - valueSum;
}

// calculate the split gini index for the feature in this node
// input: the variable records is all the data in this node
//        the variable featureIndex is the feature index whose feature is to split
// return the split gini index which is a double variable
double calSplitGini(const vector<vector<double> >& records, const int& featureIndex)
{
	int negLabels[26] = {0};
	int posLabels[26] = {0};
	int totals = records.size();
	int labelIndex = records[0].size() - 1;
	for (int i = 0; i < totals; i++)
	{
		if (records[i][featureIndex] < allModifyFeature[featureIndex])
			negLabels[int(records[i][labelIndex]) - 1]++;
		else
			posLabels[int(records[i][labelIndex]) - 1]++;
	}

	double negGini = calGini(negLabels);
	double posGini = calGini(posLabels);
	double splitGini = (double(calSum(negLabels, 26)) / totals) * negGini + (double(calSum(posLabels, 26)) / totals) * posGini;
	return splitGini;
}

// find the best feature index which will be split
// input: the variable records containing all the data, the variable featIndex containing all the feature index in this node
// return a feature index which has a smallest split gini index
int findBestFeatureIndex(const vector<vector<double> >& records, const vector<int>& featIndex)
{
	double minGini = 1.0, gini;
	int bestFeatureIndex = featIndex[0];
	for (int i = 0, length = featIndex.size(); i < length; i++)
	{
		gini = calSplitGini(records, featIndex[i]);
		if (minGini > gini)
		{
			minGini = gini;
			bestFeatureIndex = featIndex[i];
		}
	}
	return bestFeatureIndex;
}

// split the data into two sets, one is that all the feature values is less than this node feature value
// and the other one is that all the feature values is no less than this node feature value
// the variable records is all the records in this node which will be split
// the variable bestIndex is this node' feature index in all features
// the variable lessRecords will contain all the records whose feature value whose index is bestIndex is less than this node feature value
// the variable moreRecords will contain all the records whose featuer value whose index is bestIndex is no less than this node feature value
void splitData(const vector<vector<double> >& records, const int& bestIndex, vector<vector<double> >& lessRecords, vector<vector<double> >& moreRecords)
{
	lessRecords.clear();
	moreRecords.clear();
	for (int i = 0, length = records.size(); i < length; i++)
	{
		if (records[i][bestIndex] < allModifyFeature[bestIndex])
			lessRecords.push_back(records[i]);
		else
			moreRecords.push_back(records[i]);
	}
}

// create a decision tree
void createTree(vector<vector<double> > records, vector<int> featuresIndex, Node* tree)
{
	// the records have only one label
	int label = 1;
	if (isOneLabel(records, label))
	{
		//cout << "leaf" << endl;  // for debug
		tree->label = label;
		tree->isLeaf = true;
		return;
	}

	// choose some feature index randomly from all feature index
	vector<int> featIndex = chooseFeatureIndex(featuresIndex);

	// there is no feature choosen
	if (featIndex.empty())
	{
		//cout << "leaf" << endl;  // for debug
		int label = findMostLabels(records);
		tree->label = label;
		tree->isLeaf = true;
		return;
	}

	// find the best feature index to split
	int bestFeatureIndex = findBestFeatureIndex(records, featIndex);

	// split the data into two sets
	vector<vector<double> > lessRecords, moreRecords;
	splitData(records, bestFeatureIndex, lessRecords, moreRecords);

	// update this node's information
	tree->featureIndex = bestFeatureIndex;

	// make the tree group up left if its left child has data to split
	if (lessRecords.size() > 0)
	{
		//cout << "left " << tree->featureIndex << endl;  // for debug
		Node* leftNode = new Node();
		tree->leftChild = leftNode;
		createTree(lessRecords, featuresIndex, leftNode);
	}

	// make the tree group up right if its right child has data to split
	if (moreRecords.size() > 0)
	{
		//cout << "right " << tree->featureIndex << endl;  // for debug
		Node* rightNode = new Node();
		tree->rightChild = rightNode;
		createTree(moreRecords, featuresIndex, rightNode);
	}
}

// predict the label of the data with the decision tree
// input: the variable tree is the root node of the decisio tree and the variable data is the data to predict
// return a predict label of the data
int classify(Node* tree, const vector<double>& data)
{
	if (tree->isLeaf)
		return tree->label;

	int index = tree->featureIndex;
	if (data[index] < allModifyFeature[index])
		return classify(tree->leftChild, data);
	else
		return classify(tree->rightChild, data);
}

// predict the label of the data with the random forest
// input: the variable trees is a vector containing every root node of decision tree and the variable records is the data to predict
// return a vector containing the label of every record predicted by the random forest
vector<int> predict(const vector<Node*>& trees, const vector<vector<double> >& records)
{
	vector<int> results;
	for (int i = 0, length = records.size(); i < length; i++)
	{
		int labels[27] = {0};
		for (int j = 0, treeCount = trees.size(); j < treeCount; j++)
			labels[classify(trees[j], records[i])]++;

		// find the majority label from all labels predeicted by random forest
		int maxCount = -1, label = 1;
		for (int i = 1; i < 27; i++)
			if (labels[i] > maxCount)
			{
				maxCount = labels[i];
				label = i;
			}

		results.push_back(label);
	}
	return results;
}

// split the data into two sets, one is the records which don't have label and the other one is all the labels
// only useful for the data which has the label and wants to split
// the variable trainData is the data to split
// the variable allLabel will contain all the labels split from the trainData
// return a two dimension vector whose data is split from the trainData and the data don't have the labels
vector<vector<double> > delLabel(const vector<vector<double> >& trainData, vector<int>& allLabel)
{
	allLabel.clear();
	vector<vector<double> > result;
	for (int i = 0, length = trainData.size(); i < length; i++)
	{
		vector<double> temp;
		int count = trainData[0].size() - 1;
		for (int j = 0; j < count; j++)
			temp.push_back(trainData[i][j]);
		allLabel.push_back(int(trainData[i][count]));
		result.push_back(temp);
	}
	return result;
}

// print the correct predict rate on train data
// no use for test data
void outputCorrectRate(vector<int> result, vector<int> label)
{
	int count = 0, length = result.size();
	for (int i = 0; i < length; i++)
	{
		if (result[i] == label[i])
			count++;
		cout << result[i] << " = " << label[i] << endl;
	}
	cout << "The correct rate on train.csv is: " << double(count) / length << endl;
}

int main()
{
	clock_t startTime, endTime, endReadTime;

	startTime = clock();

	// read file
	vector<vector<double> > trainData = readFile("train.csv");
	vector<vector<double> > testData = readFile("test.csv");

	endReadTime = clock();

	allModifyFeature = modifyFeature(trainData);  // get the average feature value for every feature group

	// get all the feature index
	vector<int> featuresIndex;
	for (int i = 0, length = trainData[0].size() - 1; i < length; i++)
		featuresIndex.push_back(i);
	
	vector<Node*> trees;  // random forest containing the root node for every decision tree

	// create random forest parallelly
	// attention! this uses the OpenMP feature, and the compiler must support this feature
	omp_init_lock(&lock);  // initialize the lock
	#pragma omp parallel for num_threads(10)
	for (int i = 0; i < 10; i++)
	{
		vector<vector<double> > records = chooseRow(trainData, i);
		Node* tree = new Node();
		createTree(records, featuresIndex, tree);
	
		// for the threads, the variable trees is global and it is shared by the threads
		// set a lock when push a tree into a trees because of paralleling
		// so that every thread can change the variable trees independent
		omp_set_lock(&lock);    // lock the variable
		trees.push_back(tree);
		cout << "complete creating tree " << i << endl;
		omp_unset_lock(&lock);  // unlock the variable
	}
	omp_destroy_lock(&lock);  // destroy the lock
	cout << "complete creating Forest" << endl;
	
	// predict in the train data and print the predict correct rate
	vector<int> allLabel;
	vector<vector<double> > temp = delLabel(trainData, allLabel);
	vector<int> results = predict(trees, temp);
	outputCorrectRate(results, allLabel);

	// predict in the test data and output to the file
	vector<int> testResult = predict(trees, testData);
	ofstream outfile("output.csv", std::ios::out);
	outfile << "id,label" << endl;
	for (int i = 0, length = testResult.size(); i < length; i++)
		outfile << i << "," << testResult[i] << endl;

	endTime = clock();

	cout << "The read file time is: " << double(endReadTime - startTime) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "The creat forest and predict time is: " << double(endTime - endReadTime) / CLOCKS_PER_SEC << " seconds" << endl;

	return 0;
}