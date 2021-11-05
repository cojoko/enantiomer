# Enantiomer

Search for chemical similarity in vector space

## Background

 One of the most challenging problems in computational chemistry involves the
search for chemically similar molecules. To determine whether a candidate
compound is worth investigating, it can be useful to first study better known,
structurally similar molecules. Since chemicals with shared structural features
often share certain properties, it is possible to infer the properties of a
theoretical compound if you have a substantial library of known molecules and
their features.

However, searching these libraries is a computationally intensive task. [Chemical space](https://en.wikipedia.org/wiki/Chemical_space) is vast, and
the ability to efficiently search this space is vital for medical chemists' ability to accurately infer the properties of many molecules for which no
experimental data exists. In this project, I borrow some techniques from
natural language processing (and machine learning, more generally) to explore
an alternative way to think about chemical search.

## Concepts

*Some useful terms and concepts for understanding this project.*

* [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system): A common format for representing molecules as text strings. Glucose, for instance, is represented as `OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1` in SMILES format.

* [Tanimoto Similarity](https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance): A method of quantifying the chemical similarity of a set of molecules. As the Tanimoto similarity of two molecules approaches one, those molecules are considered to be more structurally
similar, with a score of one indicating that the two molecules are identical. These scores are calculated between two chemical fingerprints.

* Chemical fingerprints: A bit-vector representation of a chemical, so just 1's
and 0's representing a chemical. RDkit's `chem` module allows us to convert
back and forth between SMILES and fingerprints easily.

## Approach

### Using NLP principles

The concepts of [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) and vector spaces is crucial in the study of
natural language processing. Put simply, one of the ways computational linguists
quantify semantic similarity is by building a high-dimensional vector, where each dimension is a single word's likelihood to appear around another word in a corpus of text. The basic premise of this technique is that semantically similar
words will often be found in semantically similar contexts. And since a word's co-occurrence rate with other words is represented as a vector, finding two words' similarity is as simple as calculating the distance between the two embedding vectors of those words. [Distance in high-dimensional](https://hlab.stanford.edu/brian/euclidean_distance_in.html) space is calculated similarly to distance in a 2 or 3 dimensional space, it's just impossible for us piddly humans to fully visualize.

One of the famous outcomes of this sort of vector representation is the sort of
semantic arithmetic that can be performed with the embeddings. Most famously,
many embeddings give results like W<sub>king</sub> - W<sub>man</sub> + W<sub>woman</sub> ends up closest to the word `queen` in vector space.

![vector math](https://github.com/cojoko/enantiomer/blob/main/images/vector_math.png)

*Some example embedding relationships projected into three dimensions*
<sub><[image source](https://developers.google.com/machine-learning/crash-course/embeddings/translating-to-a-lower-dimensional-space)></sub>

### K Nearest Neighbors

Using the vector space technique from the world of NLP, I will construct a high dimensional chemical space using fingerprints from a library of known compounds. Theoretical chemicals can then be plotted into this space, and it's nearest neighbors found. Ideally, similarity in the bits of a fingerprint will correspond to similarity in molecular structure, and therefore similarity in chemical properties.

In order to get the nearest neighbors of a point, I will employ a tool traditionally used for machine learning. [K nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a ML classification technique where you build labeled feature vectors into a vector space, then plot unclassed vectors into that space. By accessing
the K points closest to the unlabled vector, the neighbors "vote" on which class
the new point belongs to. We are not trying to classify data here, but we are
definitely interested in finding the K closest neighbors to a point in vector space! We will pass [Scikit-learn's NearestNeighbors module](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) a stack of fingerprints from every chemical in our library.

```python
# Convert SMILES to fingerprints
ref_mols["fp"] = ref_mols['smiles'].map(smile_to_fp)
 
# Create feature list from fingerprint column
X = ref_mols["fp"].tolist()

# Unsupervised learner for searching nearest neighbor in chemical space
neigh = NearestNeighbors(
  n_neighbors=n, metric="hamming", radius=2)

# Fit features to learner
neigh.fit(X)
```

Notice that NearestNeighbor takes a "metric" parameter, which specifies which [distance metric](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric) will be used to calculate distance between points. Rather than using a straight Euclidean distance, we will use a [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance), which specially suited for bit-vectors like our Tanimoto
fingerprints.

Now we need only to pass a list of fingerprints into the model's `kneighbors` function, and it will return us a list of K nearest neighbors per fingerprint along with a distance score for each neighbor.

```python
kneighbor_list = neigh.kneighbors(
        sample_test_fps, n_neighbors=n, return_distance=False)
```

The returned fingerprints need to be converted back to SMILES, and we have a
list of N closest chemicals to the theoretical molecule, as well as a score of HOW similar each chemical is.

## Time Complexity

In the worst case, using the most naive search / space structure, the searched
point would be compared against every point in vector space, resulting in a
time complexity of *O*(N). Optimizations to the vector space, such as [space partitioning](https://en.wikipedia.org/wiki/Space_partitioning) could
potentially be applied to the problem to greatly improve search time. Searching
N-dimensional space is a complex problem, but a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) could reduce time complexity to *O*(log N) were points evenly
distributed across the space.

## Running the demo

This project requires a Conda environment to use RDkit's chemistry tooling. To make setup easier, I have included a Dockerfile and docker-compose.yml which
will run the needed environment within a container. The dockerfile simply
executes the module as a script, comparing some unknown molecules against the demo chemical library in `demo_library.sdf`. To run, just make sure you have [Docker](https://docs.docker.com/engine/install/) installed on your machine, and run

```bash
$ docker-compose up --build
``` 

from within the
project directory to see the code in action.
