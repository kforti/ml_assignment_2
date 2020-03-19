from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.cluster import KMeans


def clustering(vectorizer, num_clusters):
    X = vectorizer.fit_transform(twenty_train.data)

    km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    print(metrics.homogeneity_score(labels, km.labels_), metrics.completeness_score(labels, km.labels_), metrics.v_measure_score(labels, km.labels_),
          metrics.adjusted_rand_score(labels, km.labels_), metrics.silhouette_score(X, km.labels_, sample_size=1000), sep=',')
    print()

    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()


def naive_bayes(pipeline, train_data, test_data):
    pipeline.fit(train_data.data, train_data.target)

    predicted = pipeline.predict(test_data.data)
    print("Naïve Bayes Classifier Accuracy: ", np.mean(predicted == test_data.target))



if __name__ == '__main__':
    # data and params
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True,
                                      random_state=42)
    twenty_test = fetch_20newsgroups(subset='test',
                                     shuffle=True,
                                     random_state=42)

    labels = twenty_train.target
    true_k = np.unique(labels).shape[0]
    num_clusters = true_k

    # transformers and pipelines
    tfidf_transformer = TfidfTransformer()
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=2000000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    count_vect = CountVectorizer()
    text_clf = Pipeline([
        ('vect', count_vect),
        ('tfidf', tfidf_transformer),
        ('clf', MultinomialNB()),
        ])

    clustering(vectorizer, num_clusters)



    naive_bayes(pipeline=text_clf, train_data=twenty_train, test_data=twenty_test)