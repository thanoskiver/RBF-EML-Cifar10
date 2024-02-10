import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers,models
from keras.optimizers import SGD
import os
from sklearn.cluster import KMeans
np.random.seed(42)

def showFalseTrueClassifications(y_pred,y_true):
    """
    Η συναρτηση δημιουργεί δυο λιστες με τα παραδειγματα που το μοντελο ταξινομησε σωστα και λανθασμενα.
    Στην συνέχεια επιλέγει τυχαία 5 δειγματα απο καθε μία λιστα για να τα εμφανισει στον χρήστη με το καταλληλο μηνυμα. 
    """

    trueValues={0:2,1:1}
    i=0
    trueList=[]
    falseList=[]
    while (i<y_pred.shape[0]):
        if(y_pred[i]==y_true[i]):
            transformedV=y_true[i]
            #print("Το δειγμα "+str(i)+" κατηγοριοποιήθηκε σωστα ως "+str(trueValues[transformedV]))
            trueList.append([i,trueValues[transformedV]])
        if(y_pred[i]!=y_true[i]):
            transformedV_True=y_true[i]
            transformedV_Pred=y_pred[i]
            #print("Το δειγμα "+str(i)+" κατηγοριοποιήθηκε λανθασμένα ως "+str(trueValues[transformedV_Pred])+" ενώ ηταν "+str(trueValues[transformedV_True]) )
            falseList.append([i,trueValues[transformedV_Pred],trueValues[transformedV_True]])

        i+=1
    random_indices_ofTrue = np.random.choice(len(trueList), size=5, replace=False)
    random_indices_ofFalse = np.random.choice(len(falseList), size=5, replace=False)

    for i in random_indices_ofTrue:
        print("Το δειγμα "+str(trueList[i][0])+" κατηγοριοποιήθηκε σωστα ως "+str(trueList[i][1]))
    for i in random_indices_ofFalse:
        print("Το δειγμα "+str(falseList[i][0])+" κατηγοριοποιήθηκε λανθασμένα ως "+str(falseList[i][1])+" ενώ ήταν "+str(falseList[i][2]))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getTheData(filename):
    file_path = os.path.join("cifar-10-batches-py", filename)
    dict=unpickle(file_path)
    lista=list(dict.keys())
    Y =dict[lista[1]]
    X=dict[lista[2]]
    X=X/255
    return np.array(X),np.array(Y)

def gatherTheData():
    x_test,y_test=getTheData("test_batch")
    x_train=np.random.random((0,3072))
    y_train=np.random.random((0))

    for i in range (1,6):
        TEMPX,TEMPY=getTheData("data_batch_"+str(i))
        x_train=np.concatenate([x_train,TEMPX],axis=0)
        y_train=np.concatenate([y_train,TEMPY],axis=0)
    return x_train,y_train,x_test,y_test

def calculateDistance(x,centers):
    """
    Υπολογιζει εναν πινακα  Πλήθος εισόδου επί Πλήθος Κέντρων.
    Κάθε resultList[i][j] δηλώνει την απόσταση του j απο το i
    """
    resultList=[]
    for i in range(centers.shape[0]):
        distances = np.linalg.norm(x - centers[i], axis=1)
        resultList.append(distances)
    resultArray=np.array(resultList)
    return resultArray.T
def polysqrFunction(norm,s):
     """
     norm->πινακας Πληθος εισοδων επι κεντρα. Καθε στοιχειο norm[i][j] δηλωνει την ευκλειδια αποσταση του i απο το κεντρο j
     s->s[i]=το σ του κεντρου ι
     Τύπος της πολυτετραγωνικης συναρτησης.
     """
     polysqrResult=(np.sqrt(norm**2+s**2))
     return polysqrResult


def gauss(norm,s):
    """
    υπολογισμός της αποστάσης κάθε δείγματος από κάθε κέντρο και εφαρμογή αυτης της τιμής σε
    gauss function 
    επιστρέφει πλήθος δειγμάτων επι πλήθος κέντρων πινακα.
    """
    gaussResult = np.zeros((norm.shape[0], s.shape[0]))
    for j in range(s.shape[0]):
        if s[j] != 0:
            gaussResult[:, j] = np.exp(-((norm[:, j]**2) / (s[j]**2)))

    return gaussResult
    
x_train,y_train,x_test,y_test=gatherTheData()

def plotModel(history):
    """
    Μέθοδος που τυπώνει στην οθονη τα στατιστικα ενός μοντέλου κατα την εκπαιδευση του.(απωλεια και ακριβεια στα δεδομενα ελεγχου και εκπαιδευσης)
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Απώλεια συνόλου Εκπαίδευσης και Ελέγχου')
    plt.xlabel('Εποχή')
    plt.ylabel('Απώλεια')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Ακρίβεια συνόλου Εκπαίδευσης και Ελέγχου')
    plt.xlabel('Εποχή')
    plt.ylabel('Ακρίβεια')
    plt.legend()

    plt.tight_layout()
    plt.show()




def pick2classes(x,y):
    """
    x->παραδειγματα εισόδου
    y->στοχοι παραδειγμάτων
    Επιλέγει τις κλασεις 1 και 2 απο τις συνολικα 10 κλασεις του αρχικου συνολου και αποθηκευει
    τα δειγματα τους στους πινακες finalDatasetOfX και finalDatasetOfY
    """
    finalDatasetOfX=[]
    finalDatasetOfY=[]
    for i in range (x.shape[0]):
        if y[i]==1 or y[i]==2:
            finalDatasetOfX.append(x[i]) 
            finalDatasetOfY.append(y[i]%2)#0 h klasi 2 ,1 h klasi 1
    finalDatasetOfX=np.array(finalDatasetOfX)
    finalDatasetOfY=np.array(finalDatasetOfY)
    return finalDatasetOfX,finalDatasetOfY
x_train,y_train=pick2classes(x_train,y_train)
x_test,y_test=pick2classes(x_test,y_test)

y_test=keras.utils.to_categorical(y_test)
y_train=keras.utils.to_categorical(y_train)







"""
centers->κεντρα
x->δειγματα
Μια μέθοδος που ,αφού επιλεχθουν τα κεντρα, υπολογίζει σε ποια ομάδα ανήκουν τα δείγματα.
Επιστρέφει τον πινακα assignments[ι]=ομαδα_του_ι

"""
def findAssignments(centers,x):
    num_points = x.shape[0]
    num_centers = centers.shape[0]
    assignments = np.zeros((num_points,), dtype=int)
    for c in range(num_centers):
        distances = np.linalg.norm(x-centers[c],axis=1)
        mask = distances < np.linalg.norm(x-centers[assignments],axis=1)
        assignments[mask] = c
    return assignments


def randomCenters(n_centers,x_train):
    """
    n_centers->το πληθος των κέντρων που θέλουμε να βρούμε
    x_train->το συνολο δεδομένων που αναλύουμε
    Μια συνάρτηση που επιλέγει τυχαία n_centers απο τα δειγματα x_train
    """
    num_rows = x_train.shape[0]
    random_indices = np.random.choice(num_rows, size=n_centers, replace=False)
    random_centers = x_train[random_indices]
    center_assignments= findAssignments(random_centers,x_train)
    return random_centers,center_assignments

def KMEANS_Clusters(n_centers,x_train):
    """
    Μια συναρτηση κέλυφος για την ευρεση κέντρων μεσω της K-MEANS και των ομάδων που ανηκεί κάθε δειγμά
    n_centers->το πληθος των κέντρων που θέλουμε να βρούμε
    x_train->το συνολο δεδομένων που αναλύουμε

    """
    km = KMeans(n_clusters=n_centers, verbose=0,n_init=10)
    km.fit(x_train)
    clusters= km.cluster_centers_
    cluster_assignments = km.labels_
    return clusters,cluster_assignments

def sigma(cluster_assignments,centers,x):
    """
    Για κάθε κεντρο i αυτη η συναρτηση υπολογιζει το σ του.
    σ=(μεγιστη αποσταση μελους ομαδας απο το κεντρο)/τετραγωνικη_ριζα(2*πλήθος ομαδας) 

    cluster_assignments->ενας μονοδιαστατος πινακας οπου ισχυει cluster_assignments[ι]=η ομαδα που ανηκει το δειγμα ι
    centers -> κεντρα που επιλέχθηκαν
    x->τα δείγματα
    """
    no_clusters=centers.shape[0]
    max_d_of_clusters=np.zeros((no_clusters,))
    population_of_clusters=np.zeros((no_clusters,))
    for i in range (cluster_assignments.shape[0]):
        cluster_index=cluster_assignments[i]
        population_of_clusters[cluster_index]=population_of_clusters[cluster_index]+1
        distance=np.linalg.norm(x[i] - centers[cluster_index])

        if (max_d_of_clusters[cluster_index]<distance):
           max_d_of_clusters[cluster_index]=distance

    sigmaArray=np.zeros((no_clusters,))
    for i in range (no_clusters):
        sigmaArray[i]=max_d_of_clusters[i]/(np.sqrt(2*population_of_clusters[i]))
    

    return sigmaArray

def linearModelTraining(finalTrain_X,x_test_final,epoch):
    """
    finalTrain_X->συνολο δεδομενων εισοδου που το μοντελο θα εκπαιδευτει
    x_test_final->συνολο δεδομενων εισοδου που το μοντελο θα ελεγχθει
    epoch->πλήθος εποχων που θα εκπαιδευτει το μοντέλο
    Η συνάρτηση αφορά ενα μοντέλο που δέχεται ως είσοδο τις ομοιότητες των δειγμάτων με κάθε κεντρο.
    (δηλαδη εναν πινακα Ν επι πλήθος κεντρα)
    Το στρώμα εισόδου συνδέεται πλήρως με 2 νευρώνες εξόδου που αξιοποιουν την λογιστικη συναρτηση 
    και ταξινομούν το δειγμα ως μία εκ των δύο κλασεων
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(finalTrain_X.shape[1],)))
    model.add(layers.Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(optimizer=SGD(learning_rate=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
    h=model.fit(finalTrain_X,y_train,epochs=epoch,validation_data=(x_test_final,y_test),batch_size=35,verbose=0)
    test_precision=model.evaluate(x_test_final,y_test,verbose=0)[1]
    train_precision=model.evaluate(finalTrain_X,y_train,verbose=0)[1]
    print("Το μοντέλο στα δεδομένα ελέγχου πετυχαίνει ακρίβεια "+str(test_precision))
    print("Το μοντέλο στα δεδομένα εκπαίδευσης πετυχαίνει ακρίβεια "+str(train_precision))
    y_pred=model.predict(x_test_final,verbose=0)
    showFalseTrueClassifications(np.argmax(y_pred,axis=1),np.argmax(y_test,axis=1))
    print("")
    plotModel(h)
    return train_precision,test_precision
def extraLayerlinearModelTraining(finalTrain_X,x_test_final,epoch):
    """
    finalTrain_X->συνολο δεδομενων εισοδου που το μοντελο θα εκπαιδευτει
    x_test_final->συνολο δεδομενων εισοδου που το μοντελο θα ελεγχθει
    epoch->πλήθος εποχων που θα εκπαιδευτει το μοντέλο
    Η συνάρτηση αφορά ενα μοντέλο που δέχεται ως είσοδο τις ομοιότητες των δειγμάτων με κάθε κεντρο.
    (δηλαδη εναν πινακα Ν επι πλήθος κεντρα)
    Το στρώμα εισόδου συνδέεται πλήρως με ένα layer που καθε φορά θα εχει τους μισους νευρώνες απο όσους υπαρχει στην εισοδο
    και αυτο θα συνδέεται πλήρος με το τελικό layer που εχει 2 sigmoid νευρωνες για την ταξινομηση των δειγματων 
    και ταξινομούν το δειγμα ως μία εκ των δύο κλασεων
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(finalTrain_X.shape[1],)))
    model.add(layers.Dense(finalTrain_X.shape[1]/2, activation='relu'))
    model.add(layers.Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(optimizer=SGD(learning_rate=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
    h=model.fit(finalTrain_X,y_train,epochs=epoch,validation_data=(x_test_final,y_test),batch_size=35,verbose=0)
    test_precision=model.evaluate(x_test_final,y_test,verbose=0)[1]
    train_precision=model.evaluate(finalTrain_X,y_train,verbose=0)[1]
    print("Το μοντέλο στα δεδομένα ελέγχου πετυχαίνει ακρίβεια "+str(test_precision))
    print("Το μοντέλο στα δεδομένα εκπαίδευσης πετυχαίνει ακρίβεια "+str(train_precision))
    y_pred=model.predict(x_test_final,verbose=0)
    showFalseTrueClassifications(np.argmax(y_pred,axis=1),np.argmax(y_test,axis=1))
    print("")
    plotModel(h)
    return train_precision,test_precision
def rbfNETGaussBased(title,clusters,sigmaArray,x_train_distance_from_center,x_test_distance_from_center):
    """
    title->Αναγνωριστικο της μεθόδου επιλογης κέντρων
    clusters->κέντρα
    sigmaArray->σ κεντρων
    x_train_distance_from_center->ευκλειδια αποσταση του καθε δειγματος εκπαιδευσης απο καθε κεντρο
    x_test_distance_from_center->ευκλειδια αποσταση του καθε δειγματος ελεγχου απο καθε κεντρο
    Αναπαριστα την διαδικασια εκπαιδευσης ενος RBF NN που χρησιμοποιει την gauss function στους radial νευρωνες και 2 γραμμικους νευρωνες
    εξόδου
    """
    print("Αλγόριθμος Επιλογής Κέντρών: "+title+", πλήθος κέντρων: "+str(clusters.shape[0])+" και συναρτηση ενεργοποίησης Gauss")
    finalTrain_X=gauss(x_train_distance_from_center,sigmaArray)
    x_test_final=gauss(x_test_distance_from_center,sigmaArray)
    epoch=400
    train_precision,test_precision=linearModelTraining(finalTrain_X,x_test_final,epoch)
    return train_precision,test_precision
def rbfNETPolySqrBased(title,clusters,sigmaArray,x_train_distance_from_center,x_test_distance_from_center):
    """
    title->Αναγνωριστικο της μεθόδου επιλογης κέντρων
    clusters->κέντρα
    sigmaArray->σ κεντρων
    x_train_distance_from_center->ευκλειδια αποσταση του καθε δειγματος εκπαιδευσης απο καθε κεντρο
    x_test_distance_from_center->ευκλειδια αποσταση του καθε δειγματος ελεγχου απο καθε κεντρο
    Αναπαριστα την διαδικασια εκπαιδευσης ενος RBF NN που χρησιμοποιει την πολυτετραγωνικη function στους radial νευρωνες και 2 γραμμικους νευρωνες
    εξόδου
    """
    print("Αλγόριθμος Επιλογής Κέντρών: "+title+", πλήθος κέντρων: "+str(clusters.shape[0])+" και συναρτηση ενεργοποίησης Πολυτετραγωνική")
    finalTrain_X=polysqrFunction(x_train_distance_from_center,sigmaArray)
    x_test_final=polysqrFunction(x_test_distance_from_center,sigmaArray)
    epoch=400
    train_precision,test_precision=linearModelTraining(finalTrain_X,x_test_final,epoch)
    return train_precision,test_precision
def rbfNETPolySqrBasedWithExtraLayer(title,clusters,sigmaArray,x_train_distance_from_center,x_test_distance_from_center):
    """
    title->Αναγνωριστικο της μεθόδου επιλογης κέντρων
    clusters->κέντρα
    sigmaArray->σ κεντρων
    x_train_distance_from_center->ευκλειδια αποσταση του καθε δειγματος εκπαιδευσης απο καθε κεντρο
    x_test_distance_from_center->ευκλειδια αποσταση του καθε δειγματος ελεγχου απο καθε κεντρο
    Αναπαριστα την διαδικασια εκπαιδευσης ενος RBF NN που χρησιμοποιει την πολυτετραγωνικη function στους radial νευρωνες και 2 επιπέδα γραμμικων νευρωνων
    εξόδου
    """
    print("Αλγόριθμος Επιλογής Κέντρών: "+title+", πλήθος κέντρων: "+str(clusters.shape[0])+" και συναρτηση ενεργοποίησης Πολυτετραγωνική")
    finalTrain_X=polysqrFunction(x_train_distance_from_center,sigmaArray)
    x_test_final=polysqrFunction(x_test_distance_from_center,sigmaArray)
    epoch=400
    train_precision,test_precision=extraLayerlinearModelTraining(finalTrain_X,x_test_final,epoch)
    return train_precision,test_precision
def dictionaryPlot(title,data_dict):
    """
    Δεχεται ενα λεξικο με μια λιστα 2 στοιχειων και δημιουργει το γραφημα μιας συναρτησης οπου ανεξάρτητη μεταβλητη ειναι 
    η τιμη του λεξικου και εξαρτημένες οι τιμες μεσα στην λιστα
    """
    plt.title(title)
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    column_index = 0
    column_vector0 = [row[column_index] for row in values]
    column_index = 1
    column_vector = [row[column_index] for row in values]  
    plt.plot(keys, column_vector0, color='red', label='Δεδομένα Εκπαίδευσης')
    plt.plot(keys, column_vector, color='blue', label='Δεδομένα Ελέγχου')
    plt.xlabel('Αριθμός Κέντρων')
    plt.ylabel('Τιμές Ακρίβειας')
    plt.legend()
    plt.show()
progress_of_Kmeans_poly={}
progress_of_Kmeans_gauss={}
progress_of_Kmeans_poly_extra_layer={}
file_paths = ['C450.npz','C650.npz','C850.npz','C1250.npz','C1450.npz','C1650.npz','C1850.npz','C2500.npz']
#τρεχει μια for loop για κάθε αριθμο κεντρων που θελουμε να τροφοδοτήσουμε την συναρτηση k_means
#για εξοικονόμηση χρόνου αποθηκευτηκαν τα κεντρα που ηρθαν ως αποτελεσμα απο την k_means
for file_path in file_paths:

    if os.path.exists(file_path):
        loaded_data = np.load(file_path)
        clusters=loaded_data['array1']
        cluster_assignments=loaded_data['array2']
        sigmaArray=loaded_data['array3']
        
    else:
        if file_path=='C2500.npz':
             clusters,cluster_assignments= KMEANS_Clusters(2500,x_train)
             sigmaArray=sigma(cluster_assignments,clusters,x_train)
        
        if file_path=='C1650.npz':
             clusters,cluster_assignments= KMEANS_Clusters(1650,x_train)
             sigmaArray=sigma(cluster_assignments,clusters,x_train)

        
        if file_path=='C1850.npz':
            clusters,cluster_assignments= KMEANS_Clusters(1850,x_train)
            sigmaArray=sigma(cluster_assignments,clusters,x_train)
        
        if file_path=='C450.npz':
            clusters,cluster_assignments= KMEANS_Clusters(450,x_train)
            sigmaArray=sigma(cluster_assignments,clusters,x_train)
        if file_path=='C500.npz':
             clusters,cluster_assignments= KMEANS_Clusters(500,x_train)
             sigmaArray=sigma(cluster_assignments,clusters,x_train)
        
        if file_path=='C650.npz':
             clusters,cluster_assignments= KMEANS_Clusters(650,x_train)
             sigmaArray=sigma(cluster_assignments,clusters,x_train)
        
        if file_path=='C850.npz':
             clusters,cluster_assignments= KMEANS_Clusters(850,x_train)
             sigmaArray=sigma(cluster_assignments,clusters,x_train)
        
        if file_path=='C1250.npz':
             clusters,cluster_assignments= KMEANS_Clusters(1250,x_train)
             sigmaArray=sigma(cluster_assignments,clusters,x_train)
        
        if file_path=='C1450.npz':
             clusters,cluster_assignments= KMEANS_Clusters(1450,x_train)
             sigmaArray=sigma(cluster_assignments,clusters,x_train)
        np.savez(file_path,array1=clusters,array2=cluster_assignments,array3=sigmaArray)
    #υπολογίζονται οι αποστασεις των x_train και x_test απο τα κεντρα
    distanceOfX_train=calculateDistance(x_train,clusters)
    distanceOfX_test=calculateDistance(x_test,clusters)


    train_acc,test_acc=rbfNETPolySqrBased("K_means ",clusters,sigmaArray,distanceOfX_train,distanceOfX_test)
    progress_of_Kmeans_poly[clusters.shape[0]]=[train_acc,test_acc]

    train_acc,test_acc=rbfNETPolySqrBasedWithExtraLayer("K_means με 2 γραμμικά επίπεδα",clusters,sigmaArray,distanceOfX_train,distanceOfX_test)
    progress_of_Kmeans_poly_extra_layer[clusters.shape[0]]=[train_acc,test_acc]

    if clusters.shape[0] in [450,850,1250,1650,1850]:
        train_acc,test_acc=rbfNETGaussBased("K_means ",clusters,sigmaArray,distanceOfX_train,distanceOfX_test)
        progress_of_Kmeans_gauss[clusters.shape[0]]=[train_acc,test_acc]
#εκτυπωση της προοδου των μοντέλων συναρτησει του πλήθουν των κρυφών νευρωνων
dictionaryPlot("Πρόοδος των Κ_Μέσων με πολυτετραγωνική",progress_of_Kmeans_poly)
dictionaryPlot("Πρόοδος των Κ_Μέσων με πολυτετραγωνική και ένα επιλέον γραμμικό επίπεδο",progress_of_Kmeans_poly_extra_layer)
dictionaryPlot("Πρόοδος των Κ_Μέσων με gauss",progress_of_Kmeans_gauss)


progress_of_RandomCenters_poly={}
progress_of_RandomCenters_poly_extraLayer={}
progress_of_RandomCenters_gauss={}
dictionaryPlot("Πρόοδος των Τυχαίων Κέντρων με Πολυτετραγωνικη",progress_of_RandomCenters_poly)
for no_centers in [200,500,1200,1800,2500,3000]:   


    randomClusters,randomAssingments=randomCenters(no_centers,x_train)

    sigmaRandom=sigma(randomAssingments,randomClusters,x_train)
    distanceOfX_test=calculateDistance(x_test,randomClusters)
    distanceOfX_train=calculateDistance(x_train,randomClusters)
    a,b=rbfNETPolySqrBased("random centers",randomClusters,sigmaRandom,distanceOfX_train,distanceOfX_test)
    progress_of_RandomCenters_poly[no_centers]=[a,b]


    randomClusters,randomAssingments=randomCenters(no_centers,x_train)

    sigmaRandom=sigma(randomAssingments,randomClusters,x_train)
    a,b=rbfNETPolySqrBasedWithExtraLayer("random centers",randomClusters,sigmaRandom,distanceOfX_train,distanceOfX_test)
    progress_of_RandomCenters_poly_extraLayer[no_centers]=[a,b]

    if no_centers in [200,500,1200,1800,2500]:
        randomClusters,randomAssingments=randomCenters(no_centers,x_train)

        sigmaRandom=sigma(randomAssingments,randomClusters,x_train)
        a,b=rbfNETGaussBased("random centers",randomClusters,sigmaRandom,distanceOfX_train,distanceOfX_test)
        progress_of_RandomCenters_gauss[no_centers]=[a,b]

dictionaryPlot("Πρόοδος των Τυχαίων Κέντρων με Πολυτετραγωνικη",progress_of_RandomCenters_poly)
dictionaryPlot("Πρόοδος των Τυχαίων Κέντρων με Πολυτετραγωνικη και ένα επιλέον γραμμικό επίπεδο",progress_of_RandomCenters_poly_extraLayer)    
dictionaryPlot("Πρόοδος των Τυχαίων Κέντρων με Gauss",progress_of_RandomCenters_gauss)