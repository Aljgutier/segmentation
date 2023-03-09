
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.style
import pandas as pd
import numpy as np
import seaborn as sns


import cartopy.crs as ccrs
import cartopy.feature as cfeat
# Grid Plot

from matplotlib import pyplot as plt

class gridplot:
    # Initializer / Instance Attributes
    def __init__(self, plotgrid=(1,1),wspace=0.1,hspace=0.1,figsize=(18,24)):
        # plotgrid Nrows x Ncol grid
        self.figsize=figsize 
        self.wspace=wspace
        self.hspace=hspace
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        self.plotgrid=plotgrid

        
    # instance method
    def ax(self,nthplot=1):
        ax = self.fig.add_subplot(self.plotgrid[0],self.plotgrid[1],nthplot)
        return ax



# Geo Plot States
# https://towardsdatascience.com/plotting-geospatial-data-with-cartopy-4b5ad0da0761




class geoplotstates:
    # Initializer / Instance Attributes
    def __init__(self, wspace=0.1,hspace=0.1,figsize=(18,24)):
        self.proj = ccrs.Stereographic(central_longitude=-95, central_latitude=35)
        self.figsize=figsize 
        self.wspace=wspace
        self.hspace=hspace
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

    # instance method
    def stateplot(self,subplot=(1,1,1),extent=[-120,-70,20,50]):

        state_borders = cfeat.NaturalEarthFeature(category='cultural', 
                                          name='admin_1_states_provinces_lakes',
                                          scale='50m', facecolor='none')
        
        proj = self.proj
        ax = self.fig.add_subplot(subplot[0],subplot[1],subplot[2],projection=proj)
        ax.add_feature(state_borders, edgecolor='black') # leave line as solid
        ax.add_feature(cfeat.BORDERS)  # probably won't show anything,
                                       #until coutry border down south
        ax.gridlines # lines of constant lat and long
        ax.set_extent(extent)  # U.S. 
        return ax



# Elbow Method

def elbowmethod(X,N):
    plt_style='seaborn'
    matplotlib.style.use(plt_style)
    wcss = [] # WCSS is defined as the sum of the squared distance between each member of the cluster and its centroid.
    Y=X.copy()
    #N=15 # Try clusters from 0 to N 
    K=np.zeros((N,N))  # matrix to keep a count of cluster membership

    for i in range(1,N+1):   # this goes from 1 to N
    	print(i,'.. ',end='', sep='')
    	kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    	kmeans.fit(Y)
    	wcss.append(kmeans.inertia_)
    	Y['clabel']=kmeans.labels_
    	K[i-1,0:i]=(Y[[Y.columns[0],'clabel']].groupby('clabel').count().T)  # count cluster membership

    color = sns.color_palette()
    sns.set_palette("bright")  # bright colors
    sns.set_style("ticks")
    sns.set_style("darkgrid",  {"axes.facecolor": ".9"}) # grey background with grid
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, axs = plt.subplots(figsize=(14,8), nrows=1, ncols=1)
    sns.set_style("ticks", {"xtick.major.size" : 2, "ytick.major.size" : .1})
    
    plt.plot(range(1, N+1), wcss)  # note, range goes from 1 to N-1 (so add 1)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()  # force plot output at this point of code, rather than after next set of code
            # default forced by Jupyter notebook at the end of the cell ... after "display(k)"


    #figsize=[14,6]
    #rcParams['figure.figsize'] = figsize
    print('cluster sizes')
    df_K=pd.DataFrame(data=K, columns=range(1,N+1))
    display(df_K)

    Y.drop(columns='clabel',inplace=True)
    del Y
    return


# Silhouette Plots Function
# Reference
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def silhouetteplots(X,range_n_clusters,figsize=(18,18)):
    

    plt_style='seaborn'
    matplotlib.style.use(plt_style)
    N=len(range_n_clusters)+1
    grdp=gridplot(plotgrid=(int(N/2 + 0.5),2),figsize=figsize,hspace=0.2)
    for n_clusters,k in zip(range_n_clusters,range(1,N)):
    # Create a subplot with 1 row and 2 columns
    #fig, ax1 = plt.subplots(1, 1)
    #fig.set_size_inches(7, 5)  # x , y
        print(n_clusters,'.. ',end='', sep='')

        ax1=grdp.ax(k)
    
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.25, 1]
        ax1.set_xlim([-0.25, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        #print("For n_clusters =", n_clusters,
        #      "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        #ax1.set_title("The silhouette plot for the various clusters.")
        
        ax1.set_ylabel("Cluster Label")
        
        if k>=N-2:
            ax1.set_xlabel("Silhouette Coefficient")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


# Cluster Segments

def KmeansCluster(X,N,verbose=0):
    kmeans = KMeans(n_clusters=N, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    Y = X.copy()
    Y['Cluster Label']=kmeans.labels_
    # km_labels = kmeans.labels_
    #Y.columns = [['clabel','count']]
    
    ## kmeans labels
    #km_labels=kmeans.labels_

    ClusterLabelCounts=Y[[X.columns[0],'Cluster Label']].groupby('Cluster Label').count()
    ClusterLabelCounts.columns=['Count']

    if verbose == 1:
        print('\nCluster Label Counts:')
        display(ClusterLabelCounts.T)
    
    Y.drop(columns='Cluster Label',inplace=True)

    # numpy array  ... cluster centers in array format
    centers=kmeans.cluster_centers_
    a=np.array([Y.columns])
    b=np.concatenate((a,centers),axis=0)

    # data frame
    df_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=Y.columns)
    if verbose==1:
        print('Cluster Centers', end='')
        display(df_centers)
    del Y
    return kmeans.labels_,kmeans.cluster_centers_,df_centers
    

# Cluster Heatmap
# https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
#corder=(0,7,5,6,9,8,1,2,3,4)
def clusterheatmap(df_centers,cmap='OrRd',clabels= None, figsize= (14,14), **kwargs): 


    df_cols = df_centers.columns
    fig = plt.figure(figsize=figsize)
    plt_style='seaborn'
    matplotlib.style.use(plt_style)
    g=fig.ax = sns.heatmap(df_centers,cmap="OrRd", 
                           yticklabels=df_centers.index,
                           annot=True, annot_kws={'size': 12})
    
    g.set_xticklabels(g.get_xticklabels(), rotation=80, horizontalalignment='right', size=16)
    g.set_yticklabels(g.get_yticklabels(), size=16)
    plt.show()    
    
    

