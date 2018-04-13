from PIL import Image
import numpy as np
from copy import deepcopy
import glob

def get_images(alpha,path='Machine Learning - Assignment 1/Assignment 1 Dataset/'):
    alphabet=alpha
    allimages=[]
    allimages_test=[]
    for id_char,x in enumerate(alphabet):
        for filename in glob.glob(path+'Train/A1'+str(x)+'*'):
            im=Image.open(filename)
            templ=list(im.getdata())
            templ.append(1)
            allimages.append([templ,id_char,filename.split('/')[-1]])    
        for filename in glob.glob(path+'Test/A1'+str(x)+'*'):
            im=Image.open(filename)
            templ=list(im.getdata())
            templ.append(1)
            allimages_test.append([templ,id_char,filename.split('/')[-1]])
    return allimages,allimages_test



def getx_train(training):
    return list(np.array(training)[:,0])
 


def getlabels(char_idx,training):
    labels=[-1]*len(training)
    for i in range(len(training)):
        if training[i][1]==char_idx:
            labels[i]=1            
    return labels



def check_classified(lista,labels,w):
    for i in range(len(lista)):
        if (np.matrix(w)*np.matrix(lista[i]).T)*labels[i] >=0:
            continue
        else:
            #print training_handwritten[i][2]
            return i
    
    return -1



def recalculate_w(lista,index,labels,w):
    w=w+0.05*np.array(lista[index])*labels[index]
    return w



def get_w(char_idx,training,x_train,w):
    labels=getlabels(char_idx,training)
    #print labels
    notdone=True
    while(notdone):
        misclass_id=check_classified(x_train,labels,w)
        if misclass_id != -1:
            w=recalculate_w(x_train,misclass_id,labels,w)
        else:
            notdone=False
    return w



def predict_char(x,w_list):
    max_val=None
    index=-1
    for i,w in enumerate(w_list):
        value=(np.matrix(w)*np.matrix(x).T)
        if  max_val is None:
            max_val=value
            index=i
        elif value >= max_val:
            max_val=value
            index=i			
    return index



def test_results(testing,w_list):
    test_result=[]
    for x in testing:
        result=predict_char(x[0],w_list)
        test_result.append([x[1],result])
    return test_result
    


def performance_calculator(results,alphabet):
    final_res=[]
    for i,_ in enumerate(alphabet[0:len(results)]):
        count_correct=0
        count_tot=0
        for r in results:
            if r[0]==i:
                if r[1]==r[0]:
                    count_correct+=1
                count_tot+=1
        final_res.append([count_correct, count_tot])
    return final_res



def which_misclassed(predictions,alphabet,testing):
	print ("Problematic predictions:")
	for k,i in enumerate(predictions):
		for j in range(len(i)):
		    i[j]=alphabet[i[j]]
		i.append(testing[k][2])
		if i[0]!=i[1]:
		    print ("The image: " +i[2]+" is of letter: " +i[0]+ ", however, it was predicted/seen as " +i[1])
	print ("Check Accuracy.jpg")


