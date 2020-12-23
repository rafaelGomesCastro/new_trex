# -*- encoding: utf-8 -*-

import configparser, os


## Creates the files to feed the classifiers
#def write_simple(class_name, hist, method, filename, dim_images):
#    sci_path=get_sckit_filepath(dim_images,out=1)
#    f=open(sci_path+"/"+method+"/"+filename, "a+")
#
#    cnt = 1
#    text=class_name+" "
#    for i in range(len(hist)):
#        for j in range(len(hist[i])):
#            text += str(cnt)+':'+str(hist[i][j]) + ' '
#            cnt += 1
#    text += '\n'
#    f.write(text)
#    f.close

def split_dims(dim_lst):
    if ('[' in dim_lst): dim_lst = dim_lst.replace('[','')
    if (']' in dim_lst): dim_lst = dim_lst.replace(']','')
    if (',' in dim_lst): dim_lst = dim_lst.split(',')
    else: dim_lst = [dim_lst]

    return dim_lst

def write_training_scikit(class_name, hist, method, filename, class_img, dim_images):
    if hist == None:
        return
    sci_path=get_sckit_filepath(dim_images,out=1)
    f=open(sci_path+"/"+method+"/"+filename, "a+")

    #text=class_name+" "
    text=class_img+" "
    for i in range(len(hist)):
        text+=str(i+1)+":"+str(hist[i])+" "

    text+="\n"
    f.write(text)
    f.close


# Reads the configurations stored in definitions.ini
def get_definitions(section,option):
    c = configparser.ConfigParser()
    c.read("./definitions.ini")
    return c.get(section,option)

# Returns a dictionary for getting the host position in TM
def read_hosts(filename, pos=0):
    i=0
    hosts={}
    with open(filename, 'r') as f:
        for line in f:
            host = line.rstrip().split(None,1)[pos]
            hosts.update({host:i})
            i+=1
    return (hosts)


# Saves an image from a TM. If path is not specified, the image is shown
def tm_to_image(TM, maxb, path):
    im = Image.new("RGB", (len(TM), len(TM)))
    pix = im.load()
    if maxb < 0:
        print("Got a bad max_bytes.")
        return
    elif maxb == 0:
        return
    for x in range(len(TM)):
        for y in range(len(TM)):
            aux = int((TM[x][y]*255.0)/maxb)
            pix[x,y] = (aux,aux,aux)
    if path=="":
        im.show(pix)
    else:
        im.save(path+".png", "PNG")

# Prints the confusion matrix
def print_cm(cm):
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            #print(str(cm[i][j])+"\t", end=' ')
            print(str(cm[i][j])+"\t",end='')
        print('')

# Returns the path for scikit (or create if it does not exist)
def get_sckit_filepath(dim_images,out=0):
    if out == 0:
        sckit_filepath=get_definitions("Paths","sckit_files_path",dim_images)
    else:
        sckit_filepath=get_definitions("Paths","sckit_files_path_out",dim_images)
    if not os.path.exists(sckit_filepath):
        os.makedirs(sckit_filepath)
    return sckit_filepath

# Returns a dictionary of applications containing the paths for their TMs
def get_applications():
    apps = get_definitions("Dataset","classes")
    return apps.split(',')

# Returns the application, its class number and the times that class occurs
def parse_classes(dig):
    sp = dig.split("=")
    app = sp[0]
    class_num = sp[1]
    vls = sp[2].strip(" ").split(",")
    lst=[]
    for i in range(len(vls)):
        if (len(vls[i].split("-"))==2):
            for j in range(int(vls[i].split("-")[0]), int(vls[i].split("-")[1])+1):
                lst.append(j)
        else:
            lst.append(int(vls[i]))

    return [app, class_num, lst]


#from sklearn.grid_search import GridSearchCV

#def GridSearch(X_train, y_train, debug):
#    # set the parameters range
#    C_range = 2. ** np.arange(-5,15,2)
#    gamma_range = 2. ** np.arange(3,-15,-2)
#    k = [ 'rbf']
#    param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)
#
#    # generate the classifier 
#    srv = svm.SVC(probability=True)
#
#    # start the searching
#    grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=False)
#    grid.fit (X_train, y_train)
#
#    # get best estimator
#    model = grid.best_estimator_
#
#    # shows debug information on classifier
#    if debug:
#        print grid.best_params_
#    return model

