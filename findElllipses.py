import numpy as np
import cv2
import os
import sys 
import math
import argparse
#import scipy

verbose = False
visualize = False


def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

#Clase para definir una elipse
#a: eje mayor
#b: eje menor
#h: coordenada x del centro
#k: coordenada y del centro
class Ellipse(object):
    
    def __init__(self, a, b, h, k, rot):
        self.a = a
        self.b = b
        self.rot = rot
        self.h = h
        self.k = k

    def getArea(self):
        return math.pi*self.a*self.b
    
    def __repr__(self):
        return '{}: a {} b {} h {} k {} rot {}'.format(self.__class__.__name__, self.a, self.b, self.h, self.k, self.rot)

def rad2deg(rads):
    return rads*(180.0/math.pi)

#resuelve un sistema de la forma Ax = 0 con SVD
def null(A, eps=1e-12):
    u, s, vh = np.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = np.compress(null_mask, vh, axis=0)
    return np.transpose(null_space)

#ajusta una ellipse usando 5 puntos
def adjust_ellispe(points):
    A = np.zeros(shape=[5, 6])
    for i in range(5):
        x = points[i, 0]
        y = points[i, 1]
        A[i, :] = np.array([x**2, x*y, y**2, x, y, 1])

    ns = null(A)
    #print ns
    rank = np.linalg.matrix_rank(ns)
    #print("rank",rank)
    if(rank) < 2:
        A = ns[0]
        B = ns[1]
        C = ns[2]
        D = ns[3]
        E = ns[4]
        F = ns[5]
    else:
        A = np.array([ns[0,0]])
        B = np.array([ns[1,0]])
        C = np.array([ns[2,0]])
        D = np.array([ns[3,0]])
        E = np.array([ns[4,0]])
        F = np.array([ns[5,0]])
    #print (A,B,C,D,E,F)
    a,b,h,k,w = conic2parametric(A, B, C, D, E, F) 
    return a,b,h,k,w

def conic2parametric(A, B, C, D, E, F):
    M0 = np.array([[F, D/2, E/2],
                   [D/2, A, B/2],
                   [E/2, B/2, C]])
    M = np.array([[A, B/2],
                  [B/2, C]])

    #print np.shape(M0)
    #print np.shape(M)
    e,v = np.linalg.eig(M[:,:,0])
    #print e
    if(abs(e[0] - A[0]) <= abs(e[0]- C[0])):
        l1 = e[0]
        l2 = e[1]
    else:
        l2 = e[0]
        l1 = e[1]

    dM0 = np.linalg.det(M0[:,:,0])
    dM = np.linalg.det(M[:,:,0])

    a = math.sqrt(abs(dM0/(dM*l1)))
    b = math.sqrt(abs(dM0/(dM*l2)))
    
    h = (B*E - 2.0*C*D)/(4.0*A*C - B**2)
    k = (B*D - 2.0*A*E)/(4.0*A*C - B**2)
    w = math.atan(B/(A-C))/2


    if(a > b):
        if(w < 0):
            w = w + math.pi
    else:
        temp = a
        a = b
        b = temp
        w = w + (math.pi/2)

    return a,b,h[0],k[0],w

#obtiene el area de una elipse con los ejes mayor y menor
#a: eje mayor
#b: eje menor
def ellipseArea(a, b):
    return math.pi*(a/2)*(b/2)

#funcion auxiliar para ordenar una lista de elipses usando h
def elliKeyh(ellip):
    return ellip.h

#funcion auxiliar para ordenar una lista de elipses usando k
def elliKeyk(ellip):
    return ellip.k

#funcion auxiliar para ordenar una lista de listas de elipses usando k
def getKeyList(ellilist):
    return ellilist[0].k

def computeHomography(x1, x2):
    if x1.shape[1] != x2.shape[1]:
        print 'Ambas matrices deben tener la misma cantidad de columnas'
        return
    if x1.shape[1] < 4:
        print 'Ambas matrices deben tener al menos 4 columnas.'
        return

    A=np.zeros((x1.shape[1]*2,9))
    j=0
    for i in range(x1.shape[1]):
        A[j,3:6]=-x2[2,i]*x1[:,i]
        A[j,6:9]= x2[1,i]*x1[:,i]
        j+=1
        A[j,0:3]= x2[2,i]*x1[:,i]
        A[j,6:9]= -x2[0,i]*x1[:,i]
        j+=1
    [u,s,v]=np.linalg.svd(A)
    return np.reshape(v[8,:]/v[8,8],(3,3))

#encuentra elipses en una imagen de un canal
#imgray: imagen de un canal
#mina: minima medida del eje mayor de la elipse
def findEllipses(imggray, minma):

    #Cs = [2, 3, 4, 5, 7]
    #blocks=[3, 7, 9, 11]

    #for c in Cs:
        #for b in blocks:
            #th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c)
    ret,th = cv2.threshold(imggray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("img", th) # display
    #cv2.waitKey()

    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = imggray.copy()
    #cv2.imshow("img", img2)
    #cv2.waitKey()

    shape = im2.shape
    #color = np.zeros((shape[0], shape[1], 3));
    #color[:, :, 0] = img2
    #color[:, :, 1] = img2
    #color[:, :, 2] = img2

    #for i in range (len(contours)):
    color = np.zeros((shape[0], shape[1], 3), dtype=imggray.dtype);
    color[:, :, 0] = img2
    color[:, :, 1] = img2
    color[:, :, 2] = img2
    #print len(contours[i])
    #cv2.drawContours(color, contours, -1, (0,255,0), 3)
    #cv2.imshow("img", color) 
    #cv2.waitKey()

    #puntos suficientes para el cotorno 30 a 1000
    ellis = []
    boxes = []
    percent_er = 0.05 # 5% error
    for contour in contours:
        if (len(contour) >= 30 and len(contour <= 1000)):
            (x, y), (Ma, ma), angle = cv2.fitEllipse(contour)
            box = ((x, y), (Ma, ma), angle)
            areaelli = ellipseArea(Ma, ma)
            arectr = cv2.contourArea(contour)
            desv = percent_er*arectr
            #asegura que el area de la ellipse no este alejado del area del contorno
            #desviacion del 5%
            if((areaelli >= arectr - desv) and (areaelli <= arectr + desv) and Ma >= minma):
                #print (box)
                #print ([x, y, Ma, ma, angle])
                #print("percent")
                ellis.append(Ellipse(Ma, ma, x, y, angle))
                boxes.append(box)

    if(verbose==True):
        print("Numero de elispses: ")
        print(len(ellis))

    if(visualize==True):
        cv2.drawContours(color, contours, -1, (0, 255, 0), 3)
        for box in boxes:
            #cv2.circle(color, (int(elli.h), int(elli.k)), 8, 255)
            cv2.ellipse(color, box, (0, 0, 255), 2)
            #cv2.ellipse(color, (int(elli.h), int(elli.k)), (int(elli.a), int(elli.b)), elli.rot, 0, 360, 255)
        cv2.imshow("Ellipses", color)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return ellis 


#encuentra elipses en una imagen de un canal
#este metodo ajusta conicas con 5 puntos del contorno
#imgray: imagen de un canal
#mina: minima medida del eje mayor de la elipse
def findEllipses2(imggray, minma, area_tol = 0.08):

    #Cs = [2, 3, 4, 5, 7]
    #blocks=[3, 7, 9, 11]

    #for c in Cs:
        #for b in blocks:
            #th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c)
    ret,th = cv2.threshold(imggray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("img", th) # display
    #cv2.waitKey()

    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = imggray.copy()
    #cv2.imshow("img", img2)
    #cv2.waitKey()

    shape = im2.shape
    #color = np.zeros((shape[0], shape[1], 3));
    #color[:, :, 0] = img2
    #color[:, :, 1] = img2
    #color[:, :, 2] = img2

    #for i in range (len(contours)):
    color = np.zeros((shape[0], shape[1], 3), dtype=imggray.dtype);
    color[:, :, 0] = img2
    color[:, :, 1] = img2
    color[:, :, 2] = img2
    #print len(contours[i])
    #cv2.drawContours(color, contours, -1, (0,255,0), 3)
    #cv2.imshow("img", color) 
    #cv2.waitKey()

    #puntos suficientes para el cotorno 30 a 1000
    ellis = []
    boxes = []
    percent_er = area_tol
    for contour in contours:
        if (len(contour) >= 30 and len(contour <= 1000)):
            #(x, y), (Ma, ma), angle = cv2.fitEllipse(contour)
            stesps = len(contour)/5
            pts = []
            #print(contour)
            #print(contour.shape)
            for i in range(5):
                pts.append([float(contour[i*stesps, 0, 0]), float(contour[i*stesps, 0, 1])])
            
            elipts = np.array(pts, dtype=float)
            #print("puntos para la elipse")
            #print(elipts)
            a,b,h,k,w = adjust_ellispe(elipts)
            w = rad2deg(w)
            box = ((h, k), (a*2.0, b*2.0), w)
            areaelli = ellipseArea(a*2.0, b*2.0)
            arectr = cv2.contourArea(contour)
            desv = percent_er*arectr
            #asegura que el area de la ellipse no este alejado del area del contorno
            #desviacion del 5%
            if((areaelli >= arectr - desv) and (areaelli <= arectr + desv) and a >= minma):
                #print (box)
                #print ([x, y, Ma, ma, angle])
                #print("percent")
                ellis.append(Ellipse(a*2.0, b*2.0, h, k, w))
                boxes.append(box)

    if(verbose==True):
        print("Numero de elispses: ")
        print(len(ellis))

    if(visualize==True):
        cv2.drawContours(color, contours, -1, (0, 255, 0), 3)
        for box in boxes:
            #cv2.circle(color, (int(elli.h), int(elli.k)), 8, 255)
            cv2.ellipse(color, box, (0, 0, 255), 2)
            #cv2.ellipse(color, (int(elli.h), int(elli.k)), (int(elli.a), int(elli.b)), elli.rot, 0, 360, 255)
        cv2.imshow("Ellipses", color)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return ellis 



#define los parametros de una linea recta con dos puntos y = mx + b
#p1: punto 1
#p2: punto 2
def points2Line(p1, p2):
    m = (p2[1] - p1[1])/(p2[0] - p1[0]);
    b = p1[1] - m*(p1[0]); 
    return [m, b]

#distancia de un punto a una linea recta
#p: punto en formato de list de dos elementos
#line: linea recta parametrizada como [m, b], y = mx+b
def pointLineDist(p, line):
    #y = mx + b
    #-mx + y - b = 0
    # A = -m, B = 1, C = -b
    return abs(-line[0]*p[0] + p[1] - line[1])/math.sqrt(line[0]*line[0] + 1)

#Genera grupos de elipses que yacen sobre la misma linea recta o se aproximan
#con una distancia epsilon
#line: linea recta parametrizada como [m, b], y = mx+b
#ellipses: lista de elipses
#epsilon: distancia minima para considerarse en la misma linea
def groupAlligEllipses(line, ellipses, epsilon):
    grup = []
    for elli in ellipses:
        center = [elli.h, elli.k]
        dist = pointLineDist(center, line)
        if(dist <= epsilon):
            grup.append(elli)
    return grup

#encuentra un patron de elipses con row files y cols columnas
#img: imagen de un solo canal
#ellipses: lista ellipses que se encuentran en la imagen
#rows: numero de filas del patron de elipses
#cols: numero de columnas del patron de elipses
def findElliGridPattern(img, ellipses, rows, cols):
    shape = img.shape
    color = np.zeros((shape[0], shape[1], 3), dtype=img.dtype);
    img2 = img.copy()
    color[:, :, 0] = img2
    color[:, :, 1] = img2
    color[:, :, 2] = img2
    groupellipses = []
    
    #agrupa ellipses trazando lineas y mediendo su distancia
    for elli1 in ellipses:
        centerf = [elli1.h, elli1.k]
        for elli2 in ellipses:
            if(elli1 != elli2):
                center2 = [elli2.h, elli2.k]
                line = points2Line(centerf, center2)
                group = groupAlligEllipses(line, ellipses, 4.0)
                groupellipses.append(group)
    
    if(verbose):
        print("Ellipses agrupadas")
        print(len(groupellipses))

    #filtra los que tienen al menos cols numero de puntos alineados
    groupcolmin = []
    epsilon = 0.001
    m_eps = 0.1
    b_eps = 4.0
    exist = 0
    for grup in groupellipses:
        exist = 0
        if(len(grup)>=cols):
            #groupcolmin.append(grup)
            #print("nuevo grupo de al menos 5")
            #print(len(grup))
            if(not groupcolmin):
                groupcolmin.append(grup)
                center11 = [grup[0].h, grup[0].k]
                center12 = [grup[1].h, grup[1].k]
                line1 = points2Line(center11, center12)
                #print("line1", line1)
            else:
                center11 = [grup[0].h, grup[0].k]
                center12 = [grup[1].h, grup[1].k]
                line1 = points2Line(center11, center12)
                for grup2 in groupcolmin:
                    #print(len(grup2))
                    center21 = [grup2[0].h, grup2[0].k]
                    center22 = [grup2[1].h, grup2[1].k]
                    line2 = points2Line(center21, center22)
                    #filtra conjuntos que estan alineados en una linea muy cercana
                    
                    if(abs(line1[0]-line2[0])<=m_eps and abs(line1[1]-line2[1])<=b_eps):
                        exist = 1
                        break
            
                if(exist == 0):
                    #print("line1", line1)
                    #print("line2", line2)
                    groupcolmin.append(grup)
                
                

    if(verbose):
        print("Grupos de elipses agrupadas en lineas con al menos " + str(cols) + " elipses")
        print(len(groupcolmin))

    #drawlines
    #ordena a lo largo del eje x
    sortedx = []
    for grup in groupcolmin:
        sortedx.append(sorted(grup, key= elliKeyh))

    #ordena a lo largo del eje y
    sortedgroup = sorted(sortedx, key=getKeyList)
    #print(groupcolmn[0][0])
    for grup in sortedx:
        center11 = (int(grup[0].h), int(grup[0].k))
        center12 = (int(grup[len(grup)-1].h), int(grup[len(grup)-1].k))
        cv2.line(color, center11, center12, (0, 255, 0), 2)
    
    if(visualize):
        cv2.imshow("lines", color)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if(verbose):
        print("grupos de elipses alineados y ordenados en xy:")
        for grup in sortedgroup:
            print ("##############")
            for ell in grup:
                print(ell)

    #retorna False si no hay al menos el numero de filas
    #no se cumplio el patron
    if(len(sortedgroup) < rows):
        return False, []

    #verifica que al menos existan cols conjuntos de puntos
    #con rows puntos que estan alineados en la misma recta
    rowpat = None
    pads = []
    epsilon = 5
    for irow in range(len(sortedgroup) - rows + 1):
        paddings = []
        padcount = []
        for i in range(rows):
            paddings.append(len(sortedgroup[irow + i]) - cols)
            padcount.append(0)
        
        perms = False
        found = False
        while(perms == False and found == False):
            rowfailallign = False
            for c in range(cols):
                elli1 = sortedgroup[irow][padcount[0] + c]
                elli2 = sortedgroup[irow+1][padcount[1] + c]
                line = points2Line([elli1.h, elli1.k], [elli2.h, elli2.k])
                allicount = 0
                for i in range(rows):
                    el = sortedgroup[irow + i][padcount[i]+c]
                    if(pointLineDist([el.h, el.k], line) <= epsilon):
                        allicount = allicount + 1
                
                #no se pudo alinear rows puntos a la linea
                if(allicount != rows):
                   rowfailallign = True
                   break
            
            if(rowfailallign == False):
                found = True 
            else:
                #incrementa los paddings
                for p in range(len(paddings)):
                    if(padcount[p] < paddings[p]):
                        padcount[p] = padcount[p] + 1
                        break
                    if(p == len(paddings)-1):
                        #no hay mas paddings por correr
                        if(padcount[p] == paddings[p]):
                            perms = True
                            break
    
        if(found == True):
            pads = padcount[:]
            #print(pads)
            rowpat = irow
            break

    #print(pads)
    #print(rowpat)

    #si no hubo el patron retorna
    if(rowpat==None or not pads):
        return False, []

    #reordena los puntos en alineados en un grid ordenado
    sortedcenters = []
    for i in range (rowpat, rows):
        for j in range(cols):
            el = sortedgroup[i][j]
            sortedcenters.append([el.h, el.k])
    
    if(len(sortedcenters)<rows*cols):
        return False, []

    if(visualize):
        for i in range(len(sortedcenters)):
            cv2.putText(color, str(i), (int(sortedcenters[i][0]), int(sortedcenters[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        cv2.imshow("pattern", color)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    return True, sortedcenters

#de los puntos calcula  
def calcCorrectionError(points1, points2):
    if(verbose==True):
        print("Distancia de diferencia de puntos")
        print(np.sqrt(np.sum(np.power(points2 - points1, 2.0), axis=1)))
    diff = np.sum(np.sqrt(np.sum(np.power(points2 - points1, 2.0), axis=1)))
    return diff/float(points1.shape[0])


#puntos en el plano del patron de circulos medido en milimetros
srcPoints = np.array([[0.0, 47.0, 94.0, 141.0, 188.0, 0.0, 47.0, 94.0, 141.0, 188.0, 0.0, 47.0, 94.0, 141.0, 188.0], 
                      [0.0, 0.0, 0.0, 0.0, 0.0, 47.0, 47.0, 47.0, 47.0, 47.0, 94.0, 94.0, 94.0, 94.0, 94.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [1.1, 1.1, 1.1, 1.1, 1.1, 1.1,1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]], dtype=float)

#los mismo puntos ordenados en lista medido en milimetros
#srcPoints2 = np.array([[0.0, 0.0, 1.0], 
#                       [47.0, 0.0, 1.0],
#                       [94.0, 0.0, 1.0],
#                       [141.0, 0.0, 1.0],
#                       [188.0, 0.0, 1.0],
#                       [0.0, 47.0, 1.0],
#                       [47.0, 47.0, 1.0],
#                       [94.0, 47.0, 1.0],
#                       [141.0, 47.0, 1.0],
#                       [188.0, 47.0, 1.0],
#                       [0.0, 94.0, 1.0],
#                       [47.0, 94.0, 1.0],
#                       [94.0, 94.0, 1.0],
#                       [141.0, 94.0, 1.0],
#                       [188.0, 94.0, 1.0]], dtype=float)

#los mismo puntos ordenados en lista medido en milimetros
srcPoints2 = np.array([[0.0, 0.0, 0.0], 
                       [47.0, 0.0, 0.0],
                       [94.0, 0.0, 0.0],
                       [141.0, 0.0, 0.0],
                       [188.0, 0.0, 0.0],
                       [0.0, 47.0, 0.0],
                       [47.0, 47.0, 0.0],
                       [94.0, 47.0, 0.0],
                       [141.0, 47.0, 0.0],
                       [188.0, 47.0, 0.0],
                       [0.0, 94.0, 0.0],
                       [47.0, 94.0, 0.0],
                       [94.0, 94.0, 0.0],
                       [141.0, 94.0, 0.0],
                       [188.0, 94.0, 0.0]], dtype=float)

#los mismo puntos ordenados en lista medido en milimetros
srcPoints2H = np.array([[0.0, 0.0, 0.0, 1.0], 
                       [47.0, 0.0, 0.0, 1.0],
                       [94.0, 0.0, 0.0, 1.0],
                       [141.0, 0.0, 0.0, 1.0],
                       [188.0, 0.0, 0.0, 1.0],
                       [0.0, 47.0, 0.0, 1.0],
                       [47.0, 47.0, 0.0, 1.0],
                       [94.0, 47.0, 0.0, 1.0],
                       [141.0, 47.0, 0.0, 1.0],
                       [188.0, 47.0, 0.0, 1.0],
                       [0.0, 94.0, 0.0, 1.0],
                       [47.0, 94.0, 0.0, 1.0],
                       [94.0, 94.0, 0.0, 1.0],
                       [141.0, 94.0, 0.0, 1.0],
                       [188.0, 94.0, 0.0, 1.0]], dtype=float)*0.001

#matriz de parametros intrinsecos de la camara
K = np.array([[1377.804, 0, 955.927], [0, 1377.804, 545.961], [0.0, 0.0, 1.0]], dtype=float)
Kinv = np.linalg.inv(K)





def main(args):
    print("puntos en el plano del mundo")
    print(srcPoints2)

    init = args["init"]
    end = args["end"]
    path = args["path"]
    efinder = args["efinder"]


    for i in range(init, end+1):
        cv2.destroyAllWindows()
        if(i < 100):
            imgp = os.path.join(path+"/NotCorrected/Circulos_0" + str(i)+".png")
        else:
            imgp = os.path.join(path+"/NotCorrected/Circulos_" + str(i)+".png")
        img = cv2.imread(imgp, cv2.IMREAD_GRAYSCALE)
        if(not img is None):
            print("###################################################")
            print(imgp)
            print("###################################################")
            if(efinder== 0):
                ellis = findEllipses(img, 15)
            else:
                ellis = findEllipses2(img, 15)

            if(len(ellis)>=15):
                ret, grid = findElliGridPattern(img, ellis, 3, 5)
            else:
                ret = False

            #se encontro el patron
            if(ret==True):
                dstPointsList = []
                for p in grid:
                    dstPointsList.append([p[0], p[1], 1.0])

                #print(srcPoints2)
                dstPoints = np.array(dstPointsList, dtype=float)
                print("puntos en la imagen")
                print(dstPoints)
                #print(dstPointsList)
                #H = computeHomography(srcPoints, dstPoints)
                dstPointsK = np.zeros(shape=(15, 3), dtype=float)
                for i in range(dstPointsK.shape[0]):
                    dstPointsK[i, :] = Kinv.dot(dstPoints[i, :])
                    dstPointsK[i, :] = dstPointsK[i, :]/dstPointsK[i, 2]

                H,mask = cv2.findHomography(srcPoints2, dstPointsK)
                print("Homografia:")
                print(H)

                H2 = H[:, :]

                #r1r2 = np.zeros(shape=(3,2), dtype=float)
                #r1r2[:, 0] = H[:, 0]
                #r1r2[:, 1] = H[:, 1]
                
                r1 = H2[:, 0]
                n = math.sqrt(r1.dot(r1))
                H2 = H2/n;
                r2 = H2[:, 1]
                r3 = H2[:, 2]
                
                r2 = r2 - r2.dot(r1) * r1
                n = r2.dot(r2)
                r2 = r2/math.sqrt(n)
                #aqui r1 no se habia actualizado
                #r1 = H2[:,0]
                r3 = np.cross(H2[:, 0], r2)

                rr = np.zeros(shape=(3, 3), dtype = float)
                rr[:, 0] = H2[:, 0]
                rr[:, 1] = r2
                rr[:, 2] = r3 


                R = rr

                print("R (gram schmidt):")
                print(R)
                print (R.T)
                print (np.linalg.inv(R))


                print(np.linalg.det(R))

                n1 = np.linalg.norm(R[:, 0])
                n2 = np.linalg.norm(R[:, 1])
                n3 = np.linalg.norm(R[:, 2])

                print("norma r1")
                print(n1)
                print("norma r2")
                print(n2)
                print("norma r3")
                print(n3)
                #cross = np.cross(R[:, 0], R[:, 1])
                #print("Producto r1xr2")
                #print(cross)

                dot = np.dot(R[:, 0], R[:, 1])
                print("Producto r1*r2")
                print(dot)
        
                RT = np.zeros(shape=(3, 4), dtype=float)
                RT[0:3, 0:3] = R[:,:]
                RT[:, 3] = H[:, 2]
                print("RT:")
                print(RT)

                x1 = np.zeros(shape=(15, 3), dtype=float)
                #calcula RT*X
                for i in range(srcPoints2.shape[0]):
                    #x1[i, :] = H.dot(srcPoints2[i, :])
                    #x1[i, :] = x1[i, :]
                    x1[i, :] = RT.dot(srcPoints2H[i, :])
                    x1[i, :] = x1[i, :]/x1[i, 2]

                x2 = np.zeros(shape=(15, 3), dtype=float)
                #calular k^-1*x
                for i in range(dstPoints.shape[0]):
                    x2[i, :] = Kinv.dot(dstPoints[i, :])

                if(verbose==True):
                    print("RT*X: ")
                    print(x1)

                if(verbose==True):
                    print("K^-1*(u, v, 1)^T: ")
                    print(x2)

                erroc = calcCorrectionError(x1, x2)
                print("Error de proyeccion: %f\n"% erroc)
                #print(erroc)
                if(visualize==True):
                    cv2.destroyAllWindows()
            else:
                print("NO SE PUDO AJUSTAR EL PATRON")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Error de correccion (Homografias)")

    parser.add_argument("--efinder", required=False, type=int, default=0, help="Bandera para indicar que tipo de busqueda de elipse se usa (0=OpenCV o 1=Ajuste de conicas)")

    parser.add_argument("--path", required=True, type=str, help="Ruta al folder de imagenes del patron")

    parser.add_argument("--init", required=False, type=int, default=26, help="Numero de inicio de la imagen")

    parser.add_argument("--end", required=False, type=int, default=139, help="Numero final de la imagen")

    parser.add_argument("--verbose", required=False, action="store_true", help="Imprime detalles del proceso")

    parser.add_argument("--viz", required=False, action="store_true", help="Muestra la visualizacion del proceso")

    #obten los argumentos
    args = parser.parse_args()

    verbose = args.verbose
    visualize = args.viz

    if(args.efinder != 0 or args.efinder != 1):
        efinder = 0
    else:
        efinder = args.efinder
    path = args.path
    init = args.init
    end = args.end

    dictargs = {"efinder": efinder, "path": path, "init": init, "end": end}

    main(dictargs)