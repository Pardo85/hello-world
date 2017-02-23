# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

" es 1 "

s=("abba","anna","qqu","marco","asia")
def match_ends(s)
p=0

for el in s:
    if len(el)>=2 and el[0]==el[-1]:
        p=p+1
        

print p
return
match_ends(s)


" es 2 front_x"
inizio = ['mix', 'xyz', 'apple', 'xanadu', 'aardvark'] 
def prova2(lista):
    lista_x = []
    lista_altro = []
    lista_el1 = []
    lista_el2 = []
    lista_finale = []
    for el in lista: 
     if el[0] == 'x':
      lista_el1=[el]
      lista_x = lista_x + lista_el1
     else:
      lista_el2 = [el]
      lista_altro = lista_altro + lista_el2
    lista_x = sorted(lista_x)
    lista_altro = sorted(lista_altro)
    lista_finale = lista_x + lista_altro
    print lista_finale
    return
prova2(inizio)

" es3 "
tlist = [(1, 7), (1, 3), (3, 4, 5), (2, 2)]

def sort_last(tup):
    diz={}
    for el in tup:
        a = el[-1] 
        diz[a]=el
    print diz   
    return

sort_last(tlist)

" es4 "

numlist = [1, 2, 2, 3] 
def adiacente(num):
    count = 0 
    while count < len(num) - 1:
     if num[count] == num[count + 1] :
      del num[count]
     else:
      count = count+1
    print num 
    return
adiacente(numlist)           
        
" es 5 "
lista1 = ['aa','xx','zz']
lista2 = ['cc','bb']
def linear_merge(x,y):
    lista_fin = sorted(lista1 + lista2)
    print lista_fin
    return
linear_merge(lista1,lista2)

" es 6 "
def donuts(donuts):
    if donuts > 10:
       print 'Number of donuts: many'
    else:
       print 'Number of donuts: ' , donuts
    return
donuts(5)
donuts(19)


"es 7 "

a1 = ''
a2 = ''
a3 = ''

def both_ends(x):
    if len(x) > 2:
       a1 = x[0:2]
       a2 = x[len(x)-2: len(x)-2 + 2]
       a3 = a1 + a2
    else:
       a3 = ""
    print a3   
    return
both_ends('saluti')


"es 8 "

def fix_start(str):
    l1 = []
    l2 = []
    l1 = list(str)
    for el in l1:
     if el in l2:
        l2.append('*')
     else:
        l2.append(el)
    print l2
fix_start('abba')

"es 9 "

def mixup(x,y):
    if len(x) > 2 and len(y) > 1:
     f1 = x[0:2]
     f2 = y[0:2]
     l1 = len(x)
     l2 = len(y)
     b1 = x[2: len(x)]
     b2 = y[2: len(y)]
     new1 = f2 + b1
     new2 = f1 + b2
     new = new1 + new2
     print new
    else:
     pass 
mixup('pane','salame')    

" es 10 "

def verbing(x):
    if len(x) > 3:
       if x[len(x) - 3: len(x)] == 'ing':
          x = x + 'ly'
       else:
          x = x + 'ing'
    else:
       x = x
    print x
verbing('playing')       

" es 11 "

def notbad(x):
     y = x
     while 'bad' in y and 'not' in y:
      if y.index('not') >= 0:
       i1 = y.index('not')
      if y.index('bad') >= 0:
       i2 = y.index('bad')
      if i2 > i1:
       rep = y[i1: i2 + 3]
       y = y.replace(rep,"good")    
      print i1 
      print i2
      print rep
      print y
notbad("this course is not ciao bad, what do you think not bad ")    

" ES12 "
def front_back(x,y):      
     if len(x)%2 == 0:
         f1 = len(x)/2 
     else:
      f1 = len(x)/2 + 1
     if len(y)%2 == 0:
         f2 = len(y)/2 
     else:
      f2 = len(y)/2 + 1     
              
     a_front =  x[0: f1]
     b_front =  y[0: f2]
     a_back  =  x[f1:len(x)]  
     b_back  =  y[f2:len(y)]
     new1 = x[0: f1] +y[0: f2]  + x[f1:len(x)] + y[f2:len(y)]
     print f1
     print f2
     print a_front
     print b_front
     print a_back
     print b_back   
     print new1
front_back("alessio","figo")
