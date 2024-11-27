# Documentation 
## Documentation des codes :
### Code en C :
Le fichier **main.c** fait la résolution du système linéaire: $$A_{h} u_{h} = b_{h}$$

Les fichiers source du dossier **src** contiennent les fonctions de la factorisation LU ainsi que la resolution du systéme , l'initialisation du vecteur B et la fonction s.
Ceci se fait dans le fichier **functions.c**. le fichier **utiles.c** contient les fonctions secondaires comme l'affichage et les tests.

Le dossier **include** quand à lui permet d'inclure le contenu des fichiers sources au main. on y retrouve les signatures des fonctions de chaque fichier source.

Le fichier **LU_efficace.c** est un programme qui compare entre la methode LU classique et la tri-diagonale, Ceci est independant du main pour le compiler et l'executer il faut donc tapez les lignes de commande **gcc Lu_efficace** et **./a.out**.

Le **makefile** permet d'executer le programe tapez :
- **make** pour compiler (generer les objets executable ).
+ **./main** pour executer et voir le resultat.
* **make** clean pour supprimer les objets executable.


### Code en Python :
Tout le code en python ce trouve sur page jupyter y compris les commentaires et tests .
## Methodes numeriques :
### Factorisation LU d'une matrice tri-diagonale.

Soit la matrice A sous la forme tri-diagonale suivante:

$$
\left(\begin{array}{cc} 
b_{1} & c_{1} & 0 & \cdots &0\\ 
a_{2} & b_{2} & c_{2}& \ddots & \vdots \\ 
\vdots  & \ddots  & \ddots  & \ddots & \vdots   \\
0 & 0 & a_{n-1} & b_{n-1}& c_{n-1} \\ 
0 & 0 & 0 & a_{n} & b_{n}\\ 
\end{array}\right)
$$ 

On peut écrire cette matrice comme produit des deux matrices suivante :

$$
\left(\begin{array}{cc} 
1 & 0 & \cdots & \cdots &0\\ 
l_{2} & 1 & \ddots & \ddots & \vdots \\ 
\vdots  & \ddots  & \ddots  & \ddots & \vdots   \\
\vdots & \ddots & l_{n-1} & 1& 0 \\ 
0 & \cdots & 0 & l_{n} & 1\\ 
\end{array}\right)
\left(\begin{array}{cc} 
v_{1} & c_{1} & 0 & \cdots &0\\ 
0 & v_{2} & c_{2}& \ddots & \vdots \\ 
\vdots  & \ddots  & \ddots  & \ddots & \vdots   \\
\vdots & \ddots & \ddots & v_{n-1}& c_{n-1} \\ 
0 & \cdots & \cdots & 0 & v_{n}\\ 
\end{array}\right)
$$ 
##### On va par la suite suivre la methode de la correspondance terme à terme pour faire la decomposition LU :

$$b_{1}=v_{1} \Rightarrow v_{1} = b_{1} $$

$$a_{k}= l_{k} v_{k−1} \Rightarrow l_{k} = a_{k}/v_{k−1}$$

$$b_{k} = l_{k} c_{k−1} + v_{k} \Rightarrow v_{k} = b_{k} − l_{k} c_{k−1}, \hspace{10pt} k = 2,\cdots, n $$

### Resolution du system Ax = b :
##### On commence par resoudre Ly = b:
Methode de descente:
$$y_{1} = b_{1}$$

$$l_{k} y_{k−1} + y_{k} = b_{k} \Rightarrow  y_{k} = b_{k} − l_{k} y_{k−1}, \hspace{10pt} k = 2, \cdots, n$$

##### Par la suite on fait la resolution de Ux = y:
Methode de remontée:
$$v_{n}x_{n} = yn \Rightarrow x_{n} = y_{n} / v_{n} $$

$$v_{k}x_{k} + c_{k}x_{k+1} = y_{k} \Rightarrow x_{k} = (y_{k} − c_{k}x_{k+1})/v_{k}, \hspace{10pt} k = n − 1,\cdots , 1$$

### Le Système linéaire sous la forme matricielle:
L'équation discrétisée est: $$u''(x_i) ≈ \frac{u(x_{i-1}) - 2u(x_i) + u(x_{i+1})}{h^2}$$


La formulation matricielle :\
$A_{h}$ est sous la forme :

$$\frac{1}{h²}
\left(\begin{array}{cc} 
2 & -1 & 0 & \cdots &0\\ 
-1 & 2 & -1& \ddots & \vdots \\ 
\vdots  & \ddots  & \ddots  & \ddots & \vdots   \\
0 & 0 & -1 & 2& -1 \\ 
0 & 0 & 0 & -1 & 2\\ 
\end{array}\right)
$$

$U_{h}$ est sous la forme :

$$\left(\begin{array}{cc} 
u_{1} \\ 
u_{2} \\ 
\vdots\\
u_{n-1} \\ 
u_{n} \\ 
\end{array}\right)
$$

$b_{h}$ est sous la forme :

$$\left(\begin{array}{cc} 
s(x_{1}) + \frac{\alpha}{h²} \\ 
s(x_{2}) \\ 
\vdots\\
s(x_{n-1}) \\ 
s(x_{n}) + + \frac{\beta}{h²} \\ 
\end{array}\right)
$$

