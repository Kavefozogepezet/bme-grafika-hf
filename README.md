# BME Computer Graphics homework 2022/23/2
## Homework 1 - UFO Hami

>K�sz�tsen CPU sug�rk�vet�s felhaszn�l�s�val lehallgat�stervez� programot.
>
>1. A megjelen�tett szoba t�glatest alak� benne k�t tov�bbi t�pus� Platon-i szab�lyos test tal�lhat� (Pl. OBJ form�tum� defin�ci�juk: https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html). A szoba falai k�v�lr�l befel� �tl�tsz�ak, �gy belel�tunk a szob�ba m�g k�v�lr�l is.
>2. A szob�ban 3 darab k�p alak� lehallgat� tal�lhat�. Az lehallgat� csak a k�p sz�g�ben k�pes �rz�kelni, az �rz�kenys�g a t�vols�ggal cs�kken (a cs�kken�s sebess�ge megv�laszthat�, c�l az eszt�tikus megjelen�s).
>3. A szob�t �s eszk�zeit alapesetben sz�rke sz�nnel spekul�ris-ambiens modellel jelen�t�k meg, amely a fel�leti norm�lis �s a n�zeti ir�ny k�z�tti sz�g koszinusz�nak line�ris f�ggv�nye, amelynek �rt�kk�szlete a [0.2, 0.4] tartom�ny (L = 0.2 * (1 + dot(N, V)), ahol L az �szlelt sug�rs�r�s�g, N a fel�leti norm�lis, V pedig a n�zeti ir�ny, mindketten egys�gvektorok).
>4. A lehallgat�s �rz�kenys�g�t a sz�rke alapsz�nhez hozz�adjuk, az els� lehallgat�ra piros �rnyalatokkal, a m�sodikra z�lddel, a harmadikra k�kkel. Az �rz�kenys�g a takart pontokban z�rus.
>5. A lehallgat�k interakt�van �thelyezhet�k. A bal eg�rgomb lenyom�s�ra, a kurzor alatt l�that� 3D ponthoz megkeress�k a legk�zelebbi lehallgat�t, annak poz�ci�j�t a megtal�lt pontra �ll�tjuk, az ir�ny�t pedig a fel�let norm�lvektor�ra.
>

The solution for other assignments is on their respective branches.
