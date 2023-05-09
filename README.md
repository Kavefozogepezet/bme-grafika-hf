# BME Computer Graphics homework 2022/23/2
## Homework 1 - UFO Hami

>Készítsen CPU sugárkövetés felhasználásával lehallgatástervezõ programot.
>
>1. A megjelenített szoba téglatest alakú benne két további típusú Platon-i szabályos test található (Pl. OBJ formátumú definíciójuk: https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html). A szoba falai kívülrõl befelé átlátszóak, így belelátunk a szobába még kívülrõl is.
>2. A szobában 3 darab kúp alakú lehallgató található. Az lehallgató csak a kúp szögében képes érzékelni, az érzékenység a távolsággal csökken (a csökkenés sebessége megválasztható, cél az esztétikus megjelenés).
>3. A szobát és eszközeit alapesetben szürke színnel spekuláris-ambiens modellel jelenítük meg, amely a felületi normális és a nézeti irány közötti szög koszinuszának lineáris függvénye, amelynek értékkészlete a [0.2, 0.4] tartomány (L = 0.2 * (1 + dot(N, V)), ahol L az észlelt sugársûrûség, N a felületi normális, V pedig a nézeti irány, mindketten egységvektorok).
>4. A lehallgatás érzékenységét a szürke alapszínhez hozzáadjuk, az elsõ lehallgatóra piros árnyalatokkal, a másodikra zölddel, a harmadikra kékkel. Az érzékenység a takart pontokban zérus.
>5. A lehallgatók interaktívan áthelyezhetõk. A bal egérgomb lenyomására, a kurzor alatt látható 3D ponthoz megkeressük a legközelebbi lehallgatót, annak pozícióját a megtalált pontra állítjuk, az irányát pedig a felület normálvektorára.
>

The solution for other assignments is on their respective branches.
