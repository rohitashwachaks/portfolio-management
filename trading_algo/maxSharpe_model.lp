\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: [ 4.47776435830348e-9 stock0 ^2 - 3.41606626711625e-10 stock0 * stock1
       + 8.04864770906131e-9 stock0 * stock2
       + 1.6691693341604e-8 stock0 * stock3
       - 5.1326960842152e-10 stock0 * stock4 + 7.87779309721586e-11 stock1 ^2
       - 4.33790892743659e-10 stock1 * stock2
       - 1.09172209045465e-9 stock1 * stock3
       + 4.05331790183974e-12 stock1 * stock4 + 3.72230821584413e-9 stock2 ^2
       + 1.43797219473262e-8 stock2 * stock3
       - 4.65979174287984e-10 stock2 * stock4 + 4.51524177786955e-8 stock3 ^2
       - 1.15991098002443e-9 stock3 * stock4 + 1.92606021994948e-11 stock4 ^2
       ] / 2
Subject To
 c1:  0.00026248677831437 stock0 + 0.0000851721926592752 stock1
      + 0.000567996143313569 stock2 - 0.000559848579769264 stock3
      + 0.000239129622020248 stock4  = 1
 c2:  stock0 >= 0
 c3:  stock1 >= 0
 c4:  stock2 >= 0
 c5:  stock3 >= 0
 c6:  stock4 >= 0
 c7:  0.55 stock0 - 0.45 stock1 - 0.45 stock2 - 0.45 stock3 - 0.45 stock4 <= 0
 c8:  - 0.45 stock0 + 0.55 stock1 - 0.45 stock2 - 0.45 stock3 - 0.45 stock4
      <= 0
 c9:  - 0.45 stock0 - 0.45 stock1 + 0.55 stock2 - 0.45 stock3 - 0.45 stock4
      <= 0
 c10: - 0.45 stock0 - 0.45 stock1 - 0.45 stock2 + 0.55 stock3 - 0.45 stock4
      <= 0
 c11: - 0.45 stock0 - 0.45 stock1 - 0.45 stock2 - 0.45 stock3 + 0.55 stock4
      <= 0
End
