\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: [ 4.47776435830348e-9 stock0 ^2 - 6.8321325342325e-10 stock0 * stock1
       + 1.60972954181226e-8 stock0 * stock2
       + 3.33833866832079e-8 stock0 * stock3
       - 1.02653921684304e-9 stock0 * stock4 + 7.87779309721586e-11 stock1 ^2
       - 8.67581785487318e-10 stock1 * stock2
       - 2.1834441809093e-9 stock1 * stock3
       + 8.10663580367948e-12 stock1 * stock4 + 3.72230821584413e-9 stock2 ^2
       + 2.87594438946524e-8 stock2 * stock3
       - 9.31958348575969e-10 stock2 * stock4 + 4.51524177786955e-8 stock3 ^2
       - 2.31982196004886e-9 stock3 * stock4 + 1.92606021994948e-11 stock4 ^2
       ] / 2
Subject To
 c1:  0.00026248677831437 stock0 + 0.0000851721926592752 stock1
      + 0.000567996143313569 stock2 - 0.000559848579769264 stock3
      + 0.000239129622020248 stock4  = 1
 c2:  1.3 stock0 + 0.3 stock1 + 0.3 stock2 + 0.3 stock3 + 0.3 stock4 >= 0
 c3:  0.3 stock0 + 1.3 stock1 + 0.3 stock2 + 0.3 stock3 + 0.3 stock4 >= 0
 c4:  0.3 stock0 + 0.3 stock1 + 1.3 stock2 + 0.3 stock3 + 0.3 stock4 >= 0
 c5:  0.3 stock0 + 0.3 stock1 + 0.3 stock2 + 1.3 stock3 + 0.3 stock4 >= 0
 c6:  0.3 stock0 + 0.3 stock1 + 0.3 stock2 + 0.3 stock3 + 1.3 stock4 >= 0
 c7:  0.55 stock0 - 0.45 stock1 - 0.45 stock2 - 0.45 stock3 - 0.45 stock4 <= 0
 c8:  - 0.45 stock0 + 0.55 stock1 - 0.45 stock2 - 0.45 stock3 - 0.45 stock4
      <= 0
 c9:  - 0.45 stock0 - 0.45 stock1 + 0.55 stock2 - 0.45 stock3 - 0.45 stock4
      <= 0
 c10: - 0.45 stock0 - 0.45 stock1 - 0.45 stock2 + 0.55 stock3 - 0.45 stock4
      <= 0
 c11: - 0.45 stock0 - 0.45 stock1 - 0.45 stock2 - 0.45 stock3 + 0.55 stock4
      <= 0
Bounds
 0 <= stock0 <= 1
 0 <= stock1 <= 1
 0 <= stock2 <= 1
 0 <= stock3 <= 1
 0 <= stock4 <= 1
End
