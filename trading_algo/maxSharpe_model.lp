\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: [ 7.43620053425364e-9 stock0 ^2 - 6.08900955741343e-8 stock0 * stock1
       + 1.62118353200839e-8 stock0 * stock2
       + 4.29517829475243e-9 stock0 * stock3
       - 3.31049764718194e-8 stock0 * stock4 + 7.5105629648483e-8 stock1 ^2
       - 5.80882412812014e-8 stock1 * stock2
       - 8.24909685173629e-9 stock1 * stock3
       + 1.45859721047057e-7 stock1 * stock4 + 2.41443863362425e-8 stock2 ^2
       + 8.27081104423603e-9 stock2 * stock3
       - 5.28450549514071e-9 stock2 * stock4 + 8.88126917719613e-10 stock3 ^2
       + 5.64509056616327e-8 stock3 * stock4 + 3.70806173205208e-7 stock4 ^2
       ] / 2
Subject To
 c1:  stock0 + stock1 + stock2 + stock3 + stock4  = 1
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
Bounds
 0 <= stock0 <= 1
 0 <= stock1 <= 1
 0 <= stock2 <= 1
 0 <= stock3 <= 1
 0 <= stock4 <= 1
End
