\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: [ 3.64450108612664e-9 stock0 ^2 + 1.08729576930426e-9 stock0 * stock1
       + 1.72071260284505e-8 stock0 * stock2
       - 1.01240237196004e-8 stock0 * stock3
       + 2.36839614245802e-10 stock0 * stock4 + 2.33641732031311e-10 stock1 ^2
       + 1.35863252506915e-9 stock1 * stock2
       - 1.69881141995524e-8 stock1 * stock3
       + 6.67498199520636e-10 stock1 * stock4 + 5.48605902976426e-9 stock2 ^2
       - 2.03525645462978e-8 stock2 * stock3
       + 7.34124939380558e-10 stock2 * stock4 + 1.02273358455241e-7 stock3 ^2
       - 1.61522912523068e-8 stock3 * stock4 + 1.63522681527237e-10 stock4 ^2
       ] / 2
Subject To
 c1:  stock0 + stock1 + stock2 + stock3 + stock4  = 1
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
