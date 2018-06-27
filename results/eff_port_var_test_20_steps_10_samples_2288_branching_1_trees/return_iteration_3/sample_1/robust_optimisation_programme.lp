\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.1538 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.909142 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.18195 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.946267 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.04826 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.928109 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.37318 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.754326 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.05533 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.93576 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.81895 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.937689 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.959605 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.7924 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.20233 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.934233 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.16793 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.780521 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.16242 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.743183 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.34024 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.11257 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.754867 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.01815 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.749111 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.934373 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.685828 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.28324 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.31321 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.959194 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.691159 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.00436 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.19044 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.69568 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.05134 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.00739 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.45056 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.04873 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20415 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.913081 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.43934 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.861878 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.02458 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.770888 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.44459 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.616299 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.39108 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.840754 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.915512 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.794594 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.850388 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.07031 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.44141 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.02321 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.30305 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.857803 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.15864 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.599231 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.52901 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.11621 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.684219 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.882338 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.771641 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.01144 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.648905 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.29895 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.43987 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.33226 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.604568 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.83503 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.46063 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.99163 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.716548 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.10526 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.16214 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.975744 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31214 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.885121 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.40465 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.884243 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.06926 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.807004 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.70679 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.558992 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.46695 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.896277 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.792262 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.733845 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.03795 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.13807 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.6323 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.924584 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.07861 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.725556 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.41571 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.461544 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.54528 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.06388 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.658576 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.84473 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.6808 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.877089 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.531949 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.319 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.58843 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.3656 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.574408 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.926864 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.23189 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.7432 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.78426 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.771132 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.55227 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.800752 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18337 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.896164 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.17932 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.945076 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.03549 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.954538 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.56276 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.688061 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.36631 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.985267 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.891634 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.833788 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.03746 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.01765 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.49639 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.872705 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.2315 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.863721 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.18831 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.649506 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.26054 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.08299 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.7955 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.914124 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.733262 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.918606 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.642389 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.35873 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.36008 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.06427 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.587318 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.901707 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.17689 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.46355 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.88326 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.998889 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.31257 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.870411 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.10177 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.962784 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.40788 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.876657 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.19234 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.615252 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.738627 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.997086 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.20883 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.649376 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.26334 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.954468 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.937404 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.59899 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.948874 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.30539 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.836032 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.767241 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.18314 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.697125 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.81884 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.926102 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.513346 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.713523 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.28925 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.02302 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.729247 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.703836 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.22113 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.82785 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.10415 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.903705 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.04777 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.46201 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.740059 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.726961 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.20882 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.32779 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16258 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.992507 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.11676 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.938217 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.97803 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.811561 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.25804 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.770979 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.30585 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.918274 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.00303 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.811273 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.15476 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.16212 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.22932 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.871167 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.03705 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.897384 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.23131 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.627238 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.10679 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.93606 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.789755 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.871393 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.834587 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.00176 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.789082 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.10075 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.30119 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.02918 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.738098 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.00179 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.14737 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.2998 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.933914 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.09805 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.24057 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.999748 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13654 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.931334 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.29613 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.907398 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.11671 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.73645 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.19879 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.798481 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.24222 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.83298 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.05062 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.921361 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.04944 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.1881 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.26327 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.12297 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.979254 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.786868 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.26949 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.625744 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.44735 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.00173 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.655103 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.800961 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.950533 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.941501 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.679021 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.971569 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.33067 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.3833 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.851501 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.904572 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.12696 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.39441 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.877592 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.850123 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.31636 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.09892 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.09325 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.969333 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.61686 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.915512 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.05539 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.701232 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.643717 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.972291 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.19134 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.686667 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.12981 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.775664 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.786259 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.69407 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.20509 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.46599 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.968751 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.814765 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.05719 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.58381 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.81116 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.921983 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.611814 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.884791 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.20442 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.13822 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.690269 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.993301 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.4932 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 2.03842 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.789991 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.764076 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.09386 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.59733 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.510872 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.848762 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.883629 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.05227 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16404 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.899455 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.1259 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.896675 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.06552 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.941044 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.38371 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.773448 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.37087 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.999299 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.952969 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.883609 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.03928 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.08083 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.42199 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01161 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.1158 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.916944 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.32405 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.644579 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.30095 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0304 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.857884 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.917355 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.785716 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.01027 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.702209 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.12475 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.36304 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.21967 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.658065 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.950534 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.13031 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.29032 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.971763 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.974805 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.25096 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.921279 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13433 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.919701 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.14576 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.923206 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.05289 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.906074 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.21906 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.850063 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.19901 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.920523 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.00622 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.900339 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.964269 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08633 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.30156 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.956506 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.08807 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.89597 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.24987 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.691904 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.21038 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.06295 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.792171 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.91878 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.89547 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.01279 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.773909 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.104 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.11769 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.19535 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.869997 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.910049 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.13946 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.32979 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.86788 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.914106 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.3022 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.966585 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.0489999413450923 z_k1_s0000
                                 - 0.0559999800514931 z_k1_s0001
                                 - 0.112999965632472 z_k1_s0002
                                 - 0.109999784244158 z_k1_s0003
                                 - 0.109000273114242 z_k1_s0004
                                 - 0.075999925841353 z_k1_s0005
                                 - 0.0819999590182679 z_k1_s0006
                                 - 0.229000112653258 z_k1_s0007
                                 - 0.327998522459208 z_k1_s0010
                                 - 0.0769995138805571 z_k1_s0011
                                 - 0.135999251674114 z_k1_s0012
                                 - 0.323997433557559 z_k1_s0013
                                 - 0.0729995961721469 z_k1_s0014
                                 - 0.125999311003933 z_k1_s0015
                                 - 0.318997902419934 z_k1_s0016
                                 - 0.0809995779881221 z_k1_s0017
                                 - 0.140999796842286 z_k1_s0020
                                 - 0.281999278084984 z_k1_s0021
                                 - 0.340000170413162 z_k1_s0022
                                 - 0.194999517590659 z_k1_s0023
                                 - 0.119999800240279 z_k1_s0024
                                 - 0.334999494937376 z_k1_s0025
                                 - 0.0829998513128731 z_k1_s0026
                                 - 0.08199987401751 z_k1_s0027
                                 - 0.530998374678128 z_k1_s0030
                                 - 0.180999866235419 z_k1_s0031
                                 - 0.310998726148776 z_k1_s0032
                                 - 0.343999450427714 z_k1_s0033
                                 - 0.791998734705668 z_k1_s0034
                                 - 0.381000168316277 z_k1_s0035
                                 - 0.322998706977649 z_k1_s0036
                                 - 0.244999763989413 z_k1_s0037
                                 - 0.143999503387657 z_k1_s0040
                                 - 0.31899827704693 z_k1_s0041
                                 - 0.145999560698385 z_k1_s0042
                                 - 0.255999322599602 z_k1_s0043
                                 - 0.0739998090045565 z_k1_s0044
                                 - 0.190998615747129 z_k1_s0045
                                 - 0.202999884208452 z_k1_s0046
                                 - 0.20799887173908 z_k1_s0047
                                 - 0.0679998636569192 z_k1_s0050
                                 - 0.0989996727419426 z_k1_s0051
                                 - 0.0679998636569192 z_k1_s0052
                                 - 0.0459999034032186 z_k1_s0053
                                 - 0.0919996584067269 z_k1_s0054
                                 - 0.0889997582628561 z_k1_s0055
                                 - 0.122999615891461 z_k1_s0056
                                 - 0.156999473520065 z_k1_s0057
                                 - 0.107999854640214 z_k1_s0060
                                 - 0.244999850248246 z_k1_s0061
                                 - 0.438998969940329 z_k1_s0062
                                 - 0.249999807518704 z_k1_s0063
                                 - 0.145999724295274 z_k1_s0064
                                 - 0.159999604652555 z_k1_s0065
                                 - 0.369998782009684 z_k1_s0066
                                 - 0.225999235022177 z_k1_s0067
                                 - 0.24199989360656 z_k1_s0070
                                 - 0.185999394327319 z_k1_s0071
                                 - 0.200999284063583 z_k1_s0072
                                 - 0.412998726199688 z_k1_s0073
                                 - 0.258000164257377 z_k1_s0074
                                 - 0.235998952115095 z_k1_s0075
                                 - 0.491998802541983 z_k1_s0076
                                 - 0.211999442136106 z_k1_s0077
                                 - 0.300999349073492 z_k1_s0100
                                 - 0.661000350584174 z_k1_s0101
                                 - 0.615001034413756 z_k1_s0102
                                 - 0.175000080945275 z_k1_s0103
                                 - 0.221000208916084 z_k1_s0104
                                 - 0.171000069817378 z_k1_s0105
                                 - 0.300999349073492 z_k1_s0106
                                 - 0.261000049594918 z_k1_s0107
                                 - 1.03499842885061 z_k1_s0110
                                 - 0.306000590052328 z_k1_s0111
                                 - 0.371999967906151 z_k1_s0112
                                 - 1.27199936864642 z_k1_s0113
                                 - 1.27499817104793 z_k1_s0114
                                 - 0.679000893759571 z_k1_s0115
                                 - 1.28899993226087 z_k1_s0116
                                 - 1.12200044852383 z_k1_s0117
                                 - 0.151000111262355 z_k1_s0120
                                 - 0.30200022252471 z_k1_s0121
                                 - 0.163999938928602 z_k1_s0122
                                 - 0.229999757326656 z_k1_s0123
                                 - 0.520999580918242 z_k1_s0124
                                 - 0.322000947857661 z_k1_s0125
                                 - 0.268001135459264 z_k1_s0126
                                 - 0.187999950927915 z_k1_s0127
                                 - 0.403999963712379 z_k1_s0130
                                 - 1.15000135553876 z_k1_s0131
                                 - 1.68600171137923 z_k1_s0132
                                 - 0.467000425628258 z_k1_s0133
                                 - 0.62300020214279 z_k1_s0134
                                 - 0.840998514255701 z_k1_s0135
                                 - 1.26800349204243 z_k1_s0136
                                 - 1.1170042799089 z_k1_s0137
                                 - 0.161000156493954 z_k1_s0140
                                 - 0.701999190826254 z_k1_s0141
                                 - 0.284999640579018 z_k1_s0142
                                 - 0.193000171125662 z_k1_s0143
                                 - 0.373001187692024 z_k1_s0144
                                 - 0.368000771014972 z_k1_s0145
                                 - 0.169000099876848 z_k1_s0146
                                 - 0.160000073158543 z_k1_s0147
                                 - 0.152999945687412 z_k1_s0150
                                 - 0.318999266152053 z_k1_s0151
                                 - 0.14800010761923 z_k1_s0152
                                 - 0.560000241640294 z_k1_s0153
                                 - 0.171999942946634 z_k1_s0154
                                 - 0.138000023082779 z_k1_s0155
                                 - 0.363000291552576 z_k1_s0156
                                 - 0.189000004978583 z_k1_s0157
                                 - 0.483999161590145 z_k1_s0160
                                 - 0.2320003841919 z_k1_s0161
                                 - 0.155999962508373 z_k1_s0162
                                 - 0.275999460245626 z_k1_s0163
                                 - 0.31199955201678 z_k1_s0164
                                 - 0.102000011351625 z_k1_s0165
                                 - 0.155999962508373 z_k1_s0166
                                 - 0.147000032815576 z_k1_s0167
                                 - 0.374000137529265 z_k1_s0170
                                 - 0.317000450683296 z_k1_s0171
                                 - 0.660000460169596 z_k1_s0172
                                 - 0.506999899770551 z_k1_s0173
                                 - 0.63399942256451 z_k1_s0174
                                 - 0.412999845434291 z_k1_s0175
                                 - 0.543999527859802 z_k1_s0176
                                 - 0.248000284942673 z_k1_s0177
                                 - 0.184999749750223 z_k1_s1000
                                 - 0.145999777208302 z_k1_s1001
                                 - 0.122999933401429 z_k1_s1002
                                 - 0.527999828259713 z_k1_s1003
                                 - 0.328000169070234 z_k1_s1004
                                 - 0.523000031779839 z_k1_s1005
                                 - 0.306999983855494 z_k1_s1006
                                 - 0.45999947613562 z_k1_s1007
                                 - 0.626999431832609 z_k1_s1010
                                 - 0.637999302538627 z_k1_s1011
                                 - 0.99599926830991 z_k1_s1012
                                 - 0.58899895099506 z_k1_s1013
                                 - 1.8350008457423 z_k1_s1014
                                 - 0.493999449298577 z_k1_s1015
                                 - 1.43000019575629 z_k1_s1016
                                 - 1.8929928986761 z_k1_s1017
                                 - 0.284998917051429 z_k1_s1020
                                 - 0.29899870661894 z_k1_s1021
                                 - 0.338999988239534 z_k1_s1022
                                 - 0.628000351452417 z_k1_s1023
                                 - 0.524998698778629 z_k1_s1024
                                 - 0.221000029657024 z_k1_s1025
                                 - 0.1320000117499 z_k1_s1026
                                 - 0.206999976489635 z_k1_s1027
                                 - 1.91299889855641 z_k1_s1030
                                 - 0.912999358599249 z_k1_s1031
                                 - 1.35700405275925 z_k1_s1032
                                 - 1.61200394011883 z_k1_s1033
                                 - 0.3930000461894 z_k1_s1034
                                 - 1.63200301550023 z_k1_s1035
                                 - 0.595000048538919 z_k1_s1036
                                 - 0.926000158746765 z_k1_s1037
                                 - 0.975998197685743 z_k1_s1040
                                 - 0.2229999521354 z_k1_s1041
                                 - 0.359000187278535 z_k1_s1042
                                 - 0.496998899086885 z_k1_s1043
                                 - 0.577999002783161 z_k1_s1044
                                 - 0.550001314468593 z_k1_s1045
                                 - 0.353000193682516 z_k1_s1046
                                 - 0.266999905172875 z_k1_s1047
                                 - 0.621000750180441 z_k1_s1050
                                 - 0.615001840825557 z_k1_s1051
                                 - 0.489998119943789 z_k1_s1052
                                 - 0.521000346273893 z_k1_s1053
                                 - 0.489998119943789 z_k1_s1054
                                 - 0.421999903526429 z_k1_s1055
                                 - 0.770000108481747 z_k1_s1056
                                 - 0.358999777705246 z_k1_s1057
                                 - 1.79400258358604 z_k1_s1060
                                 - 0.846997137153332 z_k1_s1061
                                 - 1.83100221032691 z_k1_s1062
                                 - 1.37900131565065 z_k1_s1063
                                 - 0.783000396400734 z_k1_s1064
                                 - 0.936998729200512 z_k1_s1065
                                 - 0.404000606184833 z_k1_s1066
                                 - 0.434000576266821 z_k1_s1067
                                 - 0.341999915578437 z_k1_s1070
                                 - 0.396999704258104 z_k1_s1071
                                 - 0.402999888608515 z_k1_s1072
                                 - 0.39999979643331 z_k1_s1073
                                 - 0.982000567631268 z_k1_s1074
                                 - 0.962001311129254 z_k1_s1075
                                 - 0.303999698625385 z_k1_s1076
                                 - 0.284000034723565 z_k1_s1077
                                 - 0.17999974195456 z_k1_s1100
                                 - 0.0640000498062244 z_k1_s1101
                                 - 0.0520000343425505 z_k1_s1102
                                 - 0.107000440051432 z_k1_s1103
                                 - 0.108000041172956 z_k1_s1104
                                 - 0.0930001443435006 z_k1_s1105
                                 - 0.199000003272309 z_k1_s1106
                                 - 0.176999958588888 z_k1_s1107
                                 - 0.431001419117725 z_k1_s1110
                                 - 0.464000781711743 z_k1_s1111
                                 - 0.278000824754113 z_k1_s1112
                                 - 0.329000257273262 z_k1_s1113
                                 - 0.138000440392413 z_k1_s1114
                                 - 0.343002161864442 z_k1_s1115
                                 - 0.409000887052477 z_k1_s1116
                                 - 0.479001796985255 z_k1_s1117
                                 - 0.347001590909319 z_k1_s1120
                                 - 0.113000566249076 z_k1_s1121
                                 - 0.130000553244506 z_k1_s1122
                                 - 0.127000521633393 z_k1_s1123
                                 - 0.104000471415737 z_k1_s1124
                                 - 0.305002012957686 z_k1_s1125
                                 - 0.154000373831436 z_k1_s1126
                                 - 0.16100078382557 z_k1_s1127
                                 - 0.208000561795523 z_k1_s1130
                                 - 0.0900000555460009 z_k1_s1131
                                 - 0.0790000352125908 z_k1_s1132
                                 - 0.292000694905183 z_k1_s1133
                                 - 0.188000414007422 z_k1_s1134
                                 - 0.131999512600381 z_k1_s1135
                                 - 0.160999544042991 z_k1_s1136
                                 - 0.0690000832186307 z_k1_s1137
                                 - 0.187000483148805 z_k1_s1140
                                 - 0.29099991554106 z_k1_s1141
                                 - 0.106000143277294 z_k1_s1142
                                 - 0.0760000664101065 z_k1_s1143
                                 - 0.16700059230414 z_k1_s1144
                                 - 0.126999607613858 z_k1_s1145
                                 - 0.183000504979872 z_k1_s1146
                                 - 0.0660000006876789 z_k1_s1147
                                 - 0.146000039658134 z_k1_s1150
                                 - 0.223999295938415 z_k1_s1151
                                 - 0.239999778862652 z_k1_s1152
                                 - 0.0570000594548933 z_k1_s1153
                                 - 0.239999778862652 z_k1_s1154
                                 - 0.20699999589406 z_k1_s1155
                                 - 0.137999947496034 z_k1_s1156
                                 - 0.241000088982951 z_k1_s1157
                                 - 0.0999996508625949 z_k1_s1160
                                 - 0.20599975867511 z_k1_s1161
                                 - 0.0779997303728137 z_k1_s1162
                                 - 0.211999135629178 z_k1_s1163
                                 - 0.186999931660807 z_k1_s1164
                                 - 0.109999602448906 z_k1_s1165
                                 - 0.112999560924903 z_k1_s1166
                                 - 0.343998928566829 z_k1_s1167
                                 - 0.160000050462447 z_k1_s1170
                                 - 0.60700111558222 z_k1_s1171
                                 - 0.311000584306545 z_k1_s1172
                                 - 0.171000089351753 z_k1_s1173
                                 - 0.261999895872195 z_k1_s1174
                                 - 0.404998855832644 z_k1_s1175
                                 - 0.125999930259141 z_k1_s1176
                                 - 0.533999827098304 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.935185 w_k1_ADM_s000
                                 + 1.02991 w_k1_AXP_s000
                                 + 0.956438 w_k1_DIS_s000
                                 + 0.863296 w_k1_FDX_s000
                                 + 1.65514 w_k1_IBM_s000
                                 + 1.19285 w_k1_INTC_s000
                                 + 1.19743 w_k1_KO_s000 + 2.06059 w_k1_MAT_s000
                                 + 1.01287 w_k1_MSFT_s000
                                 + 1.10557 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.912832 w_k1_ADM_s000
                                 + 0.96254 w_k1_AXP_s000
                                 + 1.48712 w_k1_DIS_s000
                                 + 1.10867 w_k1_FDX_s000
                                 + 1.54765 w_k1_IBM_s000
                                 + 1.34534 w_k1_INTC_s000
                                 + 1.28216 w_k1_KO_s000 + 1.67578 w_k1_MAT_s000
                                 + 1.33698 w_k1_MSFT_s000
                                 + 0.942141 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.635253 w_k1_ADM_s000
                                 + 0.668478 w_k1_AXP_s000
                                 + 0.550746 w_k1_DIS_s000
                                 + 0.727068 w_k1_FDX_s000
                                 + 0.802802 w_k1_IBM_s000
                                 + 0.913745 w_k1_INTC_s000
                                 + 0.695623 w_k1_KO_s000
                                 + 0.898776 w_k1_MAT_s000
                                 + 0.824705 w_k1_MSFT_s000
                                 + 0.775654 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.21811 w_k1_ADM_s000
                                 + 1.45779 w_k1_AXP_s000
                                 + 1.55819 w_k1_DIS_s000
                                 + 1.40054 w_k1_FDX_s000
                                 + 1.40509 w_k1_IBM_s000
                                 + 1.46224 w_k1_INTC_s000
                                 + 1.31667 w_k1_KO_s000 + 2.2127 w_k1_MAT_s000
                                 + 1.49984 w_k1_MSFT_s000
                                 + 1.25139 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.828357 w_k1_ADM_s000
                                 + 1.07561 w_k1_AXP_s000
                                 + 0.918974 w_k1_DIS_s000
                                 + 0.980894 w_k1_FDX_s000
                                 + 1.17565 w_k1_IBM_s000
                                 + 0.975527 w_k1_INTC_s000
                                 + 0.944279 w_k1_KO_s000 + 2.1819 w_k1_MAT_s000
                                 + 0.950276 w_k1_MSFT_s000
                                 + 0.959207 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.11976 w_k1_ADM_s000
                                 + 0.770169 w_k1_AXP_s000
                                 + 1.02572 w_k1_DIS_s000
                                 + 1.09135 w_k1_FDX_s000
                                 + 0.896706 w_k1_IBM_s000
                                 + 1.18569 w_k1_INTC_s000
                                 + 0.88033 w_k1_KO_s000
                                 + 0.602953 w_k1_MAT_s000
                                 + 1.13247 w_k1_MSFT_s000
                                 + 0.842261 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.1538 w_k1_ADM_s000
                                 + 1.54862 w_k1_AXP_s000
                                 + 1.15659 w_k1_DIS_s000
                                 + 1.18481 w_k1_FDX_s000
                                 + 1.28664 w_k1_IBM_s000
                                 + 1.07245 w_k1_INTC_s000
                                 + 1.13822 w_k1_KO_s000 + 2.18257 w_k1_MAT_s000
                                 + 1.06381 w_k1_MSFT_s000 + 1.1445 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.815481 w_k1_ADM_s000
                                 + 0.777554 w_k1_AXP_s000
                                 + 0.866508 w_k1_DIS_s000
                                 + 0.894159 w_k1_FDX_s000
                                 + 0.991026 w_k1_IBM_s000
                                 + 1.07257 w_k1_INTC_s000
                                 + 0.969012 w_k1_KO_s000
                                 + 1.15172 w_k1_MAT_s000
                                 + 1.05203 w_k1_MSFT_s000
                                 + 0.910169 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.816632 w_k1_ADM_s001
                                 + 1.17748 w_k1_AXP_s001 + 1.2755 w_k1_DIS_s001
                                 + 1.13331 w_k1_FDX_s001
                                 + 0.974863 w_k1_IBM_s001
                                 + 1.0729 w_k1_INTC_s001
                                 + 0.999603 w_k1_KO_s001
                                 + 1.17126 w_k1_MAT_s001
                                 + 1.00826 w_k1_MSFT_s001
                                 + 0.992933 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.20808 w_k1_ADM_s001
                                 + 1.17484 w_k1_AXP_s001 + 1.0129 w_k1_DIS_s001
                                 + 0.946812 w_k1_FDX_s001
                                 + 1.34257 w_k1_IBM_s001
                                 + 1.07422 w_k1_INTC_s001
                                 + 1.26724 w_k1_KO_s001
                                 + 0.914788 w_k1_MAT_s001
                                 + 0.866929 w_k1_MSFT_s001
                                 + 1.04041 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.889404 w_k1_ADM_s001
                                 + 0.9875 w_k1_AXP_s001
                                 + 0.926805 w_k1_DIS_s001
                                 + 0.847197 w_k1_FDX_s001
                                 + 1.14881 w_k1_IBM_s001
                                 + 0.903666 w_k1_INTC_s001
                                 + 1.08866 w_k1_KO_s001 + 0.8548 w_k1_MAT_s001
                                 + 0.839592 w_k1_MSFT_s001
                                 + 0.916521 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.04273 w_k1_ADM_s001
                                 + 0.909968 w_k1_AXP_s001
                                 + 0.952244 w_k1_DIS_s001
                                 + 1.00784 w_k1_FDX_s001
                                 + 0.688328 w_k1_IBM_s001
                                 + 0.917592 w_k1_INTC_s001
                                 + 0.830607 w_k1_KO_s001
                                 + 0.565906 w_k1_MAT_s001
                                 + 0.939555 w_k1_MSFT_s001
                                 + 0.892388 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.0479 w_k1_ADM_s001
                                 + 1.40791 w_k1_AXP_s001
                                 + 1.17365 w_k1_DIS_s001
                                 + 1.10019 w_k1_FDX_s001
                                 + 1.44767 w_k1_IBM_s001
                                 + 1.28839 w_k1_INTC_s001
                                 + 1.44643 w_k1_KO_s001 + 1.88634 w_k1_MAT_s001
                                 + 1.10828 w_k1_MSFT_s001 + 1.0675 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.17306 w_k1_ADM_s001
                                 + 1.59867 w_k1_AXP_s001 + 1.629 w_k1_DIS_s001
                                 + 1.48775 w_k1_FDX_s001
                                 + 1.20949 w_k1_IBM_s001
                                 + 1.27598 w_k1_INTC_s001
                                 + 1.31962 w_k1_KO_s001 + 1.20932 w_k1_MAT_s001
                                 + 1.21582 w_k1_MSFT_s001
                                 + 1.11561 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.873486 w_k1_ADM_s001
                                 + 0.875054 w_k1_AXP_s001
                                 + 0.753329 w_k1_DIS_s001
                                 + 0.806392 w_k1_FDX_s001
                                 + 0.884319 w_k1_IBM_s001
                                 + 1.03288 w_k1_INTC_s001
                                 + 0.836868 w_k1_KO_s001
                                 + 0.753796 w_k1_MAT_s001
                                 + 0.819191 w_k1_MSFT_s001
                                 + 0.82667 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.805051 w_k1_ADM_s001
                                 + 1.09652 w_k1_AXP_s001
                                 + 0.967407 w_k1_DIS_s001
                                 + 0.882695 w_k1_FDX_s001
                                 + 1.53666 w_k1_IBM_s001
                                 + 0.88214 w_k1_INTC_s001
                                 + 1.37962 w_k1_KO_s001 + 1.28862 w_k1_MAT_s001
                                 + 0.972571 w_k1_MSFT_s001
                                 + 1.09514 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.818508 w_k1_ADM_s002
                                 + 0.829219 w_k1_AXP_s002
                                 + 0.948819 w_k1_DIS_s002
                                 + 0.851146 w_k1_FDX_s002
                                 + 1.24218 w_k1_IBM_s002
                                 + 1.0179 w_k1_INTC_s002 + 1.24183 w_k1_KO_s002
                                 + 1.57989 w_k1_MAT_s002
                                 + 0.98568 w_k1_MSFT_s002
                                 + 1.04665 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.65244 w_k1_ADM_s002
                                 + 1.24451 w_k1_AXP_s002
                                 + 1.29072 w_k1_DIS_s002
                                 + 1.25878 w_k1_FDX_s002
                                 + 0.990484 w_k1_IBM_s002
                                 + 0.899767 w_k1_INTC_s002
                                 + 1.27532 w_k1_KO_s002 + 1.01434 w_k1_MAT_s002
                                 + 0.939351 w_k1_MSFT_s002
                                 + 1.1941 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.889714 w_k1_ADM_s002
                                 + 0.856016 w_k1_AXP_s002
                                 + 0.736562 w_k1_DIS_s002
                                 + 0.830075 w_k1_FDX_s002
                                 + 0.793973 w_k1_IBM_s002
                                 + 0.75449 w_k1_INTC_s002
                                 + 0.88349 w_k1_KO_s002
                                 + 0.988619 w_k1_MAT_s002
                                 + 0.679765 w_k1_MSFT_s002
                                 + 0.812349 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.977573 w_k1_ADM_s002
                                 + 1.66223 w_k1_AXP_s002 + 1.6425 w_k1_DIS_s002
                                 + 1.1799 w_k1_FDX_s002 + 1.64615 w_k1_IBM_s002
                                 + 1.44134 w_k1_INTC_s002
                                 + 1.70468 w_k1_KO_s002 + 2.06406 w_k1_MAT_s002
                                 + 1.20635 w_k1_MSFT_s002
                                 + 1.30498 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.02419 w_k1_ADM_s002
                                 + 0.643114 w_k1_AXP_s002
                                 + 0.715167 w_k1_DIS_s002
                                 + 0.892706 w_k1_FDX_s002
                                 + 0.58251 w_k1_IBM_s002 + 0.715 w_k1_INTC_s002
                                 + 0.752356 w_k1_KO_s002
                                 + 0.652704 w_k1_MAT_s002
                                 + 0.783674 w_k1_MSFT_s002
                                 + 0.941735 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.925433 w_k1_ADM_s002
                                 + 0.978568 w_k1_AXP_s002
                                 + 1.04209 w_k1_DIS_s002
                                 + 1.04929 w_k1_FDX_s002
                                 + 0.916828 w_k1_IBM_s002
                                 + 1.0684 w_k1_INTC_s002 + 1.04816 w_k1_KO_s002
                                 + 1.03533 w_k1_MAT_s002
                                 + 0.971085 w_k1_MSFT_s002
                                 + 0.968711 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.60978 w_k1_ADM_s002
                                 + 0.987182 w_k1_AXP_s002
                                 + 1.66428 w_k1_DIS_s002
                                 + 1.15415 w_k1_FDX_s002
                                 + 1.06715 w_k1_IBM_s002
                                 + 1.11075 w_k1_INTC_s002
                                 + 1.54769 w_k1_KO_s002 + 1.47138 w_k1_MAT_s002
                                 + 1.38718 w_k1_MSFT_s002 + 1.2255 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.27714 w_k1_ADM_s002
                                 + 0.679159 w_k1_AXP_s002
                                 + 1.02341 w_k1_DIS_s002
                                 + 0.937722 w_k1_FDX_s002
                                 + 0.703673 w_k1_IBM_s002
                                 + 0.978936 w_k1_INTC_s002
                                 + 1.07206 w_k1_KO_s002 + 0.56542 w_k1_MAT_s002
                                 + 0.944288 w_k1_MSFT_s002
                                 + 0.959972 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.655645 w_k1_ADM_s003
                                 + 0.59395 w_k1_AXP_s003
                                 + 0.611373 w_k1_DIS_s003
                                 + 0.733579 w_k1_FDX_s003
                                 + 0.788959 w_k1_IBM_s003
                                 + 0.801745 w_k1_INTC_s003
                                 + 0.750924 w_k1_KO_s003
                                 + 0.829848 w_k1_MAT_s003
                                 + 0.744514 w_k1_MSFT_s003
                                 + 0.83126 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.732538 w_k1_ADM_s003
                                 + 0.886466 w_k1_AXP_s003
                                 + 0.633592 w_k1_DIS_s003
                                 + 0.755002 w_k1_FDX_s003
                                 + 1.19744 w_k1_IBM_s003
                                 + 0.760909 w_k1_INTC_s003
                                 + 0.891063 w_k1_KO_s003
                                 + 1.30523 w_k1_MAT_s003
                                 + 0.785547 w_k1_MSFT_s003
                                 + 0.945041 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.39194 w_k1_ADM_s003
                                 + 1.18753 w_k1_AXP_s003
                                 + 1.59822 w_k1_DIS_s003
                                 + 1.47781 w_k1_FDX_s003
                                 + 0.989373 w_k1_IBM_s003
                                 + 1.372 w_k1_INTC_s003 + 1.31316 w_k1_KO_s003
                                 + 1.26679 w_k1_MAT_s003
                                 + 1.28543 w_k1_MSFT_s003 + 1.2905 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.931296 w_k1_ADM_s003
                                 + 0.862294 w_k1_AXP_s003
                                 + 0.718208 w_k1_DIS_s003
                                 + 0.920333 w_k1_FDX_s003
                                 + 0.737326 w_k1_IBM_s003
                                 + 0.894965 w_k1_INTC_s003
                                 + 0.73341 w_k1_KO_s003
                                 + 0.841843 w_k1_MAT_s003
                                 + 0.860405 w_k1_MSFT_s003
                                 + 0.988743 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0688 w_k1_ADM_s003
                                 + 1.31964 w_k1_AXP_s003
                                 + 1.21133 w_k1_DIS_s003
                                 + 1.23111 w_k1_FDX_s003
                                 + 1.02311 w_k1_IBM_s003 + 1.384 w_k1_INTC_s003
                                 + 1.09608 w_k1_KO_s003 + 1.20489 w_k1_MAT_s003
                                 + 1.14945 w_k1_MSFT_s003
                                 + 1.07907 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.986237 w_k1_ADM_s003
                                 + 0.834085 w_k1_AXP_s003
                                 + 0.946343 w_k1_DIS_s003
                                 + 0.951189 w_k1_FDX_s003
                                 + 1.00759 w_k1_IBM_s003
                                 + 0.899076 w_k1_INTC_s003
                                 + 1.09966 w_k1_KO_s003 + 1.20402 w_k1_MAT_s003
                                 + 0.919993 w_k1_MSFT_s003
                                 + 1.06747 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.04981 w_k1_ADM_s003
                                 + 0.88447 w_k1_AXP_s003
                                 + 1.17324 w_k1_DIS_s003
                                 + 1.02153 w_k1_FDX_s003
                                 + 1.57173 w_k1_IBM_s003
                                 + 0.927241 w_k1_INTC_s003
                                 + 1.52523 w_k1_KO_s003 + 1.3279 w_k1_MAT_s003
                                 + 1.07187 w_k1_MSFT_s003
                                 + 1.13124 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.38177 w_k1_ADM_s003
                                 + 1.51469 w_k1_AXP_s003
                                 + 1.03603 w_k1_DIS_s003
                                 + 1.17316 w_k1_FDX_s003 + 1.0988 w_k1_IBM_s003
                                 + 1.10163 w_k1_INTC_s003
                                 + 0.903536 w_k1_KO_s003
                                 + 1.17102 w_k1_MAT_s003
                                 + 1.01172 w_k1_MSFT_s003
                                 + 0.980322 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.94065 w_k1_ADM_s004
                                 + 1.14937 w_k1_AXP_s004
                                 + 1.27488 w_k1_DIS_s004
                                 + 1.05121 w_k1_FDX_s004
                                 + 1.62009 w_k1_IBM_s004
                                 + 1.08752 w_k1_INTC_s004
                                 + 1.37617 w_k1_KO_s004 + 1.87184 w_k1_MAT_s004
                                 + 1.15661 w_k1_MSFT_s004
                                 + 1.18665 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.06045 w_k1_ADM_s004
                                 + 1.07321 w_k1_AXP_s004
                                 + 1.38549 w_k1_DIS_s004
                                 + 1.13817 w_k1_FDX_s004
                                 + 1.13379 w_k1_IBM_s004
                                 + 0.962634 w_k1_INTC_s004
                                 + 1.08869 w_k1_KO_s004 + 1.20889 w_k1_MAT_s004
                                 + 0.970874 w_k1_MSFT_s004
                                 + 1.01827 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.983573 w_k1_ADM_s004
                                 + 1.00653 w_k1_AXP_s004
                                 + 1.30569 w_k1_DIS_s004
                                 + 1.14507 w_k1_FDX_s004
                                 + 0.792327 w_k1_IBM_s004
                                 + 1.01481 w_k1_INTC_s004
                                 + 1.10835 w_k1_KO_s004
                                 + 0.972143 w_k1_MAT_s004
                                 + 1.16476 w_k1_MSFT_s004
                                 + 1.08425 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.26164 w_k1_ADM_s004
                                 + 0.954252 w_k1_AXP_s004
                                 + 0.821129 w_k1_DIS_s004
                                 + 0.822408 w_k1_FDX_s004
                                 + 1.0961 w_k1_IBM_s004
                                 + 0.645312 w_k1_INTC_s004
                                 + 0.811732 w_k1_KO_s004
                                 + 1.21562 w_k1_MAT_s004
                                 + 0.826735 w_k1_MSFT_s004
                                 + 0.897625 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.906011 w_k1_ADM_s004
                                 + 0.726126 w_k1_AXP_s004
                                 + 1.09211 w_k1_DIS_s004
                                 + 0.863241 w_k1_FDX_s004
                                 + 0.960044 w_k1_IBM_s004
                                 + 1.04155 w_k1_INTC_s004
                                 + 0.953451 w_k1_KO_s004
                                 + 0.641563 w_k1_MAT_s004
                                 + 1.04005 w_k1_MSFT_s004
                                 + 0.890608 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.07984 w_k1_ADM_s004
                                 + 1.37418 w_k1_AXP_s004
                                 + 1.18442 w_k1_DIS_s004
                                 + 1.18976 w_k1_FDX_s004
                                 + 0.912801 w_k1_IBM_s004
                                 + 1.00588 w_k1_INTC_s004
                                 + 0.924348 w_k1_KO_s004
                                 + 1.32547 w_k1_MAT_s004
                                 + 1.17299 w_k1_MSFT_s004
                                 + 0.940758 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.05083 w_k1_ADM_s004
                                 + 0.758691 w_k1_AXP_s004
                                 + 0.795876 w_k1_DIS_s004
                                 + 0.735353 w_k1_FDX_s004
                                 + 0.852638 w_k1_IBM_s004
                                 + 0.816614 w_k1_INTC_s004
                                 + 0.810588 w_k1_KO_s004
                                 + 0.839834 w_k1_MAT_s004
                                 + 0.764483 w_k1_MSFT_s004
                                 + 0.772491 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.09857 w_k1_ADM_s004
                                 + 0.778075 w_k1_AXP_s004
                                 + 0.820477 w_k1_DIS_s004
                                 + 0.854965 w_k1_FDX_s004
                                 + 0.675162 w_k1_IBM_s004
                                 + 0.907986 w_k1_INTC_s004
                                 + 0.82562 w_k1_KO_s004
                                 + 0.769277 w_k1_MAT_s004
                                 + 0.896209 w_k1_MSFT_s004
                                 + 0.899855 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.179 w_k1_ADM_s005
                                 + 1.1373 w_k1_AXP_s005 + 1.48976 w_k1_DIS_s005
                                 + 1.11684 w_k1_FDX_s005
                                 + 1.40989 w_k1_IBM_s005
                                 + 1.32197 w_k1_INTC_s005
                                 + 1.30768 w_k1_KO_s005 + 1.18754 w_k1_MAT_s005
                                 + 1.3054 w_k1_MSFT_s005 + 1.18082 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.761354 w_k1_ADM_s005
                                 + 0.804038 w_k1_AXP_s005
                                 + 0.883498 w_k1_DIS_s005
                                 + 0.936637 w_k1_FDX_s005
                                 + 0.788628 w_k1_IBM_s005
                                 + 0.936116 w_k1_INTC_s005
                                 + 0.891604 w_k1_KO_s005
                                 + 0.875723 w_k1_MAT_s005
                                 + 0.881093 w_k1_MSFT_s005
                                 + 0.890075 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.734527 w_k1_ADM_s005
                                 + 0.888918 w_k1_AXP_s005
                                 + 0.717121 w_k1_DIS_s005
                                 + 0.788982 w_k1_FDX_s005
                                 + 0.981813 w_k1_IBM_s005
                                 + 1.04028 w_k1_INTC_s005
                                 + 0.875364 w_k1_KO_s005 + 1.0602 w_k1_MAT_s005
                                 + 0.823748 w_k1_MSFT_s005
                                 + 0.730679 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.719985 w_k1_ADM_s005
                                 + 0.715277 w_k1_AXP_s005
                                 + 0.591057 w_k1_DIS_s005
                                 + 0.794678 w_k1_FDX_s005
                                 + 0.668122 w_k1_IBM_s005
                                 + 0.916407 w_k1_INTC_s005
                                 + 0.596086 w_k1_KO_s005
                                 + 0.399312 w_k1_MAT_s005
                                 + 0.821617 w_k1_MSFT_s005
                                 + 0.724591 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.803487 w_k1_ADM_s005
                                 + 1.18692 w_k1_AXP_s005
                                 + 0.886877 w_k1_DIS_s005
                                 + 0.969824 w_k1_FDX_s005
                                 + 1.38705 w_k1_IBM_s005
                                 + 0.994737 w_k1_INTC_s005
                                 + 0.930955 w_k1_KO_s005
                                 + 1.35077 w_k1_MAT_s005
                                 + 0.743235 w_k1_MSFT_s005
                                 + 0.932553 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.71363 w_k1_ADM_s005
                                 + 1.00701 w_k1_AXP_s005
                                 + 1.05056 w_k1_DIS_s005
                                 + 1.09637 w_k1_FDX_s005
                                 + 0.750486 w_k1_IBM_s005
                                 + 1.26596 w_k1_INTC_s005
                                 + 0.952146 w_k1_KO_s005
                                 + 0.897782 w_k1_MAT_s005
                                 + 1.07891 w_k1_MSFT_s005
                                 + 1.00658 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.08991 w_k1_ADM_s005
                                 + 1.42363 w_k1_AXP_s005
                                 + 1.40225 w_k1_DIS_s005
                                 + 1.18994 w_k1_FDX_s005
                                 + 1.20637 w_k1_IBM_s005
                                 + 1.02008 w_k1_INTC_s005
                                 + 1.06481 w_k1_KO_s005 + 1.76178 w_k1_MAT_s005
                                 + 1.10112 w_k1_MSFT_s005
                                 + 1.08538 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.976536 w_k1_ADM_s005
                                 + 0.986498 w_k1_AXP_s005
                                 + 1.0581 w_k1_DIS_s005 + 1.08734 w_k1_FDX_s005
                                 + 0.934689 w_k1_IBM_s005
                                 + 1.01451 w_k1_INTC_s005
                                 + 1.04315 w_k1_KO_s005 + 1.23455 w_k1_MAT_s005
                                 + 0.930041 w_k1_MSFT_s005
                                 + 0.960373 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.66534 w_k1_ADM_s006
                                 + 0.789102 w_k1_AXP_s006
                                 + 0.77543 w_k1_DIS_s006
                                 + 0.780127 w_k1_FDX_s006
                                 + 1.16006 w_k1_IBM_s006
                                 + 0.790465 w_k1_INTC_s006
                                 + 0.868197 w_k1_KO_s006
                                 + 1.36617 w_k1_MAT_s006
                                 + 0.805247 w_k1_MSFT_s006
                                 + 0.884268 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.707418 w_k1_ADM_s006
                                 + 0.713873 w_k1_AXP_s006
                                 + 0.597389 w_k1_DIS_s006
                                 + 0.731604 w_k1_FDX_s006
                                 + 0.818672 w_k1_IBM_s006
                                 + 0.744028 w_k1_INTC_s006
                                 + 0.758934 w_k1_KO_s006
                                 + 0.97828 w_k1_MAT_s006
                                 + 0.701825 w_k1_MSFT_s006
                                 + 0.932499 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.04671 w_k1_ADM_s006
                                 + 1.31776 w_k1_AXP_s006
                                 + 1.35044 w_k1_DIS_s006
                                 + 1.20341 w_k1_FDX_s006
                                 + 1.37079 w_k1_IBM_s006
                                 + 1.29587 w_k1_INTC_s006
                                 + 1.23077 w_k1_KO_s006 + 1.5043 w_k1_MAT_s006
                                 + 1.17479 w_k1_MSFT_s006
                                 + 1.10373 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.987045 w_k1_ADM_s006
                                 + 0.964176 w_k1_AXP_s006
                                 + 0.655555 w_k1_DIS_s006
                                 + 0.944025 w_k1_FDX_s006
                                 + 0.731593 w_k1_IBM_s006
                                 + 0.968415 w_k1_INTC_s006
                                 + 0.683178 w_k1_KO_s006
                                 + 0.753698 w_k1_MAT_s006
                                 + 0.912978 w_k1_MSFT_s006
                                 + 0.898651 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.06882 w_k1_ADM_s006
                                 + 1.26684 w_k1_AXP_s006
                                 + 0.845044 w_k1_DIS_s006
                                 + 0.972138 w_k1_FDX_s006
                                 + 1.01056 w_k1_IBM_s006
                                 + 0.913471 w_k1_INTC_s006
                                 + 0.865726 w_k1_KO_s006
                                 + 1.15097 w_k1_MAT_s006
                                 + 0.963236 w_k1_MSFT_s006
                                 + 1.03787 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.00236 w_k1_ADM_s006
                                 + 0.892592 w_k1_AXP_s006
                                 + 0.99581 w_k1_DIS_s006
                                 + 0.962696 w_k1_FDX_s006
                                 + 0.850128 w_k1_IBM_s006
                                 + 0.807402 w_k1_INTC_s006
                                 + 0.85926 w_k1_KO_s006
                                 + 0.834525 w_k1_MAT_s006
                                 + 0.891405 w_k1_MSFT_s006
                                 + 0.982798 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.08194 w_k1_ADM_s006
                                 + 1.19708 w_k1_AXP_s006
                                 + 1.32967 w_k1_DIS_s006
                                 + 1.20646 w_k1_FDX_s006
                                 + 0.948252 w_k1_IBM_s006
                                 + 1.01426 w_k1_INTC_s006
                                 + 1.03724 w_k1_KO_s006 + 1.20233 w_k1_MAT_s006
                                 + 1.26554 w_k1_MSFT_s006 + 1.0961 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.21665 w_k1_ADM_s006
                                 + 1.0451 w_k1_AXP_s006
                                 + 0.875927 w_k1_DIS_s006
                                 + 1.02635 w_k1_FDX_s006
                                 + 0.990946 w_k1_IBM_s006
                                 + 0.879734 w_k1_INTC_s006
                                 + 0.908112 w_k1_KO_s006
                                 + 1.33269 w_k1_MAT_s006
                                 + 1.10552 w_k1_MSFT_s006
                                 + 1.11227 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.5959 w_k1_ADM_s007
                                 + 1.50613 w_k1_AXP_s007
                                 + 1.25552 w_k1_DIS_s007
                                 + 1.24767 w_k1_FDX_s007
                                 + 0.860431 w_k1_IBM_s007
                                 + 1.00883 w_k1_INTC_s007 + 1.1572 w_k1_KO_s007
                                 + 0.70497 w_k1_MAT_s007
                                 + 1.15699 w_k1_MSFT_s007
                                 + 1.03154 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.844862 w_k1_ADM_s007
                                 + 0.748711 w_k1_AXP_s007
                                 + 0.708843 w_k1_DIS_s007
                                 + 0.751344 w_k1_FDX_s007
                                 + 1.16844 w_k1_IBM_s007
                                 + 0.81341 w_k1_INTC_s007 + 1.1164 w_k1_KO_s007
                                 + 0.963884 w_k1_MAT_s007
                                 + 0.756641 w_k1_MSFT_s007
                                 + 0.890254 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 2.10826 w_k1_ADM_s007
                                 + 1.61177 w_k1_AXP_s007 + 1.8702 w_k1_DIS_s007
                                 + 1.42684 w_k1_FDX_s007
                                 + 1.38595 w_k1_IBM_s007
                                 + 1.14276 w_k1_INTC_s007
                                 + 1.75876 w_k1_KO_s007 + 1.80228 w_k1_MAT_s007
                                 + 1.45729 w_k1_MSFT_s007
                                 + 1.32181 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.37576 w_k1_ADM_s007
                                 + 1.05179 w_k1_AXP_s007
                                 + 1.21027 w_k1_DIS_s007
                                 + 1.07724 w_k1_FDX_s007
                                 + 1.13646 w_k1_IBM_s007
                                 + 0.887848 w_k1_INTC_s007
                                 + 1.35629 w_k1_KO_s007
                                 + 0.859193 w_k1_MAT_s007
                                 + 1.05885 w_k1_MSFT_s007
                                 + 1.13775 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.04313 w_k1_ADM_s007
                                 + 0.648714 w_k1_AXP_s007
                                 + 0.605557 w_k1_DIS_s007
                                 + 0.762546 w_k1_FDX_s007
                                 + 0.676519 w_k1_IBM_s007
                                 + 0.634412 w_k1_INTC_s007
                                 + 0.705635 w_k1_KO_s007
                                 + 0.480421 w_k1_MAT_s007
                                 + 0.734636 w_k1_MSFT_s007
                                 + 0.759095 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.18886 w_k1_ADM_s007
                                 + 0.933495 w_k1_AXP_s007
                                 + 0.983474 w_k1_DIS_s007
                                 + 1.10404 w_k1_FDX_s007
                                 + 0.680015 w_k1_IBM_s007
                                 + 1.09021 w_k1_INTC_s007 + 1.009 w_k1_KO_s007
                                 + 0.522899 w_k1_MAT_s007
                                 + 1.04597 w_k1_MSFT_s007
                                 + 0.980987 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.06219 w_k1_ADM_s007
                                 + 0.833215 w_k1_AXP_s007
                                 + 0.928865 w_k1_DIS_s007
                                 + 0.920531 w_k1_FDX_s007
                                 + 0.886824 w_k1_IBM_s007
                                 + 0.983002 w_k1_INTC_s007
                                 + 1.02526 w_k1_KO_s007
                                 + 0.922884 w_k1_MAT_s007
                                 + 0.927081 w_k1_MSFT_s007
                                 + 1.00595 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.25562 w_k1_ADM_s007
                                 + 1.0345 w_k1_AXP_s007
                                 + 0.999713 w_k1_DIS_s007
                                 + 0.869806 w_k1_FDX_s007
                                 + 1.63523 w_k1_IBM_s007
                                 + 0.877411 w_k1_INTC_s007
                                 + 1.53345 w_k1_KO_s007 + 1.39584 w_k1_MAT_s007
                                 + 0.901245 w_k1_MSFT_s007
                                 + 1.12852 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.02643 w_k1_ADM_s010
                                 + 1.49465 w_k1_AXP_s010
                                 + 1.11386 w_k1_DIS_s010
                                 + 1.02069 w_k1_FDX_s010
                                 + 1.23769 w_k1_IBM_s010
                                 + 1.27511 w_k1_INTC_s010
                                 + 1.27687 w_k1_KO_s010 + 1.32677 w_k1_MAT_s010
                                 + 1.08367 w_k1_MSFT_s010 + 1.0506 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.20847 w_k1_ADM_s010
                                 + 1.12548 w_k1_AXP_s010 + 1.0666 w_k1_DIS_s010
                                 + 1.05734 w_k1_FDX_s010
                                 + 0.879378 w_k1_IBM_s010
                                 + 1.16079 w_k1_INTC_s010
                                 + 0.987242 w_k1_KO_s010
                                 + 0.906979 w_k1_MAT_s010
                                 + 1.10913 w_k1_MSFT_s010
                                 + 0.989752 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.903007 w_k1_ADM_s010
                                 + 0.87751 w_k1_AXP_s010
                                 + 0.745944 w_k1_DIS_s010
                                 + 0.893535 w_k1_FDX_s010
                                 + 0.614798 w_k1_IBM_s010
                                 + 1.01678 w_k1_INTC_s010
                                 + 0.688708 w_k1_KO_s010
                                 + 0.550919 w_k1_MAT_s010
                                 + 0.901948 w_k1_MSFT_s010
                                 + 0.765585 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.42877 w_k1_ADM_s010
                                 + 1.10367 w_k1_AXP_s010
                                 + 1.46241 w_k1_DIS_s010
                                 + 1.14204 w_k1_FDX_s010
                                 + 1.25929 w_k1_IBM_s010
                                 + 1.06731 w_k1_INTC_s010
                                 + 1.23074 w_k1_KO_s010 + 1.09307 w_k1_MAT_s010
                                 + 1.02189 w_k1_MSFT_s010
                                 + 1.09261 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.971882 w_k1_ADM_s010
                                 + 1.14651 w_k1_AXP_s010
                                 + 0.977616 w_k1_DIS_s010
                                 + 0.986714 w_k1_FDX_s010
                                 + 1.57114 w_k1_IBM_s010
                                 + 1.05423 w_k1_INTC_s010
                                 + 1.09284 w_k1_KO_s010 + 1.40797 w_k1_MAT_s010
                                 + 1.00822 w_k1_MSFT_s010
                                 + 0.948264 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.11963 w_k1_ADM_s010
                                 + 0.865361 w_k1_AXP_s010
                                 + 1.31095 w_k1_DIS_s010 + 1.046 w_k1_FDX_s010
                                 + 0.78222 w_k1_IBM_s010
                                 + 1.14758 w_k1_INTC_s010
                                 + 1.03331 w_k1_KO_s010
                                 + 0.520394 w_k1_MAT_s010
                                 + 1.15855 w_k1_MSFT_s010
                                 + 0.872221 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.31223 w_k1_ADM_s010
                                 + 1.91538 w_k1_AXP_s010 + 1.6572 w_k1_DIS_s010
                                 + 1.41078 w_k1_FDX_s010
                                 + 1.38031 w_k1_IBM_s010
                                 + 1.57314 w_k1_INTC_s010
                                 + 1.36659 w_k1_KO_s010 + 1.30894 w_k1_MAT_s010
                                 + 1.30662 w_k1_MSFT_s010
                                 + 1.14767 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.81584 w_k1_ADM_s010
                                 + 0.830311 w_k1_AXP_s010
                                 + 0.668488 w_k1_DIS_s010
                                 + 0.684216 w_k1_FDX_s010
                                 + 0.964756 w_k1_IBM_s010
                                 + 0.842572 w_k1_INTC_s010
                                 + 0.787611 w_k1_KO_s010
                                 + 0.619048 w_k1_MAT_s010
                                 + 0.740161 w_k1_MSFT_s010
                                 + 0.74945 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.960642 w_k1_ADM_s011
                                 + 0.954324 w_k1_AXP_s011
                                 + 0.998792 w_k1_DIS_s011
                                 + 1.08424 w_k1_FDX_s011
                                 + 0.786019 w_k1_IBM_s011
                                 + 1.20734 w_k1_INTC_s011
                                 + 0.921676 w_k1_KO_s011
                                 + 0.76438 w_k1_MAT_s011
                                 + 0.951418 w_k1_MSFT_s011
                                 + 1.00341 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.872403 w_k1_ADM_s011
                                 + 1.04549 w_k1_AXP_s011
                                 + 0.701672 w_k1_DIS_s011
                                 + 0.911267 w_k1_FDX_s011
                                 + 0.736549 w_k1_IBM_s011
                                 + 1.01705 w_k1_INTC_s011
                                 + 0.667072 w_k1_KO_s011
                                 + 0.621609 w_k1_MAT_s011
                                 + 0.756892 w_k1_MSFT_s011
                                 + 0.790821 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.08845 w_k1_ADM_s011
                                 + 1.4113 w_k1_AXP_s011 + 0.93557 w_k1_DIS_s011
                                 + 1.00698 w_k1_FDX_s011 + 1.2274 w_k1_IBM_s011
                                 + 0.891987 w_k1_INTC_s011
                                 + 1.05477 w_k1_KO_s011 + 1.23062 w_k1_MAT_s011
                                 + 0.851912 w_k1_MSFT_s011
                                 + 1.04087 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.880369 w_k1_ADM_s011
                                 + 0.901817 w_k1_AXP_s011
                                 + 0.859333 w_k1_DIS_s011
                                 + 0.820725 w_k1_FDX_s011
                                 + 1.23657 w_k1_IBM_s011
                                 + 0.912144 w_k1_INTC_s011
                                 + 0.997136 w_k1_KO_s011
                                 + 0.973009 w_k1_MAT_s011
                                 + 0.745185 w_k1_MSFT_s011
                                 + 1.02154 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.784165 w_k1_ADM_s011
                                 + 0.633237 w_k1_AXP_s011
                                 + 0.681984 w_k1_DIS_s011
                                 + 0.830069 w_k1_FDX_s011
                                 + 0.710689 w_k1_IBM_s011
                                 + 1.10102 w_k1_INTC_s011
                                 + 0.724053 w_k1_KO_s011
                                 + 0.488564 w_k1_MAT_s011
                                 + 0.890404 w_k1_MSFT_s011
                                 + 0.959434 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.842173 w_k1_ADM_s011
                                 + 0.844887 w_k1_AXP_s011
                                 + 0.876352 w_k1_DIS_s011
                                 + 0.868683 w_k1_FDX_s011
                                 + 0.963096 w_k1_IBM_s011
                                 + 1.1436 w_k1_INTC_s011
                                 + 0.935294 w_k1_KO_s011
                                 + 0.735661 w_k1_MAT_s011
                                 + 0.946857 w_k1_MSFT_s011
                                 + 0.950526 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.49216 w_k1_ADM_s011
                                 + 1.31545 w_k1_AXP_s011
                                 + 1.39076 w_k1_DIS_s011
                                 + 1.23225 w_k1_FDX_s011
                                 + 1.50847 w_k1_IBM_s011
                                 + 0.996123 w_k1_INTC_s011
                                 + 1.26982 w_k1_KO_s011 + 1.39137 w_k1_MAT_s011
                                 + 1.10293 w_k1_MSFT_s011
                                 + 1.27732 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.11617 w_k1_ADM_s011
                                 + 1.00328 w_k1_AXP_s011
                                 + 1.44408 w_k1_DIS_s011
                                 + 1.25156 w_k1_FDX_s011
                                 + 0.99779 w_k1_IBM_s011
                                 + 1.11786 w_k1_INTC_s011
                                 + 1.04524 w_k1_KO_s011
                                 + 0.788936 w_k1_MAT_s011
                                 + 1.12772 w_k1_MSFT_s011
                                 + 1.11096 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.849658 w_k1_ADM_s012
                                 + 0.916273 w_k1_AXP_s012
                                 + 0.972259 w_k1_DIS_s012
                                 + 0.855499 w_k1_FDX_s012
                                 + 1.17695 w_k1_IBM_s012
                                 + 1.15156 w_k1_INTC_s012
                                 + 1.09166 w_k1_KO_s012
                                 + 0.905319 w_k1_MAT_s012
                                 + 1.01887 w_k1_MSFT_s012
                                 + 0.842463 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.19254 w_k1_ADM_s012
                                 + 1.20673 w_k1_AXP_s012
                                 + 1.54949 w_k1_DIS_s012
                                 + 1.25784 w_k1_FDX_s012
                                 + 1.06653 w_k1_IBM_s012
                                 + 1.09566 w_k1_INTC_s012
                                 + 1.15727 w_k1_KO_s012 + 1.03312 w_k1_MAT_s012
                                 + 1.1084 w_k1_MSFT_s012 + 1.20626 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.06931 w_k1_ADM_s012
                                 + 1.36828 w_k1_AXP_s012
                                 + 1.25121 w_k1_DIS_s012
                                 + 1.13987 w_k1_FDX_s012
                                 + 1.31619 w_k1_IBM_s012
                                 + 1.23966 w_k1_INTC_s012
                                 + 1.03543 w_k1_KO_s012 + 1.08748 w_k1_MAT_s012
                                 + 1.06861 w_k1_MSFT_s012
                                 + 1.05419 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.07333 w_k1_ADM_s012
                                 + 1.18943 w_k1_AXP_s012
                                 + 0.990653 w_k1_DIS_s012
                                 + 1.1953 w_k1_FDX_s012 + 0.80925 w_k1_IBM_s012
                                 + 1.17127 w_k1_INTC_s012
                                 + 0.936265 w_k1_KO_s012
                                 + 0.827774 w_k1_MAT_s012
                                 + 1.11889 w_k1_MSFT_s012
                                 + 0.965619 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.96777 w_k1_ADM_s012
                                 + 0.743614 w_k1_AXP_s012
                                 + 0.711659 w_k1_DIS_s012
                                 + 0.817273 w_k1_FDX_s012
                                 + 0.750612 w_k1_IBM_s012
                                 + 0.881227 w_k1_INTC_s012
                                 + 0.703999 w_k1_KO_s012
                                 + 0.658317 w_k1_MAT_s012
                                 + 0.793438 w_k1_MSFT_s012
                                 + 0.806703 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.915699 w_k1_ADM_s012
                                 + 0.839548 w_k1_AXP_s012
                                 + 0.979263 w_k1_DIS_s012
                                 + 0.898261 w_k1_FDX_s012
                                 + 0.866111 w_k1_IBM_s012
                                 + 0.946029 w_k1_INTC_s012
                                 + 0.914169 w_k1_KO_s012
                                 + 0.912845 w_k1_MAT_s012
                                 + 0.998154 w_k1_MSFT_s012
                                 + 0.875839 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.932533 w_k1_ADM_s012
                                 + 1.5463 w_k1_AXP_s012 + 1.30404 w_k1_DIS_s012
                                 + 1.12697 w_k1_FDX_s012
                                 + 1.65246 w_k1_IBM_s012
                                 + 1.07824 w_k1_INTC_s012 + 1.4542 w_k1_KO_s012
                                 + 1.84003 w_k1_MAT_s012
                                 + 1.10672 w_k1_MSFT_s012
                                 + 1.15299 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.08251 w_k1_ADM_s012
                                 + 1.02298 w_k1_AXP_s012
                                 + 1.01045 w_k1_DIS_s012
                                 + 0.921975 w_k1_FDX_s012
                                 + 1.20273 w_k1_IBM_s012
                                 + 1.17038 w_k1_INTC_s012
                                 + 1.10464 w_k1_KO_s012 + 0.95975 w_k1_MAT_s012
                                 + 1.06654 w_k1_MSFT_s012
                                 + 0.951924 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.1855 w_k1_ADM_s013
                                 + 1.13249 w_k1_AXP_s013
                                 + 1.06405 w_k1_DIS_s013
                                 + 1.27985 w_k1_FDX_s013
                                 + 0.658966 w_k1_IBM_s013
                                 + 1.02161 w_k1_INTC_s013
                                 + 0.944307 w_k1_KO_s013
                                 + 0.695415 w_k1_MAT_s013
                                 + 1.18651 w_k1_MSFT_s013
                                 + 0.908213 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.9466 w_k1_ADM_s013
                                 + 0.906416 w_k1_AXP_s013
                                 + 0.899395 w_k1_DIS_s013
                                 + 0.842818 w_k1_FDX_s013
                                 + 1.4282 w_k1_IBM_s013
                                 + 0.908731 w_k1_INTC_s013
                                 + 1.22277 w_k1_KO_s013 + 1.30498 w_k1_MAT_s013
                                 + 0.897102 w_k1_MSFT_s013
                                 + 1.04709 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.990005 w_k1_ADM_s013
                                 + 0.877948 w_k1_AXP_s013
                                 + 0.733399 w_k1_DIS_s013
                                 + 0.83362 w_k1_FDX_s013
                                 + 1.00135 w_k1_IBM_s013
                                 + 0.971302 w_k1_INTC_s013
                                 + 0.947603 w_k1_KO_s013
                                 + 0.747801 w_k1_MAT_s013
                                 + 0.836383 w_k1_MSFT_s013
                                 + 0.938063 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.44442 w_k1_ADM_s013
                                 + 0.939479 w_k1_AXP_s013
                                 + 1.33346 w_k1_DIS_s013
                                 + 1.15425 w_k1_FDX_s013
                                 + 0.852103 w_k1_IBM_s013
                                 + 1.03153 w_k1_INTC_s013
                                 + 1.14453 w_k1_KO_s013
                                 + 0.847665 w_k1_MAT_s013
                                 + 1.1622 w_k1_MSFT_s013
                                 + 0.971474 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.3159 w_k1_ADM_s013
                                 + 1.16623 w_k1_AXP_s013
                                 + 0.993087 w_k1_DIS_s013
                                 + 1.0854 w_k1_FDX_s013
                                 + 0.718725 w_k1_IBM_s013
                                 + 1.09363 w_k1_INTC_s013
                                 + 0.907302 w_k1_KO_s013
                                 + 0.853532 w_k1_MAT_s013
                                 + 1.00666 w_k1_MSFT_s013
                                 + 1.09496 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.20926 w_k1_ADM_s013
                                 + 1.54961 w_k1_AXP_s013
                                 + 1.44058 w_k1_DIS_s013
                                 + 1.10172 w_k1_FDX_s013
                                 + 1.54626 w_k1_IBM_s013
                                 + 1.03033 w_k1_INTC_s013
                                 + 1.35614 w_k1_KO_s013 + 1.54775 w_k1_MAT_s013
                                 + 1.11321 w_k1_MSFT_s013
                                 + 1.16195 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.922296 w_k1_ADM_s013
                                 + 0.691683 w_k1_AXP_s013
                                 + 0.777415 w_k1_DIS_s013
                                 + 0.829777 w_k1_FDX_s013
                                 + 0.629407 w_k1_IBM_s013
                                 + 0.852813 w_k1_INTC_s013
                                 + 0.691124 w_k1_KO_s013
                                 + 0.453753 w_k1_MAT_s013
                                 + 0.903851 w_k1_MSFT_s013
                                 + 0.858793 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.27293 w_k1_ADM_s013
                                 + 1.26732 w_k1_AXP_s013
                                 + 1.51813 w_k1_DIS_s013
                                 + 1.33579 w_k1_FDX_s013
                                 + 0.983231 w_k1_IBM_s013
                                 + 1.20997 w_k1_INTC_s013
                                 + 1.22512 w_k1_KO_s013
                                 + 0.931042 w_k1_MAT_s013
                                 + 1.19045 w_k1_MSFT_s013 + 1.1924 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.21395 w_k1_ADM_s014
                                 + 1.14137 w_k1_AXP_s014
                                 + 1.58355 w_k1_DIS_s014
                                 + 1.13855 w_k1_FDX_s014
                                 + 0.87668 w_k1_IBM_s014
                                 + 1.1175 w_k1_INTC_s014 + 1.24078 w_k1_KO_s014
                                 + 0.74672 w_k1_MAT_s014
                                 + 1.17034 w_k1_MSFT_s014
                                 + 1.10744 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.877108 w_k1_ADM_s014
                                 + 0.948929 w_k1_AXP_s014
                                 + 0.955545 w_k1_DIS_s014
                                 + 0.87573 w_k1_FDX_s014
                                 + 1.00866 w_k1_IBM_s014
                                 + 0.776977 w_k1_INTC_s014
                                 + 1.01034 w_k1_KO_s014 + 1.36021 w_k1_MAT_s014
                                 + 0.811845 w_k1_MSFT_s014
                                 + 0.968084 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.06083 w_k1_ADM_s014
                                 + 1.37615 w_k1_AXP_s014
                                 + 1.19871 w_k1_DIS_s014
                                 + 1.14923 w_k1_FDX_s014
                                 + 1.06083 w_k1_IBM_s014
                                 + 1.01592 w_k1_INTC_s014
                                 + 1.00272 w_k1_KO_s014 + 1.06771 w_k1_MAT_s014
                                 + 1.03002 w_k1_MSFT_s014
                                 + 0.972854 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.913759 w_k1_ADM_s014
                                 + 1.59388 w_k1_AXP_s014
                                 + 1.63625 w_k1_DIS_s014
                                 + 1.35321 w_k1_FDX_s014
                                 + 1.43368 w_k1_IBM_s014
                                 + 1.12071 w_k1_INTC_s014 + 1.3416 w_k1_KO_s014
                                 + 1.80636 w_k1_MAT_s014
                                 + 1.11791 w_k1_MSFT_s014
                                 + 0.988893 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.767637 w_k1_ADM_s014
                                 + 0.741873 w_k1_AXP_s014
                                 + 0.676266 w_k1_DIS_s014
                                 + 0.7797 w_k1_FDX_s014
                                 + 0.790302 w_k1_IBM_s014
                                 + 0.883047 w_k1_INTC_s014
                                 + 0.899973 w_k1_KO_s014
                                 + 0.771013 w_k1_MAT_s014
                                 + 0.796125 w_k1_MSFT_s014
                                 + 0.839533 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.834393 w_k1_ADM_s014
                                 + 0.970843 w_k1_AXP_s014
                                 + 1.01851 w_k1_DIS_s014 + 1.0551 w_k1_FDX_s014
                                 + 0.772529 w_k1_IBM_s014
                                 + 0.918578 w_k1_INTC_s014
                                 + 0.989173 w_k1_KO_s014
                                 + 0.660046 w_k1_MAT_s014
                                 + 0.968577 w_k1_MSFT_s014
                                 + 0.808843 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.97536 w_k1_ADM_s014
                                 + 1.1113 w_k1_AXP_s014 + 1.12654 w_k1_DIS_s014
                                 + 0.94303 w_k1_FDX_s014
                                 + 1.52491 w_k1_IBM_s014
                                 + 0.905184 w_k1_INTC_s014
                                 + 1.27417 w_k1_KO_s014 + 2.2033 w_k1_MAT_s014
                                 + 1.03239 w_k1_MSFT_s014
                                 + 1.03487 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.07884 w_k1_ADM_s014
                                 + 1.62527 w_k1_AXP_s014
                                 + 0.990297 w_k1_DIS_s014
                                 + 1.08631 w_k1_FDX_s014 + 1.1283 w_k1_IBM_s014
                                 + 0.794027 w_k1_INTC_s014
                                 + 1.05449 w_k1_KO_s014 + 1.75891 w_k1_MAT_s014
                                 + 0.811203 w_k1_MSFT_s014
                                 + 1.06943 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.864834 w_k1_ADM_s015
                                 + 1.24046 w_k1_AXP_s015
                                 + 0.95885 w_k1_DIS_s015
                                 + 1.07136 w_k1_FDX_s015
                                 + 1.02114 w_k1_IBM_s015
                                 + 1.26121 w_k1_INTC_s015
                                 + 1.09471 w_k1_KO_s015 + 1.39095 w_k1_MAT_s015
                                 + 0.990334 w_k1_MSFT_s015
                                 + 1.02861 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.707241 w_k1_ADM_s015
                                 + 0.688188 w_k1_AXP_s015
                                 + 0.664972 w_k1_DIS_s015
                                 + 0.778429 w_k1_FDX_s015
                                 + 0.673947 w_k1_IBM_s015
                                 + 0.82347 w_k1_INTC_s015
                                 + 0.843193 w_k1_KO_s015
                                 + 0.732683 w_k1_MAT_s015
                                 + 0.871211 w_k1_MSFT_s015
                                 + 0.939884 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.07107 w_k1_ADM_s015
                                 + 0.975721 w_k1_AXP_s015
                                 + 1.80154 w_k1_DIS_s015
                                 + 1.11438 w_k1_FDX_s015 + 1.2347 w_k1_IBM_s015
                                 + 1.25999 w_k1_INTC_s015
                                 + 1.30501 w_k1_KO_s015 + 1.03376 w_k1_MAT_s015
                                 + 1.33213 w_k1_MSFT_s015
                                 + 1.28285 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.856247 w_k1_ADM_s015
                                 + 0.840756 w_k1_AXP_s015
                                 + 1.01046 w_k1_DIS_s015
                                 + 0.832949 w_k1_FDX_s015
                                 + 1.31439 w_k1_IBM_s015
                                 + 0.961547 w_k1_INTC_s015
                                 + 1.22287 w_k1_KO_s015 + 1.37943 w_k1_MAT_s015
                                 + 1.04159 w_k1_MSFT_s015
                                 + 1.02638 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.13115 w_k1_ADM_s015
                                 + 1.16097 w_k1_AXP_s015 + 1.2767 w_k1_DIS_s015
                                 + 1.15723 w_k1_FDX_s015
                                 + 0.796284 w_k1_IBM_s015
                                 + 1.31313 w_k1_INTC_s015 + 1.0248 w_k1_KO_s015
                                 + 0.791758 w_k1_MAT_s015
                                 + 1.25205 w_k1_MSFT_s015 + 1.1109 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.51024 w_k1_ADM_s015
                                 + 1.46587 w_k1_AXP_s015
                                 + 1.69619 w_k1_DIS_s015
                                 + 1.37104 w_k1_FDX_s015
                                 + 0.930221 w_k1_IBM_s015
                                 + 1.61894 w_k1_INTC_s015
                                 + 1.18683 w_k1_KO_s015 + 0.76983 w_k1_MAT_s015
                                 + 1.43386 w_k1_MSFT_s015
                                 + 1.30601 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.00224 w_k1_ADM_s015
                                 + 0.897779 w_k1_AXP_s015
                                 + 0.961207 w_k1_DIS_s015
                                 + 1.02048 w_k1_FDX_s015
                                 + 0.680571 w_k1_IBM_s015
                                 + 1.31448 w_k1_INTC_s015
                                 + 0.873965 w_k1_KO_s015
                                 + 0.646528 w_k1_MAT_s015
                                 + 1.15921 w_k1_MSFT_s015
                                 + 1.04563 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.0074 w_k1_ADM_s015
                                 + 1.49778 w_k1_AXP_s015
                                 + 1.53191 w_k1_DIS_s015
                                 + 1.11949 w_k1_FDX_s015
                                 + 1.64961 w_k1_IBM_s015
                                 + 1.12202 w_k1_INTC_s015
                                 + 1.50945 w_k1_KO_s015 + 2.67416 w_k1_MAT_s015
                                 + 1.263 w_k1_MSFT_s015 + 1.29766 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.10796 w_k1_ADM_s016
                                 + 1.12297 w_k1_AXP_s016
                                 + 1.18826 w_k1_DIS_s016 + 1.0448 w_k1_FDX_s016
                                 + 1.4808 w_k1_IBM_s016
                                 + 0.822939 w_k1_INTC_s016
                                 + 1.25703 w_k1_KO_s016 + 1.65472 w_k1_MAT_s016
                                 + 0.930204 w_k1_MSFT_s016
                                 + 1.18514 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.872562 w_k1_ADM_s016
                                 + 0.763114 w_k1_AXP_s016
                                 + 0.704902 w_k1_DIS_s016
                                 + 0.7509 w_k1_FDX_s016
                                 + 0.701993 w_k1_IBM_s016
                                 + 0.883264 w_k1_INTC_s016
                                 + 0.837252 w_k1_KO_s016
                                 + 0.750974 w_k1_MAT_s016
                                 + 0.825083 w_k1_MSFT_s016
                                 + 0.90727 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.847068 w_k1_ADM_s016
                                 + 1.09339 w_k1_AXP_s016
                                 + 0.89125 w_k1_DIS_s016
                                 + 0.975729 w_k1_FDX_s016
                                 + 0.973329 w_k1_IBM_s016
                                 + 1.08749 w_k1_INTC_s016
                                 + 0.990722 w_k1_KO_s016
                                 + 0.915631 w_k1_MAT_s016
                                 + 0.967393 w_k1_MSFT_s016
                                 + 0.957277 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.937153 w_k1_ADM_s016
                                 + 1.11324 w_k1_AXP_s016
                                 + 1.04092 w_k1_DIS_s016
                                 + 1.08023 w_k1_FDX_s016
                                 + 0.884358 w_k1_IBM_s016
                                 + 0.940091 w_k1_INTC_s016
                                 + 0.967007 w_k1_KO_s016
                                 + 1.16947 w_k1_MAT_s016
                                 + 0.963734 w_k1_MSFT_s016
                                 + 0.979239 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.40884 w_k1_ADM_s016
                                 + 1.35897 w_k1_AXP_s016
                                 + 1.75002 w_k1_DIS_s016
                                 + 1.35548 w_k1_FDX_s016
                                 + 0.959449 w_k1_IBM_s016
                                 + 1.09829 w_k1_INTC_s016
                                 + 1.31182 w_k1_KO_s016 + 1.07957 w_k1_MAT_s016
                                 + 1.41565 w_k1_MSFT_s016 + 1.3659 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.01692 w_k1_ADM_s016
                                 + 1.11697 w_k1_AXP_s016
                                 + 1.12341 w_k1_DIS_s016
                                 + 1.15837 w_k1_FDX_s016
                                 + 0.651583 w_k1_IBM_s016
                                 + 0.955098 w_k1_INTC_s016
                                 + 0.842572 w_k1_KO_s016
                                 + 0.768136 w_k1_MAT_s016
                                 + 1.00443 w_k1_MSFT_s016
                                 + 1.13398 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.681452 w_k1_ADM_s016
                                 + 0.575469 w_k1_AXP_s016
                                 + 0.81934 w_k1_DIS_s016
                                 + 0.814738 w_k1_FDX_s016
                                 + 0.935346 w_k1_IBM_s016
                                 + 1.01273 w_k1_INTC_s016
                                 + 0.908911 w_k1_KO_s016
                                 + 0.684227 w_k1_MAT_s016
                                 + 0.922461 w_k1_MSFT_s016
                                 + 0.923302 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.754828 w_k1_ADM_s016
                                 + 0.953923 w_k1_AXP_s016
                                 + 0.653205 w_k1_DIS_s016
                                 + 0.741883 w_k1_FDX_s016
                                 + 1.02568 w_k1_IBM_s016
                                 + 0.663582 w_k1_INTC_s016
                                 + 0.919607 w_k1_KO_s016
                                 + 1.21514 w_k1_MAT_s016
                                 + 0.682917 w_k1_MSFT_s016
                                 + 0.972003 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.30766 w_k1_ADM_s017
                                 + 1.49817 w_k1_AXP_s017
                                 + 1.09163 w_k1_DIS_s017
                                 + 1.17441 w_k1_FDX_s017
                                 + 1.34225 w_k1_IBM_s017
                                 + 1.06986 w_k1_INTC_s017
                                 + 1.17079 w_k1_KO_s017 + 1.66738 w_k1_MAT_s017
                                 + 1.17936 w_k1_MSFT_s017 + 1.0938 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.722634 w_k1_ADM_s017
                                 + 0.65284 w_k1_AXP_s017
                                 + 0.627819 w_k1_DIS_s017
                                 + 0.734505 w_k1_FDX_s017
                                 + 0.859572 w_k1_IBM_s017
                                 + 0.91078 w_k1_INTC_s017
                                 + 0.799622 w_k1_KO_s017
                                 + 0.91241 w_k1_MAT_s017
                                 + 0.751749 w_k1_MSFT_s017
                                 + 0.80218 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.865465 w_k1_ADM_s017
                                 + 1.2025 w_k1_AXP_s017
                                 + 0.903466 w_k1_DIS_s017
                                 + 0.928861 w_k1_FDX_s017
                                 + 1.18755 w_k1_IBM_s017
                                 + 1.04059 w_k1_INTC_s017
                                 + 1.01143 w_k1_KO_s017 + 1.51714 w_k1_MAT_s017
                                 + 0.983342 w_k1_MSFT_s017
                                 + 1.00194 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.913758 w_k1_ADM_s017
                                 + 0.900955 w_k1_AXP_s017
                                 + 0.748976 w_k1_DIS_s017
                                 + 0.948335 w_k1_FDX_s017
                                 + 0.627364 w_k1_IBM_s017
                                 + 0.93814 w_k1_INTC_s017
                                 + 0.699343 w_k1_KO_s017
                                 + 0.694988 w_k1_MAT_s017
                                 + 0.977845 w_k1_MSFT_s017
                                 + 0.851563 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.11517 w_k1_ADM_s017
                                 + 1.04791 w_k1_AXP_s017
                                 + 1.18943 w_k1_DIS_s017
                                 + 1.08839 w_k1_FDX_s017
                                 + 0.862788 w_k1_IBM_s017
                                 + 1.10135 w_k1_INTC_s017
                                 + 1.06102 w_k1_KO_s017 + 1.04171 w_k1_MAT_s017
                                 + 1.09298 w_k1_MSFT_s017
                                 + 1.07626 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.21752 w_k1_ADM_s017
                                 + 1.32135 w_k1_AXP_s017
                                 + 1.55611 w_k1_DIS_s017
                                 + 1.32913 w_k1_FDX_s017
                                 + 1.22949 w_k1_IBM_s017
                                 + 1.13487 w_k1_INTC_s017
                                 + 1.30001 w_k1_KO_s017 + 1.53565 w_k1_MAT_s017
                                 + 1.26629 w_k1_MSFT_s017
                                 + 1.18795 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.812507 w_k1_ADM_s017
                                 + 0.842058 w_k1_AXP_s017
                                 + 0.942564 w_k1_DIS_s017
                                 + 0.929278 w_k1_FDX_s017
                                 + 0.998615 w_k1_IBM_s017
                                 + 1.11354 w_k1_INTC_s017
                                 + 0.928848 w_k1_KO_s017
                                 + 1.07472 w_k1_MAT_s017
                                 + 0.985571 w_k1_MSFT_s017
                                 + 0.930772 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.945434 w_k1_ADM_s017
                                 + 1.26615 w_k1_AXP_s017
                                 + 1.19399 w_k1_DIS_s017
                                 + 1.01963 w_k1_FDX_s017 + 1.6577 w_k1_IBM_s017
                                 + 1.19275 w_k1_INTC_s017 + 1.3833 w_k1_KO_s017
                                 + 1.52406 w_k1_MAT_s017
                                 + 1.03393 w_k1_MSFT_s017
                                 + 1.05073 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.14519 w_k1_ADM_s100
                                 + 0.897026 w_k1_AXP_s100
                                 + 0.980941 w_k1_DIS_s100
                                 + 0.918282 w_k1_FDX_s100
                                 + 1.5344 w_k1_IBM_s100
                                 + 0.844463 w_k1_INTC_s100
                                 + 1.08778 w_k1_KO_s100
                                 + 0.976978 w_k1_MAT_s100
                                 + 0.810018 w_k1_MSFT_s100
                                 + 1.00589 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.875554 w_k1_ADM_s100
                                 + 1.17768 w_k1_AXP_s100
                                 + 0.709845 w_k1_DIS_s100
                                 + 0.958788 w_k1_FDX_s100
                                 + 1.26749 w_k1_IBM_s100
                                 + 1.00284 w_k1_INTC_s100
                                 + 1.00217 w_k1_KO_s100 + 1.41454 w_k1_MAT_s100
                                 + 0.846198 w_k1_MSFT_s100
                                 + 0.996705 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.40519 w_k1_ADM_s100
                                 + 1.53298 w_k1_AXP_s100
                                 + 1.16479 w_k1_DIS_s100
                                 + 1.10491 w_k1_FDX_s100
                                 + 1.10867 w_k1_IBM_s100
                                 + 1.15097 w_k1_INTC_s100
                                 + 1.13575 w_k1_KO_s100 + 1.10284 w_k1_MAT_s100
                                 + 0.913677 w_k1_MSFT_s100
                                 + 1.12636 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.43716 w_k1_ADM_s100
                                 + 1.37213 w_k1_AXP_s100
                                 + 1.59647 w_k1_DIS_s100
                                 + 1.32153 w_k1_FDX_s100 + 1.1921 w_k1_IBM_s100
                                 + 1.29647 w_k1_INTC_s100
                                 + 1.27911 w_k1_KO_s100 + 1.48059 w_k1_MAT_s100
                                 + 1.17503 w_k1_MSFT_s100
                                 + 1.10645 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.834699 w_k1_ADM_s100
                                 + 0.637384 w_k1_AXP_s100
                                 + 0.523082 w_k1_DIS_s100
                                 + 0.674508 w_k1_FDX_s100
                                 + 0.812117 w_k1_IBM_s100
                                 + 0.915525 w_k1_INTC_s100
                                 + 0.65186 w_k1_KO_s100
                                 + 0.400741 w_k1_MAT_s100
                                 + 0.664972 w_k1_MSFT_s100
                                 + 0.740718 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.03295 w_k1_ADM_s100
                                 + 0.751155 w_k1_AXP_s100
                                 + 0.981521 w_k1_DIS_s100
                                 + 0.888289 w_k1_FDX_s100
                                 + 1.00594 w_k1_IBM_s100
                                 + 0.879408 w_k1_INTC_s100
                                 + 1.09474 w_k1_KO_s100
                                 + 0.771394 w_k1_MAT_s100
                                 + 0.877024 w_k1_MSFT_s100
                                 + 0.958058 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.11546 w_k1_ADM_s100
                                 + 0.943191 w_k1_AXP_s100
                                 + 1.22905 w_k1_DIS_s100 + 1.1787 w_k1_FDX_s100
                                 + 0.761822 w_k1_IBM_s100
                                 + 1.12932 w_k1_INTC_s100
                                 + 0.923258 w_k1_KO_s100
                                 + 0.464501 w_k1_MAT_s100
                                 + 1.09521 w_k1_MSFT_s100
                                 + 1.11444 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.17777 w_k1_ADM_s100
                                 + 0.945666 w_k1_AXP_s100
                                 + 0.849917 w_k1_DIS_s100
                                 + 0.993045 w_k1_FDX_s100
                                 + 0.783881 w_k1_IBM_s100
                                 + 0.911378 w_k1_INTC_s100
                                 + 0.873837 w_k1_KO_s100
                                 + 0.638119 w_k1_MAT_s100
                                 + 0.943007 w_k1_MSFT_s100
                                 + 1.04271 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.10266 w_k1_ADM_s101
                                 + 1.60847 w_k1_AXP_s101
                                 + 1.17975 w_k1_DIS_s101 + 1.2972 w_k1_FDX_s101
                                 + 0.994232 w_k1_IBM_s101
                                 + 0.997065 w_k1_INTC_s101
                                 + 1.08675 w_k1_KO_s101 + 1.22411 w_k1_MAT_s101
                                 + 1.21503 w_k1_MSFT_s101
                                 + 1.21857 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.907814 w_k1_ADM_s101
                                 + 1.19729 w_k1_AXP_s101
                                 + 1.40784 w_k1_DIS_s101 + 1.1453 w_k1_FDX_s101
                                 + 1.36275 w_k1_IBM_s101
                                 + 1.28138 w_k1_INTC_s101
                                 + 1.27832 w_k1_KO_s101 + 1.41179 w_k1_MAT_s101
                                 + 1.28181 w_k1_MSFT_s101
                                 + 1.08295 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.813151 w_k1_ADM_s101
                                 + 0.711179 w_k1_AXP_s101
                                 + 0.92442 w_k1_DIS_s101
                                 + 0.892615 w_k1_FDX_s101
                                 + 1.04854 w_k1_IBM_s101
                                 + 0.959026 w_k1_INTC_s101
                                 + 0.946057 w_k1_KO_s101
                                 + 0.773633 w_k1_MAT_s101
                                 + 0.975809 w_k1_MSFT_s101
                                 + 0.963715 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.747432 w_k1_ADM_s101
                                 + 0.680775 w_k1_AXP_s101
                                 + 0.572155 w_k1_DIS_s101
                                 + 0.765321 w_k1_FDX_s101
                                 + 0.984402 w_k1_IBM_s101
                                 + 0.682279 w_k1_INTC_s101
                                 + 0.773413 w_k1_KO_s101
                                 + 0.767892 w_k1_MAT_s101
                                 + 0.720919 w_k1_MSFT_s101
                                 + 0.898281 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.889213 w_k1_ADM_s101
                                 + 1.06121 w_k1_AXP_s101
                                 + 0.966721 w_k1_DIS_s101
                                 + 0.981123 w_k1_FDX_s101 + 1.267 w_k1_IBM_s101
                                 + 0.930506 w_k1_INTC_s101
                                 + 1.17528 w_k1_KO_s101 + 1.19045 w_k1_MAT_s101
                                 + 0.993924 w_k1_MSFT_s101
                                 + 0.955329 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.14966 w_k1_ADM_s101
                                 + 1.70961 w_k1_AXP_s101
                                 + 1.80399 w_k1_DIS_s101
                                 + 1.44696 w_k1_FDX_s101
                                 + 1.55248 w_k1_IBM_s101
                                 + 1.28753 w_k1_INTC_s101
                                 + 1.38797 w_k1_KO_s101 + 1.13286 w_k1_MAT_s101
                                 + 1.23858 w_k1_MSFT_s101
                                 + 1.24225 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04998 w_k1_ADM_s101
                                 + 0.991027 w_k1_AXP_s101
                                 + 1.28093 w_k1_DIS_s101
                                 + 1.28746 w_k1_FDX_s101
                                 + 0.927437 w_k1_IBM_s101
                                 + 1.24291 w_k1_INTC_s101 + 1.0628 w_k1_KO_s101
                                 + 0.654058 w_k1_MAT_s101
                                 + 1.17221 w_k1_MSFT_s101 + 1.0228 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.795678 w_k1_ADM_s101
                                 + 0.8942 w_k1_AXP_s101
                                 + 0.664858 w_k1_DIS_s101
                                 + 0.912731 w_k1_FDX_s101
                                 + 0.732237 w_k1_IBM_s101
                                 + 0.89468 w_k1_INTC_s101
                                 + 0.725651 w_k1_KO_s101
                                 + 0.674706 w_k1_MAT_s101
                                 + 0.824322 w_k1_MSFT_s101
                                 + 0.914708 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.93509 w_k1_ADM_s102
                                 + 1.15023 w_k1_AXP_s102
                                 + 1.11436 w_k1_DIS_s102 + 1.0812 w_k1_FDX_s102
                                 + 0.816932 w_k1_IBM_s102
                                 + 0.976509 w_k1_INTC_s102
                                 + 1.06226 w_k1_KO_s102 + 1.17369 w_k1_MAT_s102
                                 + 1.14179 w_k1_MSFT_s102
                                 + 1.08277 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.18804 w_k1_ADM_s102
                                 + 0.981888 w_k1_AXP_s102
                                 + 1.01075 w_k1_DIS_s102
                                 + 0.980172 w_k1_FDX_s102
                                 + 0.901625 w_k1_IBM_s102
                                 + 1.12276 w_k1_INTC_s102
                                 + 1.05328 w_k1_KO_s102 + 0.68384 w_k1_MAT_s102
                                 + 0.920148 w_k1_MSFT_s102
                                 + 0.994161 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.934185 w_k1_ADM_s102
                                 + 1.01633 w_k1_AXP_s102
                                 + 0.778561 w_k1_DIS_s102
                                 + 0.826705 w_k1_FDX_s102
                                 + 1.06975 w_k1_IBM_s102
                                 + 0.722131 w_k1_INTC_s102
                                 + 0.920425 w_k1_KO_s102
                                 + 1.24011 w_k1_MAT_s102
                                 + 0.769379 w_k1_MSFT_s102
                                 + 0.992656 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.2959 w_k1_ADM_s102
                                 + 1.22545 w_k1_AXP_s102
                                 + 1.39021 w_k1_DIS_s102
                                 + 1.05406 w_k1_FDX_s102
                                 + 1.41467 w_k1_IBM_s102
                                 + 1.08102 w_k1_INTC_s102
                                 + 1.24822 w_k1_KO_s102 + 1.27076 w_k1_MAT_s102
                                 + 1.10403 w_k1_MSFT_s102
                                 + 1.13738 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.12897 w_k1_ADM_s102
                                 + 1.43991 w_k1_AXP_s102
                                 + 1.62384 w_k1_DIS_s102 + 1.2317 w_k1_FDX_s102
                                 + 1.06128 w_k1_IBM_s102
                                 + 1.24633 w_k1_INTC_s102
                                 + 1.24891 w_k1_KO_s102 + 1.15742 w_k1_MAT_s102
                                 + 1.10046 w_k1_MSFT_s102
                                 + 1.27618 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.06134 w_k1_ADM_s102
                                 + 0.78355 w_k1_AXP_s102
                                 + 0.878552 w_k1_DIS_s102
                                 + 0.972918 w_k1_FDX_s102
                                 + 0.639673 w_k1_IBM_s102
                                 + 0.843154 w_k1_INTC_s102
                                 + 0.749009 w_k1_KO_s102
                                 + 0.561993 w_k1_MAT_s102
                                 + 0.855323 w_k1_MSFT_s102
                                 + 0.827189 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.842429 w_k1_ADM_s102
                                 + 0.798777 w_k1_AXP_s102
                                 + 0.733316 w_k1_DIS_s102
                                 + 0.780382 w_k1_FDX_s102
                                 + 0.85663 w_k1_IBM_s102
                                 + 0.820371 w_k1_INTC_s102
                                 + 0.803067 w_k1_KO_s102
                                 + 0.899439 w_k1_MAT_s102
                                 + 0.871773 w_k1_MSFT_s102
                                 + 0.988904 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.775885 w_k1_ADM_s102
                                 + 0.596976 w_k1_AXP_s102
                                 + 0.571757 w_k1_DIS_s102
                                 + 0.690543 w_k1_FDX_s102
                                 + 0.711978 w_k1_IBM_s102
                                 + 0.878441 w_k1_INTC_s102
                                 + 0.738201 w_k1_KO_s102
                                 + 0.767184 w_k1_MAT_s102
                                 + 0.762654 w_k1_MSFT_s102
                                 + 0.874678 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.999658 w_k1_ADM_s103
                                 + 0.947768 w_k1_AXP_s103
                                 + 1.12841 w_k1_DIS_s103
                                 + 1.07414 w_k1_FDX_s103
                                 + 0.765065 w_k1_IBM_s103
                                 + 1.06853 w_k1_INTC_s103
                                 + 1.03435 w_k1_KO_s103
                                 + 0.799296 w_k1_MAT_s103
                                 + 0.992821 w_k1_MSFT_s103
                                 + 0.999889 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.2457 w_k1_ADM_s103
                                 + 1.30554 w_k1_AXP_s103
                                 + 1.35463 w_k1_DIS_s103
                                 + 1.36933 w_k1_FDX_s103
                                 + 1.11731 w_k1_IBM_s103
                                 + 1.06157 w_k1_INTC_s103
                                 + 1.23608 w_k1_KO_s103 + 1.16256 w_k1_MAT_s103
                                 + 1.10194 w_k1_MSFT_s103
                                 + 1.07688 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.924275 w_k1_ADM_s103
                                 + 1.05124 w_k1_AXP_s103
                                 + 0.975305 w_k1_DIS_s103
                                 + 0.995463 w_k1_FDX_s103
                                 + 1.22758 w_k1_IBM_s103
                                 + 0.971725 w_k1_INTC_s103
                                 + 1.05164 w_k1_KO_s103 + 1.34512 w_k1_MAT_s103
                                 + 0.968468 w_k1_MSFT_s103
                                 + 0.942567 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.902697 w_k1_ADM_s103
                                 + 0.859094 w_k1_AXP_s103
                                 + 0.861078 w_k1_DIS_s103
                                 + 0.849644 w_k1_FDX_s103
                                 + 1.01598 w_k1_IBM_s103
                                 + 0.874368 w_k1_INTC_s103
                                 + 0.906881 w_k1_KO_s103
                                 + 0.994627 w_k1_MAT_s103
                                 + 0.793882 w_k1_MSFT_s103
                                 + 0.862457 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.942531 w_k1_ADM_s103
                                 + 1.25903 w_k1_AXP_s103
                                 + 1.43162 w_k1_DIS_s103
                                 + 1.18379 w_k1_FDX_s103
                                 + 1.70902 w_k1_IBM_s103
                                 + 1.05315 w_k1_INTC_s103
                                 + 1.26214 w_k1_KO_s103 + 1.79404 w_k1_MAT_s103
                                 + 1.08985 w_k1_MSFT_s103
                                 + 1.13048 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.732199 w_k1_ADM_s103
                                 + 0.622461 w_k1_AXP_s103
                                 + 0.722268 w_k1_DIS_s103
                                 + 0.777083 w_k1_FDX_s103
                                 + 0.767468 w_k1_IBM_s103
                                 + 0.777351 w_k1_INTC_s103
                                 + 0.840707 w_k1_KO_s103
                                 + 0.621404 w_k1_MAT_s103
                                 + 0.7361 w_k1_MSFT_s103
                                 + 0.820645 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.11151 w_k1_ADM_s103
                                 + 1.43459 w_k1_AXP_s103
                                 + 1.31797 w_k1_DIS_s103
                                 + 1.19195 w_k1_FDX_s103
                                 + 0.969208 w_k1_IBM_s103
                                 + 1.04356 w_k1_INTC_s103
                                 + 1.03236 w_k1_KO_s103 + 1.55381 w_k1_MAT_s103
                                 + 1.00928 w_k1_MSFT_s103
                                 + 1.08192 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.05235 w_k1_ADM_s103
                                 + 1.09743 w_k1_AXP_s103
                                 + 1.60159 w_k1_DIS_s103
                                 + 1.12213 w_k1_FDX_s103
                                 + 1.22314 w_k1_IBM_s103
                                 + 1.17352 w_k1_INTC_s103
                                 + 1.46097 w_k1_KO_s103 + 1.14428 w_k1_MAT_s103
                                 + 1.31591 w_k1_MSFT_s103
                                 + 1.18633 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.17561 w_k1_ADM_s104
                                 + 0.951822 w_k1_AXP_s104
                                 + 1.06831 w_k1_DIS_s104
                                 + 1.04374 w_k1_FDX_s104
                                 + 0.941439 w_k1_IBM_s104
                                 + 1.09772 w_k1_INTC_s104
                                 + 1.09789 w_k1_KO_s104 + 1.06023 w_k1_MAT_s104
                                 + 1.02074 w_k1_MSFT_s104
                                 + 0.998671 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.1903 w_k1_ADM_s104
                                 + 0.824354 w_k1_AXP_s104
                                 + 0.758709 w_k1_DIS_s104
                                 + 0.975331 w_k1_FDX_s104
                                 + 1.39561 w_k1_IBM_s104
                                 + 0.962193 w_k1_INTC_s104
                                 + 1.16389 w_k1_KO_s104 + 1.34106 w_k1_MAT_s104
                                 + 1.01816 w_k1_MSFT_s104
                                 + 1.08155 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.829989 w_k1_ADM_s104
                                 + 0.819743 w_k1_AXP_s104
                                 + 0.727797 w_k1_DIS_s104
                                 + 0.75759 w_k1_FDX_s104 + 1.1 w_k1_IBM_s104
                                 + 0.804797 w_k1_INTC_s104
                                 + 0.818847 w_k1_KO_s104
                                 + 1.22362 w_k1_MAT_s104
                                 + 0.702004 w_k1_MSFT_s104
                                 + 0.87942 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.825793 w_k1_ADM_s104
                                 + 0.719626 w_k1_AXP_s104
                                 + 0.725815 w_k1_DIS_s104
                                 + 0.914956 w_k1_FDX_s104
                                 + 0.639386 w_k1_IBM_s104
                                 + 0.771599 w_k1_INTC_s104
                                 + 0.679202 w_k1_KO_s104
                                 + 0.699331 w_k1_MAT_s104
                                 + 0.843508 w_k1_MSFT_s104
                                 + 0.883189 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.992977 w_k1_ADM_s104
                                 + 1.06231 w_k1_AXP_s104
                                 + 1.43961 w_k1_DIS_s104
                                 + 1.12334 w_k1_FDX_s104
                                 + 1.42873 w_k1_IBM_s104
                                 + 1.00398 w_k1_INTC_s104
                                 + 1.17354 w_k1_KO_s104 + 1.18196 w_k1_MAT_s104
                                 + 1.05406 w_k1_MSFT_s104
                                 + 1.06538 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.25469 w_k1_ADM_s104
                                 + 0.94336 w_k1_AXP_s104
                                 + 1.46108 w_k1_DIS_s104
                                 + 1.29264 w_k1_FDX_s104
                                 + 0.90098 w_k1_IBM_s104
                                 + 0.963005 w_k1_INTC_s104
                                 + 1.22949 w_k1_KO_s104
                                 + 0.928682 w_k1_MAT_s104
                                 + 1.24234 w_k1_MSFT_s104
                                 + 1.10262 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.24176 w_k1_ADM_s104
                                 + 1.638 w_k1_AXP_s104 + 1.17791 w_k1_DIS_s104
                                 + 1.21782 w_k1_FDX_s104
                                 + 1.35153 w_k1_IBM_s104
                                 + 1.09863 w_k1_INTC_s104
                                 + 1.08391 w_k1_KO_s104 + 1.6603 w_k1_MAT_s104
                                 + 1.14377 w_k1_MSFT_s104 + 1.1208 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.79896 w_k1_ADM_s104
                                 + 0.563476 w_k1_AXP_s104
                                 + 0.579754 w_k1_DIS_s104
                                 + 0.640199 w_k1_FDX_s104
                                 + 0.811827 w_k1_IBM_s104
                                 + 0.574734 w_k1_INTC_s104
                                 + 0.790016 w_k1_KO_s104
                                 + 0.779538 w_k1_MAT_s104
                                 + 0.735073 w_k1_MSFT_s104
                                 + 0.839081 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.823486 w_k1_ADM_s105
                                 + 0.768985 w_k1_AXP_s105
                                 + 0.68165 w_k1_DIS_s105
                                 + 0.775506 w_k1_FDX_s105
                                 + 1.1189 w_k1_IBM_s105 + 1.0681 w_k1_INTC_s105
                                 + 0.913495 w_k1_KO_s105
                                 + 0.895321 w_k1_MAT_s105
                                 + 0.902595 w_k1_MSFT_s105
                                 + 0.891169 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.3953 w_k1_ADM_s105
                                 + 1.34016 w_k1_AXP_s105
                                 + 1.70113 w_k1_DIS_s105
                                 + 1.27688 w_k1_FDX_s105
                                 + 1.38345 w_k1_IBM_s105
                                 + 1.2055 w_k1_INTC_s105 + 1.41363 w_k1_KO_s105
                                 + 1.06916 w_k1_MAT_s105
                                 + 1.36754 w_k1_MSFT_s105
                                 + 1.28577 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.681665 w_k1_ADM_s105
                                 + 0.554201 w_k1_AXP_s105
                                 + 0.628125 w_k1_DIS_s105
                                 + 0.717638 w_k1_FDX_s105
                                 + 0.753749 w_k1_IBM_s105
                                 + 0.860899 w_k1_INTC_s105
                                 + 0.772241 w_k1_KO_s105
                                 + 0.647188 w_k1_MAT_s105
                                 + 0.811461 w_k1_MSFT_s105
                                 + 0.805858 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.955574 w_k1_ADM_s105
                                 + 0.985858 w_k1_AXP_s105
                                 + 1.22314 w_k1_DIS_s105
                                 + 1.01613 w_k1_FDX_s105
                                 + 1.08438 w_k1_IBM_s105
                                 + 1.07952 w_k1_INTC_s105
                                 + 1.05236 w_k1_KO_s105 + 0.95554 w_k1_MAT_s105
                                 + 1.25279 w_k1_MSFT_s105
                                 + 0.961986 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.894657 w_k1_ADM_s105
                                 + 0.802372 w_k1_AXP_s105
                                 + 1.13472 w_k1_DIS_s105
                                 + 0.850054 w_k1_FDX_s105
                                 + 1.42439 w_k1_IBM_s105
                                 + 1.09743 w_k1_INTC_s105
                                 + 1.16473 w_k1_KO_s105 + 1.27586 w_k1_MAT_s105
                                 + 1.14362 w_k1_MSFT_s105
                                 + 1.00872 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.12534 w_k1_ADM_s105
                                 + 1.48429 w_k1_AXP_s105
                                 + 1.31874 w_k1_DIS_s105
                                 + 1.31943 w_k1_FDX_s105
                                 + 1.02922 w_k1_IBM_s105
                                 + 1.20758 w_k1_INTC_s105
                                 + 1.03704 w_k1_KO_s105 + 1.21505 w_k1_MAT_s105
                                 + 1.22117 w_k1_MSFT_s105
                                 + 1.11213 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.806983 w_k1_ADM_s105
                                 + 0.986197 w_k1_AXP_s105
                                 + 0.857433 w_k1_DIS_s105
                                 + 0.959893 w_k1_FDX_s105
                                 + 0.875688 w_k1_IBM_s105
                                 + 1.18025 w_k1_INTC_s105
                                 + 0.880512 w_k1_KO_s105
                                 + 0.929477 w_k1_MAT_s105
                                 + 0.973092 w_k1_MSFT_s105
                                 + 0.917933 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.10184 w_k1_ADM_s105
                                 + 0.96085 w_k1_AXP_s105
                                 + 1.03514 w_k1_DIS_s105
                                 + 1.08735 w_k1_FDX_s105
                                 + 0.868363 w_k1_IBM_s105
                                 + 1.32741 w_k1_INTC_s105
                                 + 1.08535 w_k1_KO_s105 + 0.52998 w_k1_MAT_s105
                                 + 1.22025 w_k1_MSFT_s105
                                 + 1.05386 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.755215 w_k1_ADM_s106
                                 + 0.926548 w_k1_AXP_s106
                                 + 0.782851 w_k1_DIS_s106
                                 + 0.758441 w_k1_FDX_s106
                                 + 1.16613 w_k1_IBM_s106
                                 + 0.879219 w_k1_INTC_s106
                                 + 0.958985 w_k1_KO_s106
                                 + 1.16269 w_k1_MAT_s106
                                 + 0.846535 w_k1_MSFT_s106
                                 + 1.01117 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.16858 w_k1_ADM_s106
                                 + 1.29077 w_k1_AXP_s106
                                 + 0.957661 w_k1_DIS_s106
                                 + 1.13659 w_k1_FDX_s106
                                 + 0.685008 w_k1_IBM_s106
                                 + 0.771823 w_k1_INTC_s106
                                 + 0.74002 w_k1_KO_s106
                                 + 0.790023 w_k1_MAT_s106
                                 + 0.853977 w_k1_MSFT_s106
                                 + 1.11486 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.912482 w_k1_ADM_s106
                                 + 0.883909 w_k1_AXP_s106
                                 + 0.903977 w_k1_DIS_s106
                                 + 0.929662 w_k1_FDX_s106
                                 + 0.943029 w_k1_IBM_s106
                                 + 0.703423 w_k1_INTC_s106
                                 + 0.949077 w_k1_KO_s106
                                 + 0.946045 w_k1_MAT_s106
                                 + 0.904899 w_k1_MSFT_s106
                                 + 1.08139 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.849067 w_k1_ADM_s106
                                 + 0.949033 w_k1_AXP_s106
                                 + 0.725159 w_k1_DIS_s106
                                 + 0.82414 w_k1_FDX_s106
                                 + 0.777523 w_k1_IBM_s106
                                 + 0.744447 w_k1_INTC_s106
                                 + 0.794182 w_k1_KO_s106
                                 + 0.963994 w_k1_MAT_s106
                                 + 0.76407 w_k1_MSFT_s106
                                 + 0.935707 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.16635 w_k1_ADM_s106
                                 + 1.11091 w_k1_AXP_s106 + 1.4159 w_k1_DIS_s106
                                 + 1.22212 w_k1_FDX_s106
                                 + 1.03252 w_k1_IBM_s106 + 1.002 w_k1_INTC_s106
                                 + 1.26672 w_k1_KO_s106 + 1.08416 w_k1_MAT_s106
                                 + 1.10571 w_k1_MSFT_s106
                                 + 1.19846 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.15516 w_k1_ADM_s106
                                 + 1.47671 w_k1_AXP_s106
                                 + 1.46609 w_k1_DIS_s106
                                 + 1.25571 w_k1_FDX_s106
                                 + 1.57977 w_k1_IBM_s106
                                 + 1.1183 w_k1_INTC_s106 + 1.41889 w_k1_KO_s106
                                 + 1.97107 w_k1_MAT_s106
                                 + 1.09783 w_k1_MSFT_s106
                                 + 1.48432 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.976777 w_k1_ADM_s106
                                 + 1.46035 w_k1_AXP_s106
                                 + 0.988342 w_k1_DIS_s106
                                 + 1.01455 w_k1_FDX_s106
                                 + 1.14485 w_k1_IBM_s106
                                 + 0.922652 w_k1_INTC_s106
                                 + 0.9577 w_k1_KO_s106 + 1.22937 w_k1_MAT_s106
                                 + 0.892843 w_k1_MSFT_s106
                                 + 1.11507 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.33994 w_k1_ADM_s106
                                 + 1.56509 w_k1_AXP_s106
                                 + 2.17673 w_k1_DIS_s106 + 1.5688 w_k1_FDX_s106
                                 + 1.00425 w_k1_IBM_s106
                                 + 1.35191 w_k1_INTC_s106
                                 + 1.43537 w_k1_KO_s106
                                 + 0.926126 w_k1_MAT_s106
                                 + 1.50313 w_k1_MSFT_s106
                                 + 1.48057 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.857613 w_k1_ADM_s107
                                 + 0.634428 w_k1_AXP_s107
                                 + 0.641819 w_k1_DIS_s107
                                 + 0.739434 w_k1_FDX_s107
                                 + 1.0827 w_k1_IBM_s107
                                 + 0.774532 w_k1_INTC_s107
                                 + 0.804078 w_k1_KO_s107
                                 + 0.891222 w_k1_MAT_s107
                                 + 0.633988 w_k1_MSFT_s107
                                 + 0.820135 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.807764 w_k1_ADM_s107
                                 + 0.611478 w_k1_AXP_s107
                                 + 0.525811 w_k1_DIS_s107
                                 + 0.648314 w_k1_FDX_s107
                                 + 0.745154 w_k1_IBM_s107
                                 + 1.06101 w_k1_INTC_s107
                                 + 0.772429 w_k1_KO_s107
                                 + 0.587791 w_k1_MAT_s107
                                 + 0.614837 w_k1_MSFT_s107
                                 + 0.749811 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.03225 w_k1_ADM_s107
                                 + 0.768813 w_k1_AXP_s107
                                 + 0.988868 w_k1_DIS_s107
                                 + 0.909734 w_k1_FDX_s107
                                 + 1.03119 w_k1_IBM_s107
                                 + 0.779461 w_k1_INTC_s107
                                 + 0.970939 w_k1_KO_s107
                                 + 0.985531 w_k1_MAT_s107
                                 + 0.818873 w_k1_MSFT_s107
                                 + 0.977689 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.29789 w_k1_ADM_s107
                                 + 1.29624 w_k1_AXP_s107
                                 + 1.46967 w_k1_DIS_s107
                                 + 1.32718 w_k1_FDX_s107
                                 + 1.04025 w_k1_IBM_s107
                                 + 1.11365 w_k1_INTC_s107
                                 + 1.20203 w_k1_KO_s107 + 1.2873 w_k1_MAT_s107
                                 + 1.20374 w_k1_MSFT_s107
                                 + 1.11259 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.2878 w_k1_ADM_s107
                                 + 1.44242 w_k1_AXP_s107
                                 + 1.06829 w_k1_DIS_s107
                                 + 1.04194 w_k1_FDX_s107
                                 + 1.36739 w_k1_IBM_s107
                                 + 1.21498 w_k1_INTC_s107
                                 + 1.23525 w_k1_KO_s107 + 1.41808 w_k1_MAT_s107
                                 + 0.993265 w_k1_MSFT_s107
                                 + 1.09722 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.945228 w_k1_ADM_s107
                                 + 0.842464 w_k1_AXP_s107
                                 + 0.830436 w_k1_DIS_s107
                                 + 0.974823 w_k1_FDX_s107
                                 + 0.721499 w_k1_IBM_s107
                                 + 0.875617 w_k1_INTC_s107
                                 + 0.885916 w_k1_KO_s107
                                 + 0.752498 w_k1_MAT_s107
                                 + 0.845909 w_k1_MSFT_s107
                                 + 0.81669 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.01404 w_k1_ADM_s107
                                 + 1.08284 w_k1_AXP_s107
                                 + 0.698987 w_k1_DIS_s107
                                 + 0.962454 w_k1_FDX_s107
                                 + 0.92123 w_k1_IBM_s107
                                 + 0.863443 w_k1_INTC_s107
                                 + 0.962836 w_k1_KO_s107
                                 + 1.36563 w_k1_MAT_s107
                                 + 0.800088 w_k1_MSFT_s107
                                 + 1.05697 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 2.0007 w_k1_ADM_s107
                                 + 1.86923 w_k1_AXP_s107
                                 + 1.94619 w_k1_DIS_s107
                                 + 1.80514 w_k1_FDX_s107
                                 + 1.38278 w_k1_IBM_s107
                                 + 1.58505 w_k1_INTC_s107
                                 + 1.58862 w_k1_KO_s107 + 1.54655 w_k1_MAT_s107
                                 + 1.52061 w_k1_MSFT_s107
                                 + 1.34931 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.24135 w_k1_ADM_s110
                                 + 1.4101 w_k1_AXP_s110 + 1.09306 w_k1_DIS_s110
                                 + 1.11514 w_k1_FDX_s110
                                 + 1.18708 w_k1_IBM_s110
                                 + 1.03105 w_k1_INTC_s110
                                 + 1.23621 w_k1_KO_s110 + 1.40782 w_k1_MAT_s110
                                 + 1.18904 w_k1_MSFT_s110
                                 + 1.02263 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.36488 w_k1_ADM_s110
                                 + 1.54892 w_k1_AXP_s110
                                 + 1.30462 w_k1_DIS_s110
                                 + 1.15817 w_k1_FDX_s110
                                 + 1.90657 w_k1_IBM_s110
                                 + 1.30585 w_k1_INTC_s110
                                 + 1.45759 w_k1_KO_s110 + 2.22416 w_k1_MAT_s110
                                 + 1.3059 w_k1_MSFT_s110 + 1.14251 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.725254 w_k1_ADM_s110
                                 + 0.79539 w_k1_AXP_s110
                                 + 0.688144 w_k1_DIS_s110
                                 + 0.786305 w_k1_FDX_s110
                                 + 1.1509 w_k1_IBM_s110 + 0.8995 w_k1_INTC_s110
                                 + 1.01043 w_k1_KO_s110
                                 + 0.917339 w_k1_MAT_s110
                                 + 1.00789 w_k1_MSFT_s110
                                 + 0.850735 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.19033 w_k1_ADM_s110
                                 + 1.54898 w_k1_AXP_s110
                                 + 1.22376 w_k1_DIS_s110
                                 + 1.18843 w_k1_FDX_s110
                                 + 1.41688 w_k1_IBM_s110
                                 + 1.15978 w_k1_INTC_s110
                                 + 1.32378 w_k1_KO_s110 + 1.58495 w_k1_MAT_s110
                                 + 1.28089 w_k1_MSFT_s110
                                 + 1.08937 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.611231 w_k1_ADM_s110
                                 + 0.74554 w_k1_AXP_s110
                                 + 0.502796 w_k1_DIS_s110
                                 + 0.701168 w_k1_FDX_s110
                                 + 0.587516 w_k1_IBM_s110
                                 + 0.642932 w_k1_INTC_s110
                                 + 0.581524 w_k1_KO_s110
                                 + 0.758677 w_k1_MAT_s110
                                 + 0.687528 w_k1_MSFT_s110
                                 + 0.655432 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.865045 w_k1_ADM_s110
                                 + 1.04273 w_k1_AXP_s110
                                 + 0.852507 w_k1_DIS_s110
                                 + 0.923429 w_k1_FDX_s110
                                 + 0.940608 w_k1_IBM_s110
                                 + 0.983645 w_k1_INTC_s110
                                 + 1.07715 w_k1_KO_s110 + 1.07525 w_k1_MAT_s110
                                 + 0.913206 w_k1_MSFT_s110
                                 + 0.896806 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.748612 w_k1_ADM_s110
                                 + 0.852604 w_k1_AXP_s110
                                 + 0.726803 w_k1_DIS_s110
                                 + 0.873172 w_k1_FDX_s110
                                 + 0.795247 w_k1_IBM_s110
                                 + 0.81176 w_k1_INTC_s110
                                 + 0.676699 w_k1_KO_s110
                                 + 0.615303 w_k1_MAT_s110
                                 + 0.861571 w_k1_MSFT_s110
                                 + 0.722846 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.12626 w_k1_ADM_s110
                                 + 1.30985 w_k1_AXP_s110
                                 + 0.950459 w_k1_DIS_s110
                                 + 1.13025 w_k1_FDX_s110
                                 + 0.935486 w_k1_IBM_s110
                                 + 1.0364 w_k1_INTC_s110
                                 + 0.867503 w_k1_KO_s110
                                 + 1.04499 w_k1_MAT_s110
                                 + 1.15629 w_k1_MSFT_s110
                                 + 0.90588 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.903332 w_k1_ADM_s111
                                 + 1.13631 w_k1_AXP_s111
                                 + 1.14138 w_k1_DIS_s111
                                 + 1.08575 w_k1_FDX_s111
                                 + 1.14611 w_k1_IBM_s111
                                 + 0.885792 w_k1_INTC_s111
                                 + 1.16649 w_k1_KO_s111
                                 + 0.891022 w_k1_MAT_s111
                                 + 0.995345 w_k1_MSFT_s111
                                 + 1.15136 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.840443 w_k1_ADM_s111
                                 + 0.903377 w_k1_AXP_s111
                                 + 0.918186 w_k1_DIS_s111
                                 + 0.981271 w_k1_FDX_s111
                                 + 0.823405 w_k1_IBM_s111
                                 + 0.967644 w_k1_INTC_s111
                                 + 0.951693 w_k1_KO_s111
                                 + 0.692641 w_k1_MAT_s111
                                 + 0.888665 w_k1_MSFT_s111
                                 + 0.981217 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13496 w_k1_ADM_s111
                                 + 1.58591 w_k1_AXP_s111
                                 + 1.58482 w_k1_DIS_s111
                                 + 1.28889 w_k1_FDX_s111
                                 + 1.38012 w_k1_IBM_s111
                                 + 1.26478 w_k1_INTC_s111
                                 + 1.42691 w_k1_KO_s111 + 1.36369 w_k1_MAT_s111
                                 + 1.17832 w_k1_MSFT_s111
                                 + 1.07806 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.903301 w_k1_ADM_s111
                                 + 0.955442 w_k1_AXP_s111
                                 + 1.00318 w_k1_DIS_s111
                                 + 0.906716 w_k1_FDX_s111
                                 + 1.35949 w_k1_IBM_s111
                                 + 0.955849 w_k1_INTC_s111
                                 + 1.14666 w_k1_KO_s111 + 1.12258 w_k1_MAT_s111
                                 + 0.959002 w_k1_MSFT_s111
                                 + 1.10273 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.09072 w_k1_ADM_s111
                                 + 1.505 w_k1_AXP_s111 + 2.06497 w_k1_DIS_s111
                                 + 1.6488 w_k1_FDX_s111 + 1.1197 w_k1_IBM_s111
                                 + 1.38055 w_k1_INTC_s111
                                 + 1.46537 w_k1_KO_s111 + 1.25368 w_k1_MAT_s111
                                 + 1.42718 w_k1_MSFT_s111
                                 + 1.30611 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.903125 w_k1_ADM_s111
                                 + 0.944461 w_k1_AXP_s111
                                 + 0.652381 w_k1_DIS_s111
                                 + 0.829964 w_k1_FDX_s111
                                 + 1.14553 w_k1_IBM_s111
                                 + 0.905842 w_k1_INTC_s111
                                 + 0.975879 w_k1_KO_s111
                                 + 1.00793 w_k1_MAT_s111
                                 + 0.859219 w_k1_MSFT_s111
                                 + 0.996739 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.31794 w_k1_ADM_s111
                                 + 1.32503 w_k1_AXP_s111
                                 + 1.25047 w_k1_DIS_s111
                                 + 1.27359 w_k1_FDX_s111 + 1.007 w_k1_IBM_s111
                                 + 1.0413 w_k1_INTC_s111 + 1.1823 w_k1_KO_s111
                                 + 0.842548 w_k1_MAT_s111
                                 + 1.14851 w_k1_MSFT_s111
                                 + 1.12411 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.705505 w_k1_ADM_s111
                                 + 0.77546 w_k1_AXP_s111
                                 + 0.594489 w_k1_DIS_s111
                                 + 0.727388 w_k1_FDX_s111
                                 + 0.857887 w_k1_IBM_s111
                                 + 0.821877 w_k1_INTC_s111
                                 + 0.784252 w_k1_KO_s111
                                 + 0.562287 w_k1_MAT_s111
                                 + 0.743709 w_k1_MSFT_s111
                                 + 0.841605 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.10675 w_k1_ADM_s112
                                 + 0.979712 w_k1_AXP_s112
                                 + 1.03164 w_k1_DIS_s112
                                 + 1.05762 w_k1_FDX_s112
                                 + 0.913874 w_k1_IBM_s112
                                 + 1.0875 w_k1_INTC_s112
                                 + 0.990918 w_k1_KO_s112
                                 + 0.755861 w_k1_MAT_s112
                                 + 1.01747 w_k1_MSFT_s112
                                 + 0.984699 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.838836 w_k1_ADM_s112
                                 + 0.745808 w_k1_AXP_s112
                                 + 0.708619 w_k1_DIS_s112
                                 + 0.909235 w_k1_FDX_s112
                                 + 0.807007 w_k1_IBM_s112
                                 + 0.740962 w_k1_INTC_s112
                                 + 0.715966 w_k1_KO_s112
                                 + 0.861243 w_k1_MAT_s112
                                 + 0.797949 w_k1_MSFT_s112
                                 + 0.853438 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.782135 w_k1_ADM_s112
                                 + 0.960202 w_k1_AXP_s112
                                 + 0.835848 w_k1_DIS_s112
                                 + 0.958831 w_k1_FDX_s112
                                 + 0.788396 w_k1_IBM_s112
                                 + 0.942518 w_k1_INTC_s112
                                 + 0.846594 w_k1_KO_s112
                                 + 0.803653 w_k1_MAT_s112
                                 + 0.918355 w_k1_MSFT_s112
                                 + 0.863616 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.898504 w_k1_ADM_s112
                                 + 0.86249 w_k1_AXP_s112
                                 + 0.764026 w_k1_DIS_s112
                                 + 0.788179 w_k1_FDX_s112
                                 + 1.15941 w_k1_IBM_s112
                                 + 0.851837 w_k1_INTC_s112
                                 + 0.939324 w_k1_KO_s112
                                 + 1.06162 w_k1_MAT_s112
                                 + 0.801243 w_k1_MSFT_s112
                                 + 0.921116 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.08966 w_k1_ADM_s112
                                 + 1.30361 w_k1_AXP_s112
                                 + 1.26846 w_k1_DIS_s112
                                 + 1.35457 w_k1_FDX_s112
                                 + 0.929303 w_k1_IBM_s112
                                 + 1.36166 w_k1_INTC_s112
                                 + 1.00233 w_k1_KO_s112
                                 + 0.966243 w_k1_MAT_s112
                                 + 1.17936 w_k1_MSFT_s112
                                 + 0.886409 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.999879 w_k1_ADM_s112
                                 + 1.03537 w_k1_AXP_s112
                                 + 0.996215 w_k1_DIS_s112
                                 + 0.968717 w_k1_FDX_s112
                                 + 1.44691 w_k1_IBM_s112
                                 + 1.07858 w_k1_INTC_s112
                                 + 1.04295 w_k1_KO_s112 + 1.23196 w_k1_MAT_s112
                                 + 0.922831 w_k1_MSFT_s112
                                 + 1.15804 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.11604 w_k1_ADM_s112
                                 + 1.61937 w_k1_AXP_s112
                                 + 1.42201 w_k1_DIS_s112
                                 + 1.35433 w_k1_FDX_s112
                                 + 1.15437 w_k1_IBM_s112
                                 + 1.34978 w_k1_INTC_s112
                                 + 1.20831 w_k1_KO_s112 + 1.65913 w_k1_MAT_s112
                                 + 1.16507 w_k1_MSFT_s112
                                 + 1.13789 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.817845 w_k1_ADM_s112
                                 + 0.832162 w_k1_AXP_s112
                                 + 0.659521 w_k1_DIS_s112
                                 + 0.784581 w_k1_FDX_s112
                                 + 1.00355 w_k1_IBM_s112
                                 + 0.963826 w_k1_INTC_s112
                                 + 0.788983 w_k1_KO_s112
                                 + 0.960017 w_k1_MAT_s112
                                 + 0.804129 w_k1_MSFT_s112
                                 + 0.813262 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.676759 w_k1_ADM_s113
                                 + 0.874243 w_k1_AXP_s113
                                 + 0.725341 w_k1_DIS_s113
                                 + 0.813975 w_k1_FDX_s113
                                 + 1.13527 w_k1_IBM_s113
                                 + 1.13821 w_k1_INTC_s113
                                 + 1.05136 w_k1_KO_s113
                                 + 0.928197 w_k1_MAT_s113
                                 + 0.920707 w_k1_MSFT_s113
                                 + 0.920535 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.869894 w_k1_ADM_s113
                                 + 1.1166 w_k1_AXP_s113 + 0.97004 w_k1_DIS_s113
                                 + 0.948933 w_k1_FDX_s113
                                 + 1.32807 w_k1_IBM_s113
                                 + 0.937673 w_k1_INTC_s113
                                 + 1.01615 w_k1_KO_s113 + 1.44188 w_k1_MAT_s113
                                 + 0.858894 w_k1_MSFT_s113
                                 + 0.902293 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.868966 w_k1_ADM_s113
                                 + 1.00876 w_k1_AXP_s113
                                 + 1.59955 w_k1_DIS_s113
                                 + 1.16409 w_k1_FDX_s113
                                 + 1.07569 w_k1_IBM_s113
                                 + 1.4389 w_k1_INTC_s113 + 1.52146 w_k1_KO_s113
                                 + 1.06398 w_k1_MAT_s113
                                 + 1.35629 w_k1_MSFT_s113
                                 + 1.01616 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.23895 w_k1_ADM_s113
                                 + 1.25713 w_k1_AXP_s113
                                 + 1.29292 w_k1_DIS_s113
                                 + 1.29525 w_k1_FDX_s113
                                 + 0.936009 w_k1_IBM_s113
                                 + 1.2646 w_k1_INTC_s113 + 1.0493 w_k1_KO_s113
                                 + 0.844735 w_k1_MAT_s113
                                 + 1.24497 w_k1_MSFT_s113
                                 + 1.06615 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.883134 w_k1_ADM_s113
                                 + 0.883234 w_k1_AXP_s113
                                 + 1.01981 w_k1_DIS_s113
                                 + 1.00765 w_k1_FDX_s113
                                 + 0.813465 w_k1_IBM_s113
                                 + 0.974371 w_k1_INTC_s113
                                 + 1.02681 w_k1_KO_s113 + 0.98185 w_k1_MAT_s113
                                 + 1.04274 w_k1_MSFT_s113
                                 + 0.94057 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.738816 w_k1_ADM_s113
                                 + 0.990279 w_k1_AXP_s113
                                 + 0.893098 w_k1_DIS_s113
                                 + 0.983044 w_k1_FDX_s113
                                 + 1.17269 w_k1_IBM_s113
                                 + 1.09101 w_k1_INTC_s113
                                 + 1.12534 w_k1_KO_s113 + 1.11734 w_k1_MAT_s113
                                 + 1.01265 w_k1_MSFT_s113
                                 + 0.890822 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.676117 w_k1_ADM_s113
                                 + 0.585448 w_k1_AXP_s113
                                 + 0.574593 w_k1_DIS_s113
                                 + 0.767372 w_k1_FDX_s113
                                 + 0.728949 w_k1_IBM_s113
                                 + 0.893226 w_k1_INTC_s113
                                 + 0.750991 w_k1_KO_s113
                                 + 0.499768 w_k1_MAT_s113
                                 + 0.813137 w_k1_MSFT_s113
                                 + 0.829771 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.941614 w_k1_ADM_s113
                                 + 1.09683 w_k1_AXP_s113
                                 + 1.26157 w_k1_DIS_s113
                                 + 1.06443 w_k1_FDX_s113
                                 + 1.91806 w_k1_IBM_s113
                                 + 1.00825 w_k1_INTC_s113
                                 + 1.42065 w_k1_KO_s113 + 1.24937 w_k1_MAT_s113
                                 + 1.05775 w_k1_MSFT_s113
                                 + 1.12232 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.21617 w_k1_ADM_s114
                                 + 1.08905 w_k1_AXP_s114
                                 + 1.20066 w_k1_DIS_s114
                                 + 1.12045 w_k1_FDX_s114
                                 + 0.972169 w_k1_IBM_s114
                                 + 1.00362 w_k1_INTC_s114
                                 + 0.998944 w_k1_KO_s114
                                 + 0.977623 w_k1_MAT_s114
                                 + 1.03172 w_k1_MSFT_s114
                                 + 1.14616 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.722054 w_k1_ADM_s114
                                 + 0.766978 w_k1_AXP_s114
                                 + 0.748207 w_k1_DIS_s114
                                 + 0.831889 w_k1_FDX_s114
                                 + 0.725113 w_k1_IBM_s114
                                 + 0.766678 w_k1_INTC_s114
                                 + 0.619982 w_k1_KO_s114
                                 + 0.697485 w_k1_MAT_s114
                                 + 0.804869 w_k1_MSFT_s114
                                 + 0.925639 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.878415 w_k1_ADM_s114
                                 + 1.0296 w_k1_AXP_s114
                                 + 0.943119 w_k1_DIS_s114
                                 + 0.849595 w_k1_FDX_s114
                                 + 1.24794 w_k1_IBM_s114
                                 + 0.860349 w_k1_INTC_s114
                                 + 0.990419 w_k1_KO_s114
                                 + 1.25545 w_k1_MAT_s114
                                 + 0.806629 w_k1_MSFT_s114
                                 + 0.973038 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.17121 w_k1_ADM_s114
                                 + 1.364 w_k1_AXP_s114 + 1.5687 w_k1_DIS_s114
                                 + 1.11579 w_k1_FDX_s114
                                 + 1.39628 w_k1_IBM_s114
                                 + 1.1612 w_k1_INTC_s114 + 1.55339 w_k1_KO_s114
                                 + 1.53675 w_k1_MAT_s114
                                 + 1.17452 w_k1_MSFT_s114
                                 + 1.24213 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.898194 w_k1_ADM_s114
                                 + 1.02342 w_k1_AXP_s114
                                 + 1.16847 w_k1_DIS_s114
                                 + 1.12382 w_k1_FDX_s114
                                 + 0.884202 w_k1_IBM_s114
                                 + 1.0414 w_k1_INTC_s114
                                 + 0.919813 w_k1_KO_s114
                                 + 0.843707 w_k1_MAT_s114
                                 + 1.10963 w_k1_MSFT_s114
                                 + 1.04445 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.52348 w_k1_ADM_s114
                                 + 1.69014 w_k1_AXP_s114
                                 + 2.04331 w_k1_DIS_s114
                                 + 1.61684 w_k1_FDX_s114
                                 + 0.97055 w_k1_IBM_s114
                                 + 1.33178 w_k1_INTC_s114
                                 + 1.40303 w_k1_KO_s114 + 1.29567 w_k1_MAT_s114
                                 + 1.48267 w_k1_MSFT_s114
                                 + 1.47543 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.640954 w_k1_ADM_s114
                                 + 0.654529 w_k1_AXP_s114
                                 + 0.595463 w_k1_DIS_s114
                                 + 0.657135 w_k1_FDX_s114
                                 + 0.949158 w_k1_IBM_s114
                                 + 0.741621 w_k1_INTC_s114
                                 + 0.776655 w_k1_KO_s114
                                 + 0.950142 w_k1_MAT_s114
                                 + 0.698392 w_k1_MSFT_s114
                                 + 0.827432 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.25045 w_k1_ADM_s114
                                 + 1.48936 w_k1_AXP_s114
                                 + 1.34149 w_k1_DIS_s114
                                 + 1.33194 w_k1_FDX_s114
                                 + 1.05315 w_k1_IBM_s114
                                 + 1.37013 w_k1_INTC_s114
                                 + 0.871789 w_k1_KO_s114
                                 + 0.74752 w_k1_MAT_s114
                                 + 1.0615 w_k1_MSFT_s114
                                 + 0.980379 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.11053 w_k1_ADM_s115
                                 + 0.893761 w_k1_AXP_s115
                                 + 1.3281 w_k1_DIS_s115 + 1.11411 w_k1_FDX_s115
                                 + 0.745885 w_k1_IBM_s115
                                 + 1.16796 w_k1_INTC_s115
                                 + 0.820051 w_k1_KO_s115
                                 + 0.475906 w_k1_MAT_s115
                                 + 1.28373 w_k1_MSFT_s115
                                 + 1.02491 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.1063 w_k1_ADM_s115
                                 + 1.08177 w_k1_AXP_s115
                                 + 1.18157 w_k1_DIS_s115
                                 + 1.06182 w_k1_FDX_s115 + 1.0171 w_k1_IBM_s115
                                 + 0.915115 w_k1_INTC_s115
                                 + 1.10535 w_k1_KO_s115 + 1.05072 w_k1_MAT_s115
                                 + 1.05742 w_k1_MSFT_s115
                                 + 1.06718 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.799605 w_k1_ADM_s115
                                 + 0.623452 w_k1_AXP_s115
                                 + 0.707284 w_k1_DIS_s115
                                 + 0.747439 w_k1_FDX_s115
                                 + 1.03891 w_k1_IBM_s115
                                 + 0.892313 w_k1_INTC_s115
                                 + 0.946188 w_k1_KO_s115
                                 + 0.835334 w_k1_MAT_s115
                                 + 0.880789 w_k1_MSFT_s115
                                 + 0.943183 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.04586 w_k1_ADM_s115
                                 + 1.01512 w_k1_AXP_s115
                                 + 1.33023 w_k1_DIS_s115
                                 + 1.11386 w_k1_FDX_s115
                                 + 1.43258 w_k1_IBM_s115
                                 + 0.981089 w_k1_INTC_s115
                                 + 1.2429 w_k1_KO_s115 + 1.44679 w_k1_MAT_s115
                                 + 1.24526 w_k1_MSFT_s115
                                 + 1.25832 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.37245 w_k1_ADM_s115
                                 + 1.39544 w_k1_AXP_s115
                                 + 1.59111 w_k1_DIS_s115
                                 + 1.52779 w_k1_FDX_s115
                                 + 1.11559 w_k1_IBM_s115
                                 + 1.22982 w_k1_INTC_s115
                                 + 1.27293 w_k1_KO_s115
                                 + 0.849715 w_k1_MAT_s115
                                 + 1.21019 w_k1_MSFT_s115
                                 + 1.28033 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.845627 w_k1_ADM_s115
                                 + 0.863193 w_k1_AXP_s115
                                 + 0.643345 w_k1_DIS_s115
                                 + 0.780203 w_k1_FDX_s115
                                 + 0.693877 w_k1_IBM_s115
                                 + 0.802042 w_k1_INTC_s115
                                 + 0.67678 w_k1_KO_s115
                                 + 0.610057 w_k1_MAT_s115
                                 + 0.72125 w_k1_MSFT_s115
                                 + 0.782275 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.32952 w_k1_ADM_s115
                                 + 1.28701 w_k1_AXP_s115
                                 + 1.24962 w_k1_DIS_s115
                                 + 1.16467 w_k1_FDX_s115 + 1.4277 w_k1_IBM_s115
                                 + 1.02898 w_k1_INTC_s115 + 1.0925 w_k1_KO_s115
                                 + 0.926037 w_k1_MAT_s115
                                 + 0.89894 w_k1_MSFT_s115
                                 + 1.20857 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.931093 w_k1_ADM_s115
                                 + 0.702153 w_k1_AXP_s115
                                 + 0.849542 w_k1_DIS_s115
                                 + 0.895129 w_k1_FDX_s115
                                 + 0.863818 w_k1_IBM_s115
                                 + 0.866129 w_k1_INTC_s115
                                 + 0.81798 w_k1_KO_s115
                                 + 0.606487 w_k1_MAT_s115
                                 + 0.98539 w_k1_MSFT_s115
                                 + 0.961239 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.95725 w_k1_ADM_s116
                                 + 1.38888 w_k1_AXP_s116 + 1.0513 w_k1_DIS_s116
                                 + 1.18674 w_k1_FDX_s116
                                 + 1.05387 w_k1_IBM_s116
                                 + 1.2095 w_k1_INTC_s116
                                 + 0.980484 w_k1_KO_s116
                                 + 1.46627 w_k1_MAT_s116
                                 + 1.17789 w_k1_MSFT_s116
                                 + 0.934348 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.800865 w_k1_ADM_s116
                                 + 0.69621 w_k1_AXP_s116
                                 + 0.780669 w_k1_DIS_s116
                                 + 0.845691 w_k1_FDX_s116
                                 + 0.660683 w_k1_IBM_s116
                                 + 0.893753 w_k1_INTC_s116
                                 + 0.883354 w_k1_KO_s116
                                 + 0.914522 w_k1_MAT_s116
                                 + 0.919614 w_k1_MSFT_s116
                                 + 0.85116 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.04812 w_k1_ADM_s116
                                 + 1.50503 w_k1_AXP_s116
                                 + 1.76763 w_k1_DIS_s116
                                 + 1.33688 w_k1_FDX_s116
                                 + 1.41918 w_k1_IBM_s116
                                 + 1.45753 w_k1_INTC_s116
                                 + 1.55413 w_k1_KO_s116 + 1.3121 w_k1_MAT_s116
                                 + 1.39981 w_k1_MSFT_s116
                                 + 1.05718 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.946903 w_k1_ADM_s116
                                 + 0.904114 w_k1_AXP_s116
                                 + 1.06976 w_k1_DIS_s116
                                 + 0.960592 w_k1_FDX_s116
                                 + 0.995182 w_k1_IBM_s116
                                 + 1.00259 w_k1_INTC_s116
                                 + 0.940661 w_k1_KO_s116
                                 + 1.37837 w_k1_MAT_s116
                                 + 1.12568 w_k1_MSFT_s116
                                 + 1.00759 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.92115 w_k1_ADM_s116
                                 + 1.10523 w_k1_AXP_s116 + 1.1163 w_k1_DIS_s116
                                 + 0.961921 w_k1_FDX_s116
                                 + 1.50567 w_k1_IBM_s116
                                 + 1.03741 w_k1_INTC_s116
                                 + 1.20479 w_k1_KO_s116 + 1.51445 w_k1_MAT_s116
                                 + 1.07056 w_k1_MSFT_s116 + 1.017 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.860223 w_k1_ADM_s116
                                 + 1.36799 w_k1_AXP_s116
                                 + 1.30044 w_k1_DIS_s116
                                 + 1.14378 w_k1_FDX_s116
                                 + 0.917489 w_k1_IBM_s116
                                 + 1.14149 w_k1_INTC_s116
                                 + 1.15295 w_k1_KO_s116 + 1.38754 w_k1_MAT_s116
                                 + 1.39142 w_k1_MSFT_s116
                                 + 1.06863 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.27174 w_k1_ADM_s116
                                 + 1.58269 w_k1_AXP_s116
                                 + 1.59325 w_k1_DIS_s116
                                 + 1.52812 w_k1_FDX_s116
                                 + 0.799128 w_k1_IBM_s116
                                 + 1.17833 w_k1_INTC_s116
                                 + 1.22373 w_k1_KO_s116 + 1.45933 w_k1_MAT_s116
                                 + 1.60595 w_k1_MSFT_s116
                                 + 1.15466 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.883881 w_k1_ADM_s116
                                 + 0.887036 w_k1_AXP_s116
                                 + 0.725827 w_k1_DIS_s116
                                 + 0.847305 w_k1_FDX_s116
                                 + 1.0177 w_k1_IBM_s116
                                 + 1.02326 w_k1_INTC_s116
                                 + 1.02712 w_k1_KO_s116 + 1.2985 w_k1_MAT_s116
                                 + 0.968232 w_k1_MSFT_s116
                                 + 0.865702 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.707035 w_k1_ADM_s117
                                 + 0.632788 w_k1_AXP_s117
                                 + 0.507996 w_k1_DIS_s117
                                 + 0.671794 w_k1_FDX_s117
                                 + 0.823673 w_k1_IBM_s117
                                 + 0.581152 w_k1_INTC_s117
                                 + 0.623954 w_k1_KO_s117
                                 + 0.893789 w_k1_MAT_s117
                                 + 0.640128 w_k1_MSFT_s117
                                 + 0.770807 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.812707 w_k1_ADM_s117
                                 + 0.886871 w_k1_AXP_s117
                                 + 0.818469 w_k1_DIS_s117
                                 + 0.876298 w_k1_FDX_s117
                                 + 0.717386 w_k1_IBM_s117
                                 + 0.885228 w_k1_INTC_s117
                                 + 0.725198 w_k1_KO_s117
                                 + 0.975532 w_k1_MAT_s117
                                 + 0.836765 w_k1_MSFT_s117
                                 + 0.866782 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.0112 w_k1_ADM_s117
                                 + 0.992207 w_k1_AXP_s117
                                 + 1.19131 w_k1_DIS_s117
                                 + 1.11826 w_k1_FDX_s117
                                 + 0.802211 w_k1_IBM_s117
                                 + 0.983024 w_k1_INTC_s117
                                 + 0.919157 w_k1_KO_s117
                                 + 0.914386 w_k1_MAT_s117
                                 + 1.01816 w_k1_MSFT_s117
                                 + 0.919411 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.28932 w_k1_ADM_s117
                                 + 1.63222 w_k1_AXP_s117
                                 + 1.91459 w_k1_DIS_s117
                                 + 1.34263 w_k1_FDX_s117
                                 + 1.42082 w_k1_IBM_s117
                                 + 0.980265 w_k1_INTC_s117
                                 + 1.14737 w_k1_KO_s117 + 1.33741 w_k1_MAT_s117
                                 + 1.08895 w_k1_MSFT_s117
                                 + 1.16238 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.827644 w_k1_ADM_s117
                                 + 0.962971 w_k1_AXP_s117 + 1.177 w_k1_DIS_s117
                                 + 0.888479 w_k1_FDX_s117
                                 + 1.44228 w_k1_IBM_s117
                                 + 0.876021 w_k1_INTC_s117
                                 + 1.10922 w_k1_KO_s117 + 1.35246 w_k1_MAT_s117
                                 + 0.89996 w_k1_MSFT_s117
                                 + 0.999779 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.895462 w_k1_ADM_s117
                                 + 0.860556 w_k1_AXP_s117
                                 + 0.963139 w_k1_DIS_s117
                                 + 0.843475 w_k1_FDX_s117
                                 + 0.925242 w_k1_IBM_s117
                                 + 0.89229 w_k1_INTC_s117
                                 + 1.01133 w_k1_KO_s117
                                 + 0.851334 w_k1_MAT_s117
                                 + 1.00196 w_k1_MSFT_s117
                                 + 0.962184 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.99753 w_k1_ADM_s117
                                 + 1.51576 w_k1_AXP_s117
                                 + 1.30188 w_k1_DIS_s117
                                 + 1.38294 w_k1_FDX_s117
                                 + 0.824266 w_k1_IBM_s117
                                 + 1.32453 w_k1_INTC_s117
                                 + 1.00202 w_k1_KO_s117 + 0.71244 w_k1_MAT_s117
                                 + 1.00521 w_k1_MSFT_s117
                                 + 1.05684 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.966041 w_k1_ADM_s117
                                 + 1.32294 w_k1_AXP_s117
                                 + 1.45026 w_k1_DIS_s117
                                 + 1.14161 w_k1_FDX_s117
                                 + 1.05683 w_k1_IBM_s117
                                 + 1.05039 w_k1_INTC_s117
                                 + 1.13686 w_k1_KO_s117 + 1.45041 w_k1_MAT_s117
                                 + 1.16016 w_k1_MSFT_s117
                                 + 1.16853 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00759467598068827 w_k1_ADM_s000
                                 + 0.0138782580609833 w_k1_ADM_s001
                                 + 0.017457845265768 w_k1_ADM_s002
                                 + 0.0313385789555228 w_k1_ADM_s003
                                 + 0.0165542734206895 w_k1_ADM_s004
                                 + 0.00663420188649389 w_k1_ADM_s005
                                 + 0.0194314267972587 w_k1_ADM_s006
                                 + 0.0287378170183361 w_k1_ADM_s007
                                 + 0.0292729654978265 w_k1_ADM_s010
                                 + 0.0753333351074586 w_k1_ADM_s011
                                 + 0.021631736522415 w_k1_ADM_s012
                                 + 0.0833936275526373 w_k1_ADM_s013
                                 + 0.0222070425471421 w_k1_ADM_s014
                                 + 0.0195312882473848 w_k1_ADM_s015
                                 + 0.0189003225634467 w_k1_ADM_s016
                                 + 0.0363894686984751 w_k1_ADM_s017
                                 + 0.0296958314289782 w_k1_ADM_s100
                                 + 0.077280087804036 w_k1_ADM_s101
                                 + 0.0285130918731134 w_k1_ADM_s102
                                 + 0.0896025446902754 w_k1_ADM_s103
                                 + 0.0403689913230222 w_k1_ADM_s104
                                 + 0.0413157737762746 w_k1_ADM_s105
                                 + 0.0815805184129228 w_k1_ADM_s106
                                 + 0.0459953564369544 w_k1_ADM_s107
                                 + 0.00970658181208128 w_k1_ADM_s110
                                 + 0.027292801200194 w_k1_ADM_s111
                                 + 0.0141645593778623 w_k1_ADM_s112
                                 + 0.0108685824867557 w_k1_ADM_s113
                                 + 0.0116297108298464 w_k1_ADM_s114
                                 + 0.0157376661857147 w_k1_ADM_s115
                                 + 0.0125783771110056 w_k1_ADM_s116
                                 + 0.0236245548203611 w_k1_ADM_s117
                                 + 0.00821083627667423 w_k1_AXP_s000
                                 + 0.0157796659448949 w_k1_AXP_s001
                                 + 0.0162567051505863 w_k1_AXP_s002
                                 + 0.0316150678596761 w_k1_AXP_s003
                                 + 0.0153115422391628 w_k1_AXP_s004
                                 + 0.00779090210442375 w_k1_AXP_s005
                                 + 0.0208654337676738 w_k1_AXP_s006
                                 + 0.0227902200409914 w_k1_AXP_s007
                                 + 0.0312123833506267 w_k1_AXP_s010
                                 + 0.0718210922270207 w_k1_AXP_s011
                                 + 0.0226524468096605 w_k1_AXP_s012
                                 + 0.0774130402777873 w_k1_AXP_s013
                                 + 0.0263157311945232 w_k1_AXP_s014
                                 + 0.0203550348957464 w_k1_AXP_s015
                                 + 0.0196631055641591 w_k1_AXP_s016
                                 + 0.0399987740419293 w_k1_AXP_s017
                                 + 0.02577413080404 w_k1_AXP_s100
                                 + 0.0878344018557878 w_k1_AXP_s101
                                 + 0.028936444760861 w_k1_AXP_s102
                                 + 0.091968947270998 w_k1_AXP_s103
                                 + 0.0362627439581766 w_k1_AXP_s104
                                 + 0.0421077867801938 w_k1_AXP_s105
                                 + 0.0920541569977831 w_k1_AXP_s106
                                 + 0.0437501134699816 w_k1_AXP_s107
                                 + 0.0113905442797759 w_k1_AXP_s110
                                 + 0.0310917181498211 w_k1_AXP_s111
                                 + 0.0149333012841564 w_k1_AXP_s112
                                 + 0.0119581493161707 w_k1_AXP_s113
                                 + 0.0124328100903331 w_k1_AXP_s114
                                 + 0.0144070758907592 w_k1_AXP_s115
                                 + 0.0143250892709447 w_k1_AXP_s116
                                 + 0.027255224035632 w_k1_AXP_s117
                                 + 0.00835172762252788 w_k1_DIS_s000
                                 + 0.0154052310482246 w_k1_DIS_s001
                                 + 0.0172545731204145 w_k1_DIS_s002
                                 + 0.0313613645602457 w_k1_DIS_s003
                                 + 0.016656491819271 w_k1_DIS_s004
                                 + 0.00778411506285794 w_k1_DIS_s005
                                 + 0.0195947744534428 w_k1_DIS_s006
                                 + 0.023687006094265 w_k1_DIS_s007
                                 + 0.0286848971308699 w_k1_DIS_s010
                                 + 0.075670833376301 w_k1_DIS_s011
                                 + 0.0227335226121546 w_k1_DIS_s012
                                 + 0.0783516167087114 w_k1_DIS_s013
                                 + 0.0255906551522306 w_k1_DIS_s014
                                 + 0.022834315411641 w_k1_DIS_s015
                                 + 0.0204941415780305 w_k1_DIS_s016
                                 + 0.0378894276261275 w_k1_DIS_s017
                                 + 0.0272450223316713 w_k1_DIS_s100
                                 + 0.0865102939872321 w_k1_DIS_s101
                                 + 0.0301861418561105 w_k1_DIS_s102
                                 + 0.101156124630304 w_k1_DIS_s103
                                 + 0.0404015500660047 w_k1_DIS_s104
                                 + 0.0455889702782392 w_k1_DIS_s105
                                 + 0.086971195117195 w_k1_DIS_s106
                                 + 0.0402778202211042 w_k1_DIS_s107
                                 + 0.00893422898843211 w_k1_DIS_s110
                                 + 0.0299354266796446 w_k1_DIS_s111
                                 + 0.0140468896032245 w_k1_DIS_s112
                                 + 0.0123124388306599 w_k1_DIS_s113
                                 + 0.0131358734815218 w_k1_DIS_s114
                                 + 0.0159637034594306 w_k1_DIS_s115
                                 + 0.0141212628477665 w_k1_DIS_s116
                                 + 0.0291290359776912 w_k1_DIS_s117
                                 + 0.00832381824222365 w_k1_FDX_s000
                                 + 0.0148288798036091 w_k1_FDX_s001
                                 + 0.016186155625842 w_k1_FDX_s002
                                 + 0.0325719311505068 w_k1_FDX_s003
                                 + 0.0151039410623698 w_k1_FDX_s004
                                 + 0.00762751616021501 w_k1_FDX_s005
                                 + 0.0200210419653663 w_k1_FDX_s006
                                 + 0.0226796591528857 w_k1_FDX_s007
                                 + 0.0275566552280907 w_k1_FDX_s010
                                 + 0.0746038718382694 w_k1_FDX_s011
                                 + 0.0216130600270907 w_k1_FDX_s012
                                 + 0.075778137671587 w_k1_FDX_s013
                                 + 0.0239905686127114 w_k1_FDX_s014
                                 + 0.020138815014865 w_k1_FDX_s015
                                 + 0.019074668255951 w_k1_FDX_s016
                                 + 0.0376328679480104 w_k1_FDX_s017
                                 + 0.0264801121806324 w_k1_FDX_s100
                                 + 0.0896788174859192 w_k1_FDX_s101
                                 + 0.0265102264718359 w_k1_FDX_s102
                                 + 0.0950723703963928 w_k1_FDX_s103
                                 + 0.0392396106884531 w_k1_FDX_s104
                                 + 0.0425071902792153 w_k1_FDX_s105
                                 + 0.0838629494849781 w_k1_FDX_s106
                                 + 0.0417397348651432 w_k1_FDX_s107
                                 + 0.00978318907113839 w_k1_FDX_s110
                                 + 0.0296142675340459 w_k1_FDX_s111
                                 + 0.0146570963285719 w_k1_FDX_s112
                                 + 0.0124108057779458 w_k1_FDX_s113
                                 + 0.0122764235488455 w_k1_FDX_s114
                                 + 0.0154800416498113 w_k1_FDX_s115
                                 + 0.0137065008857349 w_k1_FDX_s116
                                 + 0.0257502800243016 w_k1_FDX_s117
                                 + 0.0094179158079974 w_k1_IBM_s000
                                 + 0.0146702361635032 w_k1_IBM_s001
                                 + 0.0156872470104206 w_k1_IBM_s002
                                 + 0.0316807219143274 w_k1_IBM_s003
                                 + 0.0155015461300863 w_k1_IBM_s004
                                 + 0.00760972865585229 w_k1_IBM_s005
                                 + 0.0196890108423733 w_k1_IBM_s006
                                 + 0.0229149386726392 w_k1_IBM_s007
                                 + 0.0270054602675546 w_k1_IBM_s010
                                 + 0.0769243065385459 w_k1_IBM_s011
                                 + 0.0224072569917724 w_k1_IBM_s012
                                 + 0.0763939614543001 w_k1_IBM_s013
                                 + 0.0244557365218361 w_k1_IBM_s014
                                 + 0.0211417250512802 w_k1_IBM_s015
                                 + 0.0193798828852504 w_k1_IBM_s016
                                 + 0.0388548586220924 w_k1_IBM_s017
                                 + 0.0262165716138991 w_k1_IBM_s100
                                 + 0.089211981360975 w_k1_IBM_s101
                                 + 0.0271246090910543 w_k1_IBM_s102
                                 + 0.0942072828430108 w_k1_IBM_s103
                                 + 0.0395793085786952 w_k1_IBM_s104
                                 + 0.0459826112620684 w_k1_IBM_s105
                                 + 0.0865820469550278 w_k1_IBM_s106
                                 + 0.0420740242012897 w_k1_IBM_s107
                                 + 0.0102191186248659 w_k1_IBM_s110
                                 + 0.0307721432561054 w_k1_IBM_s111
                                 + 0.0153534973162497 w_k1_IBM_s112
                                 + 0.0127139101933473 w_k1_IBM_s113
                                 + 0.0114532910690464 w_k1_IBM_s114
                                 + 0.0148430086826899 w_k1_IBM_s115
                                 + 0.0138603298683975 w_k1_IBM_s116
                                 + 0.0248049391753166 w_k1_IBM_s117
                                 + 0.00927892399684787 w_k1_INTC_s000
                                 + 0.0151058345710852 w_k1_INTC_s001
                                 + 0.0155102270198947 w_k1_INTC_s002
                                 + 0.0330607852977389 w_k1_INTC_s003
                                 + 0.0140087097608296 w_k1_INTC_s004
                                 + 0.00784395071284468 w_k1_INTC_s005
                                 + 0.0191529070306503 w_k1_INTC_s006
                                 + 0.0208241782213504 w_k1_INTC_s007
                                 + 0.0308583672527202 w_k1_INTC_s010
                                 + 0.0777142443998952 w_k1_INTC_s011
                                 + 0.0225021329125221 w_k1_INTC_s012
                                 + 0.0755786330894574 w_k1_INTC_s013
                                 + 0.0217862451333394 w_k1_INTC_s014
                                 + 0.0231908704642355 w_k1_INTC_s015
                                 + 0.0172795079382606 w_k1_INTC_s016
                                 + 0.0391979619517683 w_k1_INTC_s017
                                 + 0.026549024220045 w_k1_INTC_s100
                                 + 0.0861423238218194 w_k1_INTC_s101
                                 + 0.0266847607100997 w_k1_INTC_s102
                                 + 0.0913155399815625 w_k1_INTC_s103
                                 + 0.0360957184270547 w_k1_INTC_s104
                                 + 0.0482161460888368 w_k1_INTC_s105
                                 + 0.07337509251571 w_k1_INTC_s106
                                 + 0.0419378942282682 w_k1_INTC_s107
                                 + 0.00945933009427708 w_k1_INTC_s110
                                 + 0.0281764508738866 w_k1_INTC_s111
                                 + 0.0152543067957798 w_k1_INTC_s112
                                 + 0.0134464942324126 w_k1_INTC_s113
                                 + 0.0115942460522829 w_k1_INTC_s114
                                 + 0.0145749994682152 w_k1_INTC_s115
                                 + 0.0143600780200779 w_k1_INTC_s116
                                 + 0.0242235777971142 w_k1_INTC_s117
                                 + 0.00838983259329682 w_k1_KO_s000
                                 + 0.014931844437029 w_k1_KO_s001
                                 + 0.0182531809038554 w_k1_KO_s002
                                 + 0.0322178347430817 w_k1_KO_s003
                                 + 0.014984611650375 w_k1_KO_s004
                                 + 0.00729268624615296 w_k1_KO_s005
                                 + 0.0184369216845703 w_k1_KO_s006
                                 + 0.0265104166933326 w_k1_KO_s007
                                 + 0.027109632399612 w_k1_KO_s010
                                 + 0.0718657776158796 w_k1_KO_s011
                                 + 0.0215803374398694 w_k1_KO_s012
                                 + 0.0787041641146149 w_k1_KO_s013
                                 + 0.0253748826478053 w_k1_KO_s014
                                 + 0.0225700135095557 w_k1_KO_s015
                                 + 0.0199629220366116 w_k1_KO_s016
                                 + 0.0377140891573589 w_k1_KO_s017
                                 + 0.0263438665751726 w_k1_KO_s100
                                 + 0.0863052303402464 w_k1_KO_s101
                                 + 0.0279359854938554 w_k1_KO_s102
                                 + 0.0983139628464536 w_k1_KO_s103
                                 + 0.0391069565102976 w_k1_KO_s104
                                 + 0.0443932372952767 w_k1_KO_s105
                                 + 0.0851135967339634 w_k1_KO_s106
                                 + 0.0426288730053639 w_k1_KO_s107
                                 + 0.00961181570543498 w_k1_KO_s110
                                 + 0.0311444891129845 w_k1_KO_s111
                                 + 0.0138955872728968 w_k1_KO_s112
                                 + 0.0129724796325258 w_k1_KO_s113
                                 + 0.011217199264151 w_k1_KO_s114
                                 + 0.0145874995610071 w_k1_KO_s115
                                 + 0.0144438862989472 w_k1_KO_s116
                                 + 0.0245562791401504 w_k1_KO_s117
                                 + 0.0126613735490861 w_k1_MAT_s000
                                 + 0.0138912659675143 w_k1_MAT_s001
                                 + 0.0184107671621901 w_k1_MAT_s002
                                 + 0.0348927078501896 w_k1_MAT_s003
                                 + 0.0173944163252284 w_k1_MAT_s004
                                 + 0.00872605340254794 w_k1_MAT_s005
                                 + 0.0228364764916336 w_k1_MAT_s006
                                 + 0.0206431454661056 w_k1_MAT_s007
                                 + 0.0248468415785778 w_k1_MAT_s010
                                 + 0.0647789731856894 w_k1_MAT_s011
                                 + 0.021279231308707 w_k1_MAT_s012
                                 + 0.0688707443155157 w_k1_MAT_s013
                                 + 0.0291228296193741 w_k1_MAT_s014
                                 + 0.0235454333276211 w_k1_MAT_s015
                                 + 0.0214126108294261 w_k1_MAT_s016
                                 + 0.0452378791297699 w_k1_MAT_s017
                                 + 0.0227568213012039 w_k1_MAT_s100
                                 + 0.0784768850192596 w_k1_MAT_s101
                                 + 0.0276678014865474 w_k1_MAT_s102
                                 + 0.0972245632813032 w_k1_MAT_s103
                                 + 0.0410885624690079 w_k1_MAT_s104
                                 + 0.0407236803202888 w_k1_MAT_s105
                                 + 0.0941096880139121 w_k1_MAT_s106
                                 + 0.0442107011280027 w_k1_MAT_s107
                                 + 0.0110238963264035 w_k1_MAT_s110
                                 + 0.025865236893544 w_k1_MAT_s111
                                 + 0.0148521768168374 w_k1_MAT_s112
                                 + 0.0115229803802405 w_k1_MAT_s113
                                 + 0.0116431696800659 w_k1_MAT_s114
                                 + 0.0119195961333496 w_k1_MAT_s115
                                 + 0.0177699164704697 w_k1_MAT_s116
                                 + 0.028116468618088 w_k1_MAT_s117
                                 + 0.00900470481446432 w_k1_MSFT_s000
                                 + 0.0139025272021667 w_k1_MSFT_s001
                                 + 0.0148215580628151 w_k1_MSFT_s002
                                 + 0.0308822978427416 w_k1_MSFT_s003
                                 + 0.0150055983609345 w_k1_MSFT_s004
                                 + 0.00715657374700843 w_k1_MSFT_s005
                                 + 0.020042410739178 w_k1_MSFT_s006
                                 + 0.0223451438423722 w_k1_MSFT_s007
                                 + 0.0280025185787341 w_k1_MSFT_s010
                                 + 0.0694627088182376 w_k1_MSFT_s011
                                 + 0.0215308201711264 w_k1_MSFT_s012
                                 + 0.0750309088588181 w_k1_MSFT_s013
                                 + 0.0222531081311674 w_k1_MSFT_s014
                                 + 0.0228260856885133 w_k1_MSFT_s015
                                 + 0.0184696760468945 w_k1_MSFT_s016
                                 + 0.038326504199251 w_k1_MSFT_s017
                                 + 0.0245300238172697 w_k1_MSFT_s100
                                 + 0.0864854803061501 w_k1_MSFT_s101
                                 + 0.0259439255810115 w_k1_MSFT_s102
                                 + 0.0894797494357064 w_k1_MSFT_s103
                                 + 0.0378708186112711 w_k1_MSFT_s104
                                 + 0.0471492587415491 w_k1_MSFT_s105
                                 + 0.0786003273827689 w_k1_MSFT_s106
                                 + 0.0373664759696693 w_k1_MSFT_s107
                                 + 0.0102236792254557 w_k1_MSFT_s110
                                 + 0.0280206719799844 w_k1_MSFT_s111
                                 + 0.0137738325799563 w_k1_MSFT_s112
                                 + 0.0127309152625766 w_k1_MSFT_s113
                                 + 0.0117338780193068 w_k1_MSFT_s114
                                 + 0.0150793185558837 w_k1_MSFT_s115
                                 + 0.0152284954954673 w_k1_MSFT_s116
                                 + 0.025009604121811 w_k1_MSFT_s117
                                 + 0.00803077602848641 w_k1_PG_s000
                                 + 0.013904742471129 w_k1_PG_s001
                                 + 0.016329416526905 w_k1_PG_s002
                                 + 0.0322081694847762 w_k1_PG_s003
                                 + 0.0147336750783104 w_k1_PG_s004
                                 + 0.00711089233700517 w_k1_PG_s005
                                 + 0.0199886570034495 w_k1_PG_s006
                                 + 0.0231232178403219 w_k1_PG_s007
                                 + 0.0253226954462514 w_k1_PG_s010
                                 + 0.077287689183107 w_k1_PG_s011
                                 + 0.020767592340027 w_k1_PG_s012
                                 + 0.0768655448456516 w_k1_PG_s013
                                 + 0.0228281542185454 w_k1_PG_s014
                                 + 0.0221796045455806 w_k1_PG_s015
                                 + 0.0203244407774731 w_k1_PG_s016
                                 + 0.0369628824152516 w_k1_PG_s017
                                 + 0.0262015483063223 w_k1_PG_s100
                                 + 0.0850475033132561 w_k1_PG_s101
                                 + 0.0282102184064242 w_k1_PG_s102
                                 + 0.0909118665385988 w_k1_PG_s103
                                 + 0.0381245079115141 w_k1_PG_s104
                                 + 0.0428896515003811 w_k1_PG_s105
                                 + 0.0945095128299963 w_k1_PG_s106
                                 + 0.0398485533953136 w_k1_PG_s107
                                 + 0.00876372000900407 w_k1_PG_s110
                                 + 0.0299904295329347 w_k1_PG_s111
                                 + 0.0141894665688824 w_k1_PG_s112
                                 + 0.0116971997920069 w_k1_PG_s113
                                 + 0.0125916576269605 w_k1_PG_s114
                                 + 0.0155442361526428 w_k1_PG_s115
                                 + 0.0130084563154193 w_k1_PG_s116
                                 + 0.0254295323919189 w_k1_PG_s117 >= 1.06
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
