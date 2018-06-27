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
 w_k1_ADM_s0_return_constraint:  1.16085 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.935785 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.2085 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.902815 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.977975 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.97362 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.06615 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.08383 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.78575 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.993102 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.04755 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.940273 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.20179 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.809044 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.30357 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.14165 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.980897 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.02982 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.941411 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.33213 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.864264 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.718476 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.582589 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.31844 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.883555 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.989153 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.07849 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.804736 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.872205 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.886517 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.995011 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.00199 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.24101 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.0261 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.33365 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.91356 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.688934 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.783705 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23607 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.862755 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.27382 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.822268 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.00303 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.10113 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.35315 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.38901 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.853488 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.38694 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.1059 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.914868 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.18883 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.762379 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.36775 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.12685 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.33928 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.925807 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.967913 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.29455 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.975287 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.585777 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.47359 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.65598 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.958378 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.09662 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.10511 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.839224 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.719528 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.91985 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.06826 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.915334 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.22629 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.945407 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.66733 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.61239 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.557808 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.727867 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27648 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.847729 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.43907 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.84735 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.05181 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.07548 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.15201 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.86113 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.663639 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.17789 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.920909 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.916674 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.24404 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.680041 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.71027 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.17155 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.24326 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.930599 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.810143 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.22298 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.691635 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.586719 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.443568 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.71878 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.86925 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.12507 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.2315 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.630333 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.619966 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.994939 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.06921 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.624398 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.24169 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.814137 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.60031 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.93773 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.469734 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.678274 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16843 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.890565 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.24102 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.906982 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.923073 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.13137 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.983589 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.49008 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.762861 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.08137 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.0763 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.858164 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.24716 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.868479 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.21953 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.07636 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.05132 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.903373 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.956202 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.2681 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.735602 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.681547 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.646444 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.41574 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.94781 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.04235 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.24755 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.808222 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.852956 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.0511 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.927143 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.73653 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.19438 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.953036 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.28936 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.57557 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.637081 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.763139 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.2014 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.925935 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.16643 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.929706 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17937 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.821147 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.86318 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.39976 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.12017 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.26506 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.834461 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.31529 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.850254 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.5935 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.36337 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.935644 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.7684 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.05468 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.675143 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.638599 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.23465 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.705616 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.597659 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.49811 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.02404 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.47293 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.959279 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.09047 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.649263 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.72023 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.40162 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.15613 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.01572 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.834746 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.62857 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.26339 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.586472 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.728641 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.05551 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.915919 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.27474 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.923402 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.889831 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.08636 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.01253 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.24993 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.780862 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.942712 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.07599 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.933509 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.12317 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.801133 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.1582 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.20678 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.18897 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.908041 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.04757 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.26981 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.04093 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.845236 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.757067 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.58123 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.08222 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.04467 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.25163 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.00358 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.873663 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.01372 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.965095 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.780672 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.04431 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.985613 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.36534 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.25248 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.669598 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.818409 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20076 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.901253 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.20208 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.840135 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.01136 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.968612 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.56297 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.49614 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.91819 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.10813 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.878494 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.04149 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.04343 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.689623 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.52184 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.15721 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.4966 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.09224 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.82508 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.0214 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.03504 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.76378 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.597811 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.57623 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.06182 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.25057 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.10832 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.912346 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.739577 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.07162 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.25496 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.814266 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.06479 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.875509 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.53604 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.40677 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.567416 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.735991 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.34354 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.917873 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.20315 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.767107 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.05607 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.85947 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.93646 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.39235 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.06344 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.32761 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.814394 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.958776 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.814457 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.626943 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.354 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.02709 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.95029 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.15546 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.719639 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.625364 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.28146 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.557923 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.555163 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.73933 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.12269 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.52108 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.969219 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.10758 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.643235 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.594835 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.4717 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.01029 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.874414 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.779525 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 2.06297 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.26801 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.52061 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.593574 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14183 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.910517 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.21863 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.902338 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.954172 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.06359 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.931843 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.28697 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.732749 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.01013 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.915499 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.89376 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.05695 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.880928 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.30674 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.13845 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.02142 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.978124 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.97671 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.28675 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.826506 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.748474 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.666438 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.51528 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.995466 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.07024 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.20104 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.88138 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.85919 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.00525 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.01552 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.712689 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.08358 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.881551 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.32929 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.41227 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.690161 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.783371 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12179 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.907723 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.13813 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.897564 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.988263 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.01005 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.13678 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.25823 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.874928 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.00557 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.935117 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.956393 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.09773 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.828549 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.19961 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.1111 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.1719 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.09677 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.943357 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.10238 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.943541 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.821513 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.728297 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.30896 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.00313 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.16517 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.11 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.938735 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.798504 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.00105 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.09376 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.917428 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.20116 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.979989 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.34537 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.36854 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.719722 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.879375 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.213999765510466 z_k1_s0000
                                 - 0.191999598663501 z_k1_s0001
                                 - 0.150000324956175 z_k1_s0002
                                 - 0.0580000915963709 z_k1_s0003
                                 - 0.143999698997626 z_k1_s0004
                                 - 0.282000049037332 z_k1_s0005
                                 - 0.112000106420544 z_k1_s0006
                                 - 0.125000100530062 z_k1_s0007
                                 - 0.222999980428392 z_k1_s0010
                                 - 0.079000041205602 z_k1_s0011
                                 - 0.21400065516486 z_k1_s0012
                                 - 0.192999592881743 z_k1_s0013
                                 - 0.100000086488124 z_k1_s0014
                                 - 0.103000012242723 z_k1_s0015
                                 - 0.0760000024509371 z_k1_s0016
                                 - 0.1420004030533 z_k1_s0017
                                 - 0.134999841471764 z_k1_s0020
                                 - 0.409000082384541 z_k1_s0021
                                 - 0.119999859086013 z_k1_s0022
                                 - 0.345999734364599 z_k1_s0023
                                 - 0.160999881273698 z_k1_s0024
                                 - 0.0970000267611222 z_k1_s0025
                                 - 0.459998756496739 z_k1_s0026
                                 - 0.382000536089974 z_k1_s0027
                                 - 0.11299961531207 z_k1_s0030
                                 - 0.133000236695129 z_k1_s0031
                                 - 0.0790000058616918 z_k1_s0032
                                 - 0.0980000396752817 z_k1_s0033
                                 - 0.071000091208658 z_k1_s0034
                                 - 0.0780000860926022 z_k1_s0035
                                 - 0.302000478879132 z_k1_s0036
                                 - 0.239000191023445 z_k1_s0037
                                 - 0.342001063017574 z_k1_s0040
                                 - 0.548002285625959 z_k1_s0041
                                 - 0.352001260474602 z_k1_s0042
                                 - 0.243000530696193 z_k1_s0043
                                 - 0.502001377323631 z_k1_s0044
                                 - 0.233000617739804 z_k1_s0045
                                 - 0.255000483143988 z_k1_s0046
                                 - 0.370001615897252 z_k1_s0047
                                 - 0.151999486594191 z_k1_s0050
                                 - 0.226999628020273 z_k1_s0051
                                 - 0.272999140827515 z_k1_s0052
                                 - 0.27399917749708 z_k1_s0053
                                 - 0.265998884140558 z_k1_s0054
                                 - 0.138999444688395 z_k1_s0055
                                 - 0.578997317757923 z_k1_s0056
                                 - 0.263998810801428 z_k1_s0057
                                 - 0.128000186183629 z_k1_s0060
                                 - 0.341000970123008 z_k1_s0061
                                 - 0.412001231436134 z_k1_s0062
                                 - 0.19000036256963 z_k1_s0063
                                 - 0.230000535689756 z_k1_s0064
                                 - 0.346001221638507 z_k1_s0065
                                 - 0.0950001812848151 z_k1_s0066
                                 - 0.0970002818910147 z_k1_s0067
                                 - 0.402999196487041 z_k1_s0070
                                 - 0.304999889115819 z_k1_s0071
                                 - 0.340001406392379 z_k1_s0072
                                 - 0.4250003854896 z_k1_s0073
                                 - 0.189000232788355 z_k1_s0074
                                 - 0.13700001769305 z_k1_s0075
                                 - 0.305999069751743 z_k1_s0076
                                 - 0.639999902325084 z_k1_s0077
                                 - 0.764000518838165 z_k1_s0100
                                 - 0.192000197213907 z_k1_s0101
                                 - 0.175000175841835 z_k1_s0102
                                 - 0.497999080308346 z_k1_s0103
                                 - 0.347000524571288 z_k1_s0104
                                 - 0.400000723696235 z_k1_s0105
                                 - 0.677999969072724 z_k1_s0106
                                 - 0.700002204970199 z_k1_s0107
                                 - 0.613001288932599 z_k1_s0110
                                 - 1.10499972811434 z_k1_s0111
                                 - 0.754001400655918 z_k1_s0112
                                 - 0.606998874537951 z_k1_s0113
                                 - 0.457000534227508 z_k1_s0114
                                 - 0.853998496527931 z_k1_s0115
                                 - 0.537000368127057 z_k1_s0116
                                 - 0.466000380716086 z_k1_s0117
                                 - 0.257000986685546 z_k1_s0120
                                 - 0.290000925218451 z_k1_s0121
                                 - 0.563003296955121 z_k1_s0122
                                 - 0.32500120363332 z_k1_s0123
                                 - 0.155000661263884 z_k1_s0124
                                 - 0.547001911903996 z_k1_s0125
                                 - 0.588002542962464 z_k1_s0126
                                 - 0.610003613654892 z_k1_s0127
                                 - 0.552999225741593 z_k1_s0130
                                 - 0.152999309661666 z_k1_s0131
                                 - 0.464998404117001 z_k1_s0132
                                 - 0.236999131534022 z_k1_s0133
                                 - 0.463997812680158 z_k1_s0134
                                 - 0.648997948885251 z_k1_s0135
                                 - 0.168999552183988 z_k1_s0136
                                 - 0.724998503243407 z_k1_s0137
                                 - 0.52399878756533 z_k1_s0140
                                 - 0.2789994657443 z_k1_s0141
                                 - 0.538997522328186 z_k1_s0142
                                 - 0.704998031742179 z_k1_s0143
                                 - 0.741997922618021 z_k1_s0144
                                 - 0.272999561139958 z_k1_s0145
                                 - 0.685999086776961 z_k1_s0146
                                 - 0.358999152100326 z_k1_s0147
                                 - 0.423001191387515 z_k1_s0150
                                 - 0.171999897071584 z_k1_s0151
                                 - 0.428999710137256 z_k1_s0152
                                 - 0.571998495849348 z_k1_s0153
                                 - 0.252000153307294 z_k1_s0154
                                 - 0.718001570437778 z_k1_s0155
                                 - 0.489001662639551 z_k1_s0156
                                 - 0.297999973725231 z_k1_s0157
                                 - 0.249999970680818 z_k1_s0160
                                 - 0.244999882243204 z_k1_s0161
                                 - 0.282998940809119 z_k1_s0162
                                 - 0.34199959489297 z_k1_s0163
                                 - 0.775999926798058 z_k1_s0164
                                 - 0.440999843677766 z_k1_s0165
                                 - 0.26299997805862 z_k1_s0166
                                 - 0.182000103289231 z_k1_s0167
                                 - 0.762004467825634 z_k1_s0170
                                 - 0.548003493710391 z_k1_s0171
                                 - 0.219001009768456 z_k1_s0172
                                 - 0.231000819415042 z_k1_s0173
                                 - 0.239000807479901 z_k1_s0174
                                 - 0.734001922838098 z_k1_s0175
                                 - 0.194000831502394 z_k1_s0176
                                 - 0.522001376865825 z_k1_s0177
                                 - 0.38899994299779 z_k1_s1000
                                 - 0.163999865161075 z_k1_s1001
                                 - 0.585998598295503 z_k1_s1002
                                 - 0.158999803564732 z_k1_s1003
                                 - 0.343999388630706 z_k1_s1004
                                 - 0.588999712852791 z_k1_s1005
                                 - 0.161999840522538 z_k1_s1006
                                 - 0.301000475301382 z_k1_s1007
                                 - 0.362001182128293 z_k1_s1010
                                 - 0.233000832634441 z_k1_s1011
                                 - 0.937000750060996 z_k1_s1012
                                 - 0.61900195691007 z_k1_s1013
                                 - 0.343000861010585 z_k1_s1014
                                 - 0.638000957123939 z_k1_s1015
                                 - 0.48000324595952 z_k1_s1016
                                 - 0.791001248765127 z_k1_s1017
                                 - 0.354999052021232 z_k1_s1020
                                 - 0.192999958802841 z_k1_s1021
                                 - 0.265000269677303 z_k1_s1022
                                 - 0.751999591801874 z_k1_s1023
                                 - 0.152999985983589 z_k1_s1024
                                 - 0.190999985861865 z_k1_s1025
                                 - 0.443000945422551 z_k1_s1026
                                 - 0.217999877564907 z_k1_s1027
                                 - 0.270999700975167 z_k1_s1030
                                 - 0.406999966283589 z_k1_s1031
                                 - 0.217999967950428 z_k1_s1032
                                 - 0.577997733062009 z_k1_s1033
                                 - 0.515000215616733 z_k1_s1034
                                 - 0.444000441036471 z_k1_s1035
                                 - 0.89199982064122 z_k1_s1036
                                 - 1.05500146228246 z_k1_s1037
                                 - 0.252999226511313 z_k1_s1040
                                 - 0.625997744029188 z_k1_s1041
                                 - 0.26099929351135 z_k1_s1042
                                 - 0.521996873028699 z_k1_s1043
                                 - 0.469998580020956 z_k1_s1044
                                 - 0.551996053032589 z_k1_s1045
                                 - 1.17299750454863 z_k1_s1046
                                 - 0.427998442520009 z_k1_s1047
                                 - 0.399000259701159 z_k1_s1050
                                 - 0.392999944558215 z_k1_s1051
                                 - 1.25900038777975 z_k1_s1052
                                 - 0.383000053720116 z_k1_s1053
                                 - 0.795000123930792 z_k1_s1054
                                 - 0.251999913600676 z_k1_s1055
                                 - 0.544001293754171 z_k1_s1056
                                 - 0.732998612054092 z_k1_s1057
                                 - 0.291998483175943 z_k1_s1060
                                 - 0.40799748363949 z_k1_s1061
                                 - 0.279998518445552 z_k1_s1062
                                 - 0.287998494932479 z_k1_s1063
                                 - 0.500997951221524 z_k1_s1064
                                 - 0.664998457098872 z_k1_s1065
                                 - 0.425996936787405 z_k1_s1066
                                 - 0.432997821784197 z_k1_s1067
                                 - 0.766001103361948 z_k1_s1070
                                 - 0.628001003278364 z_k1_s1071
                                 - 0.243999886611849 z_k1_s1072
                                 - 0.703997826226223 z_k1_s1073
                                 - 0.534000058584637 z_k1_s1074
                                 - 0.304999574765099 z_k1_s1075
                                 - 0.373999624806891 z_k1_s1076
                                 - 0.224999735875842 z_k1_s1077
                                 - 0.216000091713184 z_k1_s1100
                                 - 0.421999444644822 z_k1_s1101
                                 - 0.319000121279186 z_k1_s1102
                                 - 0.229000063448682 z_k1_s1103
                                 - 0.474999782105164 z_k1_s1104
                                 - 0.222000205421941 z_k1_s1105
                                 - 0.0980001309754724 z_k1_s1106
                                 - 0.373000909257878 z_k1_s1107
                                 - 0.133000332613909 z_k1_s1110
                                 - 0.31300054028679 z_k1_s1111
                                 - 0.0880000871956263 z_k1_s1112
                                 - 0.135999583574882 z_k1_s1113
                                 - 0.233999624974539 z_k1_s1114
                                 - 0.129000041332197 z_k1_s1115
                                 - 0.0899999748363993 z_k1_s1116
                                 - 0.166999583507423 z_k1_s1117
                                 - 0.559999068560683 z_k1_s1120
                                 - 0.337999781948647 z_k1_s1121
                                 - 0.731000281245877 z_k1_s1122
                                 - 0.504999505210492 z_k1_s1123
                                 - 0.669999926258653 z_k1_s1124
                                 - 1.26499572704635 z_k1_s1125
                                 - 1.25499633098195 z_k1_s1126
                                 - 0.445999606435344 z_k1_s1127
                                 - 1.07100061618853 z_k1_s1130
                                 - 0.527000358318191 z_k1_s1131
                                 - 0.41400044730071 z_k1_s1132
                                 - 0.821002093362869 z_k1_s1133
                                 - 1.35200011324513 z_k1_s1134
                                 - 0.897002801702438 z_k1_s1135
                                 - 0.275000111209076 z_k1_s1136
                                 - 0.430000352761552 z_k1_s1137
                                 - 0.555997338798553 z_k1_s1140
                                 - 0.209999102928505 z_k1_s1141
                                 - 0.235998962173109 z_k1_s1142
                                 - 0.484997428870268 z_k1_s1143
                                 - 0.484997428870268 z_k1_s1144
                                 - 0.260998990330709 z_k1_s1145
                                 - 0.245998973436149 z_k1_s1146
                                 - 0.354999549641421 z_k1_s1147
                                 - 0.107999956477402 z_k1_s1150
                                 - 0.210000444039122 z_k1_s1151
                                 - 0.236999518158943 z_k1_s1152
                                 - 0.140999963512154 z_k1_s1153
                                 - 0.105999987105583 z_k1_s1154
                                 - 0.31100028811527 z_k1_s1155
                                 - 0.167999769631598 z_k1_s1156
                                 - 0.182999905920053 z_k1_s1157
                                 - 0.316000107895372 z_k1_s1160
                                 - 0.34599941459426 z_k1_s1161
                                 - 0.121000048852342 z_k1_s1162
                                 - 0.283999921083173 z_k1_s1163
                                 - 0.269999715290328 z_k1_s1164
                                 - 0.186999913089974 z_k1_s1165
                                 - 0.214999332175607 z_k1_s1166
                                 - 0.245999078931151 z_k1_s1167
                                 - 0.803002467710898 z_k1_s1170
                                 - 0.210000597850038 z_k1_s1171
                                 - 0.719000674486494 z_k1_s1172
                                 - 0.491002366610207 z_k1_s1173
                                 - 0.55100062004766 z_k1_s1174
                                 - 0.199000505985342 z_k1_s1175
                                 - 0.277000707861013 z_k1_s1176
                                 - 0.282000621180967 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.924302 w_k1_ADM_s000
                                 + 0.888447 w_k1_AXP_s000
                                 + 1.04197 w_k1_DIS_s000
                                 + 1.03528 w_k1_FDX_s000
                                 + 0.703252 w_k1_IBM_s000
                                 + 0.935017 w_k1_INTC_s000
                                 + 0.718617 w_k1_KO_s000
                                 + 0.72907 w_k1_MAT_s000
                                 + 1.08238 w_k1_MSFT_s000
                                 + 0.82751 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.18597 w_k1_ADM_s000
                                 + 1.07662 w_k1_AXP_s000
                                 + 1.18599 w_k1_DIS_s000 + 1.1111 w_k1_FDX_s000
                                 + 1.12045 w_k1_IBM_s000
                                 + 1.03491 w_k1_INTC_s000
                                 + 0.945823 w_k1_KO_s000
                                 + 1.05398 w_k1_MAT_s000
                                 + 1.10401 w_k1_MSFT_s000
                                 + 0.968136 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.773857 w_k1_ADM_s000
                                 + 0.523124 w_k1_AXP_s000
                                 + 0.727855 w_k1_DIS_s000
                                 + 0.772236 w_k1_FDX_s000
                                 + 0.880502 w_k1_IBM_s000
                                 + 0.983622 w_k1_INTC_s000
                                 + 0.792582 w_k1_KO_s000
                                 + 0.810204 w_k1_MAT_s000
                                 + 0.957478 w_k1_MSFT_s000
                                 + 0.862892 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.956812 w_k1_ADM_s000
                                 + 0.617799 w_k1_AXP_s000
                                 + 0.702695 w_k1_DIS_s000
                                 + 0.913414 w_k1_FDX_s000
                                 + 0.581539 w_k1_IBM_s000
                                 + 0.660977 w_k1_INTC_s000
                                 + 0.573778 w_k1_KO_s000
                                 + 0.422945 w_k1_MAT_s000
                                 + 0.950981 w_k1_MSFT_s000
                                 + 0.783603 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.32606 w_k1_ADM_s000
                                 + 1.36055 w_k1_AXP_s000
                                 + 1.56759 w_k1_DIS_s000
                                 + 1.43071 w_k1_FDX_s000
                                 + 0.955588 w_k1_IBM_s000
                                 + 1.23474 w_k1_INTC_s000
                                 + 1.13727 w_k1_KO_s000 + 1.36634 w_k1_MAT_s000
                                 + 1.47577 w_k1_MSFT_s000
                                 + 1.34898 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.957632 w_k1_ADM_s000
                                 + 1.18589 w_k1_AXP_s000
                                 + 1.02582 w_k1_DIS_s000
                                 + 1.14986 w_k1_FDX_s000
                                 + 0.913528 w_k1_IBM_s000
                                 + 1.20957 w_k1_INTC_s000
                                 + 0.777909 w_k1_KO_s000
                                 + 0.917373 w_k1_MAT_s000
                                 + 1.06355 w_k1_MSFT_s000
                                 + 1.06124 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.22198 w_k1_ADM_s000
                                 + 1.07526 w_k1_AXP_s000
                                 + 1.52986 w_k1_DIS_s000
                                 + 1.25106 w_k1_FDX_s000
                                 + 1.71469 w_k1_IBM_s000
                                 + 1.1645 w_k1_INTC_s000 + 1.30232 w_k1_KO_s000
                                 + 2.02939 w_k1_MAT_s000
                                 + 1.30468 w_k1_MSFT_s000 + 1.1456 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.840518 w_k1_ADM_s000
                                 + 1.05573 w_k1_AXP_s000
                                 + 1.10059 w_k1_DIS_s000
                                 + 1.01547 w_k1_FDX_s000
                                 + 1.47305 w_k1_IBM_s000
                                 + 0.960357 w_k1_INTC_s000
                                 + 0.87985 w_k1_KO_s000 + 1.36049 w_k1_MAT_s000
                                 + 1.07275 w_k1_MSFT_s000
                                 + 1.00876 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.50802 w_k1_ADM_s001
                                 + 1.30838 w_k1_AXP_s001
                                 + 1.15283 w_k1_DIS_s001
                                 + 1.14704 w_k1_FDX_s001
                                 + 0.733183 w_k1_IBM_s001
                                 + 0.959981 w_k1_INTC_s001
                                 + 0.922897 w_k1_KO_s001
                                 + 0.647007 w_k1_MAT_s001
                                 + 1.05314 w_k1_MSFT_s001
                                 + 1.11039 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.05128 w_k1_ADM_s001
                                 + 0.943143 w_k1_AXP_s001
                                 + 0.73515 w_k1_DIS_s001
                                 + 0.880649 w_k1_FDX_s001
                                 + 0.584043 w_k1_IBM_s001
                                 + 0.839157 w_k1_INTC_s001
                                 + 0.762315 w_k1_KO_s001
                                 + 0.771259 w_k1_MAT_s001
                                 + 1.04752 w_k1_MSFT_s001
                                 + 0.95004 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.09214 w_k1_ADM_s001
                                 + 1.30858 w_k1_AXP_s001
                                 + 1.29087 w_k1_DIS_s001
                                 + 1.17993 w_k1_FDX_s001
                                 + 1.11288 w_k1_IBM_s001
                                 + 1.09542 w_k1_INTC_s001 + 1.2998 w_k1_KO_s001
                                 + 0.851661 w_k1_MAT_s001
                                 + 1.26542 w_k1_MSFT_s001
                                 + 1.20092 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.07533 w_k1_ADM_s001
                                 + 1.38399 w_k1_AXP_s001
                                 + 1.16628 w_k1_DIS_s001
                                 + 0.974914 w_k1_FDX_s001
                                 + 1.58408 w_k1_IBM_s001
                                 + 1.11039 w_k1_INTC_s001
                                 + 1.21112 w_k1_KO_s001 + 1.87061 w_k1_MAT_s001
                                 + 1.03113 w_k1_MSFT_s001
                                 + 1.11728 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.734384 w_k1_ADM_s001
                                 + 0.582525 w_k1_AXP_s001
                                 + 0.701325 w_k1_DIS_s001
                                 + 0.718701 w_k1_FDX_s001
                                 + 0.688588 w_k1_IBM_s001
                                 + 0.681804 w_k1_INTC_s001
                                 + 0.723024 w_k1_KO_s001
                                 + 0.598805 w_k1_MAT_s001
                                 + 0.888303 w_k1_MSFT_s001
                                 + 0.892377 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.86982 w_k1_ADM_s001
                                 + 2.39116 w_k1_AXP_s001 + 1.3408 w_k1_DIS_s001
                                 + 1.30876 w_k1_FDX_s001 + 1.2921 w_k1_IBM_s001
                                 + 1.36037 w_k1_INTC_s001 + 1.3025 w_k1_KO_s001
                                 + 1.94767 w_k1_MAT_s001
                                 + 1.29977 w_k1_MSFT_s001
                                 + 1.34488 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.09336 w_k1_ADM_s001
                                 + 1.03687 w_k1_AXP_s001
                                 + 0.732324 w_k1_DIS_s001
                                 + 0.808737 w_k1_FDX_s001
                                 + 0.927903 w_k1_IBM_s001
                                 + 1.03204 w_k1_INTC_s001
                                 + 0.745615 w_k1_KO_s001
                                 + 0.734129 w_k1_MAT_s001
                                 + 0.899251 w_k1_MSFT_s001
                                 + 0.951623 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.921031 w_k1_ADM_s001
                                 + 0.847766 w_k1_AXP_s001
                                 + 0.807177 w_k1_DIS_s001
                                 + 0.748308 w_k1_FDX_s001
                                 + 1.07022 w_k1_IBM_s001
                                 + 0.861294 w_k1_INTC_s001
                                 + 0.947769 w_k1_KO_s001
                                 + 1.11585 w_k1_MAT_s001
                                 + 0.939909 w_k1_MSFT_s001
                                 + 1.03169 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.05295 w_k1_ADM_s002
                                 + 0.860268 w_k1_AXP_s002
                                 + 0.826781 w_k1_DIS_s002
                                 + 0.947926 w_k1_FDX_s002
                                 + 0.738565 w_k1_IBM_s002
                                 + 1.16642 w_k1_INTC_s002
                                 + 0.793531 w_k1_KO_s002
                                 + 0.618904 w_k1_MAT_s002
                                 + 1.06206 w_k1_MSFT_s002
                                 + 0.988333 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.13218 w_k1_ADM_s002
                                 + 1.27074 w_k1_AXP_s002
                                 + 1.42789 w_k1_DIS_s002
                                 + 1.17094 w_k1_FDX_s002
                                 + 1.27418 w_k1_IBM_s002
                                 + 1.24009 w_k1_INTC_s002 + 1.265 w_k1_KO_s002
                                 + 1.32012 w_k1_MAT_s002
                                 + 1.24949 w_k1_MSFT_s002 + 1.1475 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.28333 w_k1_ADM_s002
                                 + 1.16525 w_k1_AXP_s002
                                 + 1.21269 w_k1_DIS_s002
                                 + 1.16887 w_k1_FDX_s002
                                 + 0.650784 w_k1_IBM_s002
                                 + 1.34222 w_k1_INTC_s002
                                 + 0.853041 w_k1_KO_s002
                                 + 0.661131 w_k1_MAT_s002
                                 + 1.11293 w_k1_MSFT_s002
                                 + 0.981072 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.780199 w_k1_ADM_s002
                                 + 0.925135 w_k1_AXP_s002
                                 + 0.684218 w_k1_DIS_s002
                                 + 0.767666 w_k1_FDX_s002
                                 + 0.903679 w_k1_IBM_s002
                                 + 0.95843 w_k1_INTC_s002
                                 + 0.714087 w_k1_KO_s002
                                 + 0.955092 w_k1_MAT_s002
                                 + 0.842214 w_k1_MSFT_s002
                                 + 0.993296 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.12977 w_k1_ADM_s002
                                 + 0.800393 w_k1_AXP_s002
                                 + 0.801951 w_k1_DIS_s002
                                 + 0.876006 w_k1_FDX_s002
                                 + 0.589112 w_k1_IBM_s002
                                 + 0.839777 w_k1_INTC_s002
                                 + 0.738583 w_k1_KO_s002
                                 + 0.593733 w_k1_MAT_s002
                                 + 0.976025 w_k1_MSFT_s002
                                 + 0.961675 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.62646 w_k1_ADM_s002
                                 + 1.11686 w_k1_AXP_s002
                                 + 1.01074 w_k1_DIS_s002
                                 + 1.18983 w_k1_FDX_s002
                                 + 0.817771 w_k1_IBM_s002
                                 + 1.1748 w_k1_INTC_s002 + 0.89796 w_k1_KO_s002
                                 + 0.645246 w_k1_MAT_s002
                                 + 1.07601 w_k1_MSFT_s002 + 1.0164 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.835131 w_k1_ADM_s002
                                 + 0.94377 w_k1_AXP_s002
                                 + 0.876311 w_k1_DIS_s002
                                 + 0.895989 w_k1_FDX_s002
                                 + 1.10785 w_k1_IBM_s002
                                 + 0.978534 w_k1_INTC_s002
                                 + 0.991938 w_k1_KO_s002
                                 + 1.03837 w_k1_MAT_s002
                                 + 0.950341 w_k1_MSFT_s002
                                 + 1.09538 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.11771 w_k1_ADM_s002
                                 + 1.42069 w_k1_AXP_s002
                                 + 1.23012 w_k1_DIS_s002 + 1.1388 w_k1_FDX_s002
                                 + 1.06258 w_k1_IBM_s002
                                 + 1.24113 w_k1_INTC_s002
                                 + 1.09295 w_k1_KO_s002 + 1.33517 w_k1_MAT_s002
                                 + 1.27573 w_k1_MSFT_s002
                                 + 1.15939 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.0947 w_k1_ADM_s003
                                 + 0.908847 w_k1_AXP_s003
                                 + 0.781408 w_k1_DIS_s003
                                 + 0.956839 w_k1_FDX_s003
                                 + 0.619332 w_k1_IBM_s003
                                 + 1.13442 w_k1_INTC_s003
                                 + 0.769952 w_k1_KO_s003
                                 + 0.566089 w_k1_MAT_s003
                                 + 0.894797 w_k1_MSFT_s003
                                 + 0.862027 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.32793 w_k1_ADM_s003
                                 + 1.17869 w_k1_AXP_s003
                                 + 1.07085 w_k1_DIS_s003
                                 + 1.23793 w_k1_FDX_s003
                                 + 1.03138 w_k1_IBM_s003
                                 + 1.4376 w_k1_INTC_s003 + 1.04272 w_k1_KO_s003
                                 + 1.03594 w_k1_MAT_s003
                                 + 1.19392 w_k1_MSFT_s003
                                 + 1.16081 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.06567 w_k1_ADM_s003
                                 + 0.985507 w_k1_AXP_s003
                                 + 1.08775 w_k1_DIS_s003
                                 + 1.07343 w_k1_FDX_s003
                                 + 1.40174 w_k1_IBM_s003
                                 + 1.1779 w_k1_INTC_s003 + 1.1227 w_k1_KO_s003
                                 + 1.34342 w_k1_MAT_s003
                                 + 1.08337 w_k1_MSFT_s003
                                 + 1.07429 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.41612 w_k1_ADM_s003
                                 + 1.21394 w_k1_AXP_s003
                                 + 1.47821 w_k1_DIS_s003
                                 + 1.29221 w_k1_FDX_s003
                                 + 1.06935 w_k1_IBM_s003
                                 + 1.07636 w_k1_INTC_s003
                                 + 1.14599 w_k1_KO_s003 + 1.0734 w_k1_MAT_s003
                                 + 1.22757 w_k1_MSFT_s003
                                 + 1.20541 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.64935 w_k1_ADM_s003
                                 + 1.48741 w_k1_AXP_s003
                                 + 1.60582 w_k1_DIS_s003
                                 + 1.42745 w_k1_FDX_s003
                                 + 1.53356 w_k1_IBM_s003
                                 + 1.75571 w_k1_INTC_s003
                                 + 1.53334 w_k1_KO_s003 + 1.36584 w_k1_MAT_s003
                                 + 1.22946 w_k1_MSFT_s003
                                 + 1.37175 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.14018 w_k1_ADM_s003
                                 + 1.05482 w_k1_AXP_s003
                                 + 1.33704 w_k1_DIS_s003 + 1.1014 w_k1_FDX_s003
                                 + 1.30171 w_k1_IBM_s003
                                 + 1.11168 w_k1_INTC_s003
                                 + 1.30766 w_k1_KO_s003 + 1.28363 w_k1_MAT_s003
                                 + 1.09271 w_k1_MSFT_s003
                                 + 1.21887 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.06107 w_k1_ADM_s003
                                 + 0.839017 w_k1_AXP_s003
                                 + 0.8747 w_k1_DIS_s003
                                 + 0.889179 w_k1_FDX_s003
                                 + 0.90516 w_k1_IBM_s003
                                 + 0.992691 w_k1_INTC_s003
                                 + 1.06442 w_k1_KO_s003
                                 + 0.888797 w_k1_MAT_s003
                                 + 0.994084 w_k1_MSFT_s003
                                 + 1.04632 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.12734 w_k1_ADM_s003
                                 + 1.1878 w_k1_AXP_s003
                                 + 0.694497 w_k1_DIS_s003
                                 + 0.94648 w_k1_FDX_s003
                                 + 0.985587 w_k1_IBM_s003
                                 + 1.02738 w_k1_INTC_s003
                                 + 0.867652 w_k1_KO_s003 + 1.0806 w_k1_MAT_s003
                                 + 0.860836 w_k1_MSFT_s003
                                 + 1.16558 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.965664 w_k1_ADM_s004
                                 + 1.14246 w_k1_AXP_s004
                                 + 1.40764 w_k1_DIS_s004
                                 + 1.02834 w_k1_FDX_s004
                                 + 1.45616 w_k1_IBM_s004
                                 + 1.17827 w_k1_INTC_s004
                                 + 1.36305 w_k1_KO_s004 + 1.15902 w_k1_MAT_s004
                                 + 1.20229 w_k1_MSFT_s004
                                 + 1.17653 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.942501 w_k1_ADM_s004
                                 + 0.779314 w_k1_AXP_s004
                                 + 0.950744 w_k1_DIS_s004
                                 + 0.842427 w_k1_FDX_s004
                                 + 0.962031 w_k1_IBM_s004
                                 + 1.06403 w_k1_INTC_s004
                                 + 1.06611 w_k1_KO_s004 + 1.01111 w_k1_MAT_s004
                                 + 1.19845 w_k1_MSFT_s004
                                 + 1.00216 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.75614 w_k1_ADM_s004
                                 + 0.729029 w_k1_AXP_s004
                                 + 0.732341 w_k1_DIS_s004
                                 + 0.697735 w_k1_FDX_s004
                                 + 1.02669 w_k1_IBM_s004
                                 + 0.672279 w_k1_INTC_s004
                                 + 0.923906 w_k1_KO_s004
                                 + 1.17953 w_k1_MAT_s004
                                 + 0.879162 w_k1_MSFT_s004
                                 + 0.96876 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.26272 w_k1_ADM_s004
                                 + 1.54081 w_k1_AXP_s004
                                 + 1.46769 w_k1_DIS_s004
                                 + 1.36621 w_k1_FDX_s004
                                 + 1.27474 w_k1_IBM_s004
                                 + 1.00723 w_k1_INTC_s004
                                 + 1.17785 w_k1_KO_s004 + 1.68624 w_k1_MAT_s004
                                 + 1.31767 w_k1_MSFT_s004
                                 + 1.33193 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.983043 w_k1_ADM_s004
                                 + 1.06036 w_k1_AXP_s004
                                 + 0.864026 w_k1_DIS_s004
                                 + 1.02591 w_k1_FDX_s004
                                 + 0.69646 w_k1_IBM_s004
                                 + 0.957374 w_k1_INTC_s004
                                 + 0.937696 w_k1_KO_s004
                                 + 0.864429 w_k1_MAT_s004
                                 + 1.07041 w_k1_MSFT_s004
                                 + 0.96024 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.927351 w_k1_ADM_s004
                                 + 1.17959 w_k1_AXP_s004
                                 + 0.948431 w_k1_DIS_s004
                                 + 0.978757 w_k1_FDX_s004
                                 + 1.08231 w_k1_IBM_s004
                                 + 0.910896 w_k1_INTC_s004
                                 + 1.13956 w_k1_KO_s004 + 1.64139 w_k1_MAT_s004
                                 + 1.1426 w_k1_MSFT_s004 + 1.16619 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.774654 w_k1_ADM_s004
                                 + 0.912438 w_k1_AXP_s004
                                 + 1.25377 w_k1_DIS_s004
                                 + 1.03037 w_k1_FDX_s004
                                 + 0.895052 w_k1_IBM_s004
                                 + 1.01745 w_k1_INTC_s004
                                 + 1.06206 w_k1_KO_s004 + 1.16752 w_k1_MAT_s004
                                 + 1.30125 w_k1_MSFT_s004
                                 + 1.19341 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.25214 w_k1_ADM_s004
                                 + 1.01484 w_k1_AXP_s004
                                 + 1.25477 w_k1_DIS_s004
                                 + 1.16713 w_k1_FDX_s004
                                 + 0.786254 w_k1_IBM_s004
                                 + 0.996472 w_k1_INTC_s004
                                 + 1.02268 w_k1_KO_s004 + 1.01717 w_k1_MAT_s004
                                 + 1.15047 w_k1_MSFT_s004
                                 + 1.14024 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.934716 w_k1_ADM_s005
                                 + 0.846764 w_k1_AXP_s005
                                 + 1.07384 w_k1_DIS_s005 + 1.0607 w_k1_FDX_s005
                                 + 1.03551 w_k1_IBM_s005
                                 + 1.08752 w_k1_INTC_s005
                                 + 1.13509 w_k1_KO_s005 + 1.00026 w_k1_MAT_s005
                                 + 0.99406 w_k1_MSFT_s005
                                 + 1.22449 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.25388 w_k1_ADM_s005
                                 + 1.20501 w_k1_AXP_s005
                                 + 1.56878 w_k1_DIS_s005 + 1.4061 w_k1_FDX_s005
                                 + 1.50204 w_k1_IBM_s005
                                 + 1.38697 w_k1_INTC_s005
                                 + 1.58744 w_k1_KO_s005 + 1.86693 w_k1_MAT_s005
                                 + 1.49151 w_k1_MSFT_s005
                                 + 1.27804 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.801299 w_k1_ADM_s005
                                 + 0.970115 w_k1_AXP_s005
                                 + 0.889461 w_k1_DIS_s005
                                 + 0.913798 w_k1_FDX_s005
                                 + 1.19885 w_k1_IBM_s005
                                 + 1.08939 w_k1_INTC_s005
                                 + 1.04287 w_k1_KO_s005 + 1.22952 w_k1_MAT_s005
                                 + 1.01702 w_k1_MSFT_s005
                                 + 0.987697 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.979642 w_k1_ADM_s005
                                 + 1.07224 w_k1_AXP_s005
                                 + 1.17116 w_k1_DIS_s005
                                 + 1.21887 w_k1_FDX_s005
                                 + 0.674499 w_k1_IBM_s005
                                 + 1.0673 w_k1_INTC_s005 + 1.02114 w_k1_KO_s005
                                 + 1.02531 w_k1_MAT_s005
                                 + 1.14588 w_k1_MSFT_s005
                                 + 1.11015 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.919489 w_k1_ADM_s005
                                 + 1.11943 w_k1_AXP_s005
                                 + 1.31601 w_k1_DIS_s005
                                 + 0.970169 w_k1_FDX_s005
                                 + 1.43275 w_k1_IBM_s005
                                 + 1.24158 w_k1_INTC_s005
                                 + 1.40348 w_k1_KO_s005 + 1.6299 w_k1_MAT_s005
                                 + 1.12348 w_k1_MSFT_s005
                                 + 1.27747 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.06023 w_k1_ADM_s005
                                 + 0.988345 w_k1_AXP_s005
                                 + 1.39911 w_k1_DIS_s005
                                 + 1.22705 w_k1_FDX_s005
                                 + 0.810064 w_k1_IBM_s005
                                 + 1.08038 w_k1_INTC_s005
                                 + 1.21407 w_k1_KO_s005
                                 + 0.984601 w_k1_MAT_s005
                                 + 1.33556 w_k1_MSFT_s005
                                 + 1.19619 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.864031 w_k1_ADM_s005
                                 + 0.68935 w_k1_AXP_s005
                                 + 0.817067 w_k1_DIS_s005
                                 + 0.839145 w_k1_FDX_s005
                                 + 0.800319 w_k1_IBM_s005
                                 + 1.02835 w_k1_INTC_s005
                                 + 0.929661 w_k1_KO_s005
                                 + 0.925022 w_k1_MAT_s005
                                 + 0.988521 w_k1_MSFT_s005
                                 + 0.845325 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.858799 w_k1_ADM_s005
                                 + 1.04728 w_k1_AXP_s005
                                 + 1.01452 w_k1_DIS_s005 + 1.1339 w_k1_FDX_s005
                                 + 1.02566 w_k1_IBM_s005
                                 + 1.04174 w_k1_INTC_s005 + 1.0921 w_k1_KO_s005
                                 + 1.7317 w_k1_MAT_s005
                                 + 1.14845 w_k1_MSFT_s005
                                 + 1.10265 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.974278 w_k1_ADM_s006
                                 + 0.826014 w_k1_AXP_s006
                                 + 0.950268 w_k1_DIS_s006
                                 + 1.04064 w_k1_FDX_s006
                                 + 0.817006 w_k1_IBM_s006
                                 + 1.03334 w_k1_INTC_s006
                                 + 0.967045 w_k1_KO_s006
                                 + 0.719892 w_k1_MAT_s006
                                 + 1.22618 w_k1_MSFT_s006
                                 + 0.987295 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.2598 w_k1_ADM_s006
                                 + 1.45212 w_k1_AXP_s006
                                 + 1.53973 w_k1_DIS_s006
                                 + 1.46762 w_k1_FDX_s006
                                 + 1.16658 w_k1_IBM_s006
                                 + 1.25792 w_k1_INTC_s006
                                 + 1.20141 w_k1_KO_s006 + 1.45084 w_k1_MAT_s006
                                 + 1.45914 w_k1_MSFT_s006
                                 + 1.14224 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.878692 w_k1_ADM_s006
                                 + 0.798225 w_k1_AXP_s006
                                 + 0.719012 w_k1_DIS_s006
                                 + 0.871559 w_k1_FDX_s006
                                 + 0.71844 w_k1_IBM_s006
                                 + 0.843452 w_k1_INTC_s006
                                 + 0.759287 w_k1_KO_s006
                                 + 0.849574 w_k1_MAT_s006
                                 + 0.882821 w_k1_MSFT_s006
                                 + 0.884344 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.15803 w_k1_ADM_s006
                                 + 0.998604 w_k1_AXP_s006
                                 + 1.24341 w_k1_DIS_s006
                                 + 1.06504 w_k1_FDX_s006
                                 + 1.22335 w_k1_IBM_s006
                                 + 1.0206 w_k1_INTC_s006 + 1.26777 w_k1_KO_s006
                                 + 1.53877 w_k1_MAT_s006
                                 + 1.26766 w_k1_MSFT_s006
                                 + 1.16854 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.905364 w_k1_ADM_s006
                                 + 0.785424 w_k1_AXP_s006
                                 + 1.00688 w_k1_DIS_s006
                                 + 0.85532 w_k1_FDX_s006
                                 + 1.10195 w_k1_IBM_s006
                                 + 0.920156 w_k1_INTC_s006
                                 + 1.00792 w_k1_KO_s006
                                 + 0.861722 w_k1_MAT_s006
                                 + 1.02954 w_k1_MSFT_s006
                                 + 0.970062 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.733173 w_k1_ADM_s006
                                 + 1.00881 w_k1_AXP_s006
                                 + 0.919883 w_k1_DIS_s006
                                 + 0.910225 w_k1_FDX_s006
                                 + 1.28128 w_k1_IBM_s006
                                 + 1.02558 w_k1_INTC_s006
                                 + 1.01432 w_k1_KO_s006 + 1.33821 w_k1_MAT_s006
                                 + 1.01675 w_k1_MSFT_s006
                                 + 1.00281 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.13263 w_k1_ADM_s006
                                 + 0.867778 w_k1_AXP_s006
                                 + 1.39443 w_k1_DIS_s006 + 1.165 w_k1_FDX_s006
                                 + 0.969928 w_k1_IBM_s006
                                 + 1.2363 w_k1_INTC_s006 + 1.14098 w_k1_KO_s006
                                 + 0.470971 w_k1_MAT_s006
                                 + 1.27939 w_k1_MSFT_s006
                                 + 1.01434 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.14273 w_k1_ADM_s006
                                 + 1.19408 w_k1_AXP_s006
                                 + 1.02096 w_k1_DIS_s006
                                 + 1.28467 w_k1_FDX_s006
                                 + 0.706041 w_k1_IBM_s006
                                 + 1.24279 w_k1_INTC_s006
                                 + 0.843667 w_k1_KO_s006
                                 + 0.620141 w_k1_MAT_s006
                                 + 1.13254 w_k1_MSFT_s006 + 1.0281 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.075 w_k1_ADM_s007
                                 + 0.847889 w_k1_AXP_s007
                                 + 0.836152 w_k1_DIS_s007
                                 + 0.967829 w_k1_FDX_s007
                                 + 0.734326 w_k1_IBM_s007
                                 + 0.906765 w_k1_INTC_s007
                                 + 0.676867 w_k1_KO_s007
                                 + 0.616673 w_k1_MAT_s007
                                 + 0.835435 w_k1_MSFT_s007
                                 + 0.892363 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.3626 w_k1_ADM_s007
                                 + 1.43736 w_k1_AXP_s007
                                 + 1.52479 w_k1_DIS_s007
                                 + 1.28259 w_k1_FDX_s007
                                 + 0.927394 w_k1_IBM_s007
                                 + 1.30921 w_k1_INTC_s007
                                 + 1.22853 w_k1_KO_s007
                                 + 0.975749 w_k1_MAT_s007
                                 + 1.33307 w_k1_MSFT_s007
                                 + 1.23854 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.09907 w_k1_ADM_s007
                                 + 1.14502 w_k1_AXP_s007
                                 + 1.20054 w_k1_DIS_s007
                                 + 1.04833 w_k1_FDX_s007
                                 + 1.49947 w_k1_IBM_s007
                                 + 1.3449 w_k1_INTC_s007 + 1.35358 w_k1_KO_s007
                                 + 1.37961 w_k1_MAT_s007
                                 + 1.06466 w_k1_MSFT_s007 + 1.0674 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.08071 w_k1_ADM_s007
                                 + 1.0807 w_k1_AXP_s007 + 1.1417 w_k1_DIS_s007
                                 + 1.16861 w_k1_FDX_s007
                                 + 0.917172 w_k1_IBM_s007
                                 + 1.16393 w_k1_INTC_s007
                                 + 1.02312 w_k1_KO_s007
                                 + 0.918482 w_k1_MAT_s007
                                 + 1.118 w_k1_MSFT_s007 + 1.05697 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.27614 w_k1_ADM_s007
                                 + 1.55732 w_k1_AXP_s007
                                 + 1.59861 w_k1_DIS_s007
                                 + 1.27012 w_k1_FDX_s007
                                 + 1.45265 w_k1_IBM_s007
                                 + 1.60646 w_k1_INTC_s007
                                 + 1.38654 w_k1_KO_s007 + 1.32198 w_k1_MAT_s007
                                 + 1.31661 w_k1_MSFT_s007
                                 + 1.36912 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.903023 w_k1_ADM_s007
                                 + 0.969263 w_k1_AXP_s007
                                 + 0.938234 w_k1_DIS_s007
                                 + 0.829109 w_k1_FDX_s007
                                 + 1.64485 w_k1_IBM_s007
                                 + 0.917567 w_k1_INTC_s007
                                 + 1.12078 w_k1_KO_s007
                                 + 0.763376 w_k1_MAT_s007
                                 + 0.740577 w_k1_MSFT_s007
                                 + 0.957873 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.03987 w_k1_ADM_s007
                                 + 0.812199 w_k1_AXP_s007
                                 + 0.996306 w_k1_DIS_s007
                                 + 0.833953 w_k1_FDX_s007
                                 + 1.11265 w_k1_IBM_s007
                                 + 0.941317 w_k1_INTC_s007
                                 + 1.06229 w_k1_KO_s007 + 1.13525 w_k1_MAT_s007
                                 + 0.828219 w_k1_MSFT_s007
                                 + 1.07931 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.930345 w_k1_ADM_s007
                                 + 0.90538 w_k1_AXP_s007
                                 + 0.713387 w_k1_DIS_s007
                                 + 0.811062 w_k1_FDX_s007
                                 + 1.03454 w_k1_IBM_s007
                                 + 0.891482 w_k1_INTC_s007
                                 + 0.857172 w_k1_KO_s007
                                 + 0.773141 w_k1_MAT_s007
                                 + 0.733094 w_k1_MSFT_s007
                                 + 0.937813 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.888595 w_k1_ADM_s010
                                 + 0.805288 w_k1_AXP_s010
                                 + 0.911369 w_k1_DIS_s010
                                 + 0.924239 w_k1_FDX_s010
                                 + 0.914141 w_k1_IBM_s010
                                 + 0.988884 w_k1_INTC_s010
                                 + 0.987003 w_k1_KO_s010
                                 + 0.974232 w_k1_MAT_s010
                                 + 1.02838 w_k1_MSFT_s010
                                 + 1.00754 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.06061 w_k1_ADM_s010
                                 + 0.991501 w_k1_AXP_s010
                                 + 0.977845 w_k1_DIS_s010
                                 + 1.17059 w_k1_FDX_s010
                                 + 0.646357 w_k1_IBM_s010
                                 + 1.28395 w_k1_INTC_s010
                                 + 0.781963 w_k1_KO_s010
                                 + 0.60923 w_k1_MAT_s010
                                 + 1.01033 w_k1_MSFT_s010
                                 + 1.07028 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.927435 w_k1_ADM_s010
                                 + 1.35579 w_k1_AXP_s010
                                 + 1.56654 w_k1_DIS_s010
                                 + 1.13522 w_k1_FDX_s010
                                 + 1.54555 w_k1_IBM_s010
                                 + 1.34697 w_k1_INTC_s010
                                 + 1.46543 w_k1_KO_s010 + 2.16834 w_k1_MAT_s010
                                 + 1.28241 w_k1_MSFT_s010
                                 + 1.29946 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.755162 w_k1_ADM_s010
                                 + 1.01488 w_k1_AXP_s010
                                 + 0.838744 w_k1_DIS_s010
                                 + 0.872409 w_k1_FDX_s010
                                 + 1.12901 w_k1_IBM_s010
                                 + 1.07253 w_k1_INTC_s010 + 1.1055 w_k1_KO_s010
                                 + 1.44523 w_k1_MAT_s010
                                 + 0.9538 w_k1_MSFT_s010 + 1.0403 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.86297 w_k1_ADM_s010
                                 + 1.17191 w_k1_AXP_s010
                                 + 1.06105 w_k1_DIS_s010
                                 + 1.06531 w_k1_FDX_s010
                                 + 1.48405 w_k1_IBM_s010
                                 + 1.15361 w_k1_INTC_s010
                                 + 1.10279 w_k1_KO_s010 + 2.03078 w_k1_MAT_s010
                                 + 0.981473 w_k1_MSFT_s010
                                 + 1.29123 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.914349 w_k1_ADM_s010
                                 + 1.26635 w_k1_AXP_s010
                                 + 0.899354 w_k1_DIS_s010
                                 + 1.12629 w_k1_FDX_s010
                                 + 0.889754 w_k1_IBM_s010
                                 + 1.31307 w_k1_INTC_s010
                                 + 0.950001 w_k1_KO_s010
                                 + 0.909125 w_k1_MAT_s010
                                 + 0.998503 w_k1_MSFT_s010
                                 + 1.04752 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.653202 w_k1_ADM_s010
                                 + 0.756632 w_k1_AXP_s010
                                 + 0.599045 w_k1_DIS_s010
                                 + 0.764971 w_k1_FDX_s010
                                 + 0.744671 w_k1_IBM_s010
                                 + 1.01507 w_k1_INTC_s010
                                 + 0.692124 w_k1_KO_s010
                                 + 0.805432 w_k1_MAT_s010
                                 + 0.72437 w_k1_MSFT_s010
                                 + 0.763118 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.10415 w_k1_ADM_s010
                                 + 1.32548 w_k1_AXP_s010
                                 + 1.48691 w_k1_DIS_s010 + 1.3978 w_k1_FDX_s010
                                 + 1.03628 w_k1_IBM_s010
                                 + 1.09541 w_k1_INTC_s010
                                 + 1.30315 w_k1_KO_s010 + 1.39836 w_k1_MAT_s010
                                 + 1.34013 w_k1_MSFT_s010
                                 + 1.22749 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.22747 w_k1_ADM_s011
                                 + 1.18258 w_k1_AXP_s011
                                 + 1.10564 w_k1_DIS_s011
                                 + 1.20651 w_k1_FDX_s011
                                 + 1.04945 w_k1_IBM_s011
                                 + 1.1166 w_k1_INTC_s011 + 1.05836 w_k1_KO_s011
                                 + 0.916018 w_k1_MAT_s011
                                 + 1.08053 w_k1_MSFT_s011 + 1.0123 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.704778 w_k1_ADM_s011
                                 + 0.782039 w_k1_AXP_s011
                                 + 0.657326 w_k1_DIS_s011
                                 + 0.790506 w_k1_FDX_s011
                                 + 0.803131 w_k1_IBM_s011
                                 + 0.798784 w_k1_INTC_s011
                                 + 0.725267 w_k1_KO_s011
                                 + 0.89249 w_k1_MAT_s011
                                 + 0.80901 w_k1_MSFT_s011
                                 + 0.740755 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.18468 w_k1_ADM_s011
                                 + 1.20634 w_k1_AXP_s011
                                 + 1.41825 w_k1_DIS_s011 + 1.1859 w_k1_FDX_s011
                                 + 1.38424 w_k1_IBM_s011
                                 + 0.983025 w_k1_INTC_s011
                                 + 1.22295 w_k1_KO_s011 + 1.31419 w_k1_MAT_s011
                                 + 1.18541 w_k1_MSFT_s011
                                 + 1.06351 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.739754 w_k1_ADM_s011
                                 + 1.2588 w_k1_AXP_s011 + 0.91446 w_k1_DIS_s011
                                 + 0.944296 w_k1_FDX_s011
                                 + 1.05687 w_k1_IBM_s011
                                 + 1.06778 w_k1_INTC_s011
                                 + 0.929473 w_k1_KO_s011
                                 + 1.13689 w_k1_MAT_s011
                                 + 0.941845 w_k1_MSFT_s011
                                 + 0.972616 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.794878 w_k1_ADM_s011
                                 + 0.90537 w_k1_AXP_s011
                                 + 0.858579 w_k1_DIS_s011
                                 + 0.853803 w_k1_FDX_s011
                                 + 1.26701 w_k1_IBM_s011
                                 + 1.11558 w_k1_INTC_s011
                                 + 1.06176 w_k1_KO_s011
                                 + 0.910923 w_k1_MAT_s011
                                 + 0.856748 w_k1_MSFT_s011
                                 + 0.933221 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01524 w_k1_ADM_s011
                                 + 0.919413 w_k1_AXP_s011
                                 + 1.1228 w_k1_DIS_s011 + 1.08169 w_k1_FDX_s011
                                 + 0.857229 w_k1_IBM_s011
                                 + 0.941497 w_k1_INTC_s011
                                 + 0.953798 w_k1_KO_s011
                                 + 0.921892 w_k1_MAT_s011
                                 + 1.11843 w_k1_MSFT_s011
                                 + 0.989481 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.22616 w_k1_ADM_s011
                                 + 1.67591 w_k1_AXP_s011
                                 + 1.24364 w_k1_DIS_s011
                                 + 1.27507 w_k1_FDX_s011
                                 + 1.52963 w_k1_IBM_s011
                                 + 1.23248 w_k1_INTC_s011
                                 + 1.43654 w_k1_KO_s011 + 1.99697 w_k1_MAT_s011
                                 + 1.26838 w_k1_MSFT_s011 + 1.2265 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.872325 w_k1_ADM_s011
                                 + 0.979086 w_k1_AXP_s011
                                 + 1.01177 w_k1_DIS_s011
                                 + 1.12144 w_k1_FDX_s011
                                 + 0.778623 w_k1_IBM_s011
                                 + 1.04385 w_k1_INTC_s011
                                 + 0.802707 w_k1_KO_s011
                                 + 0.521744 w_k1_MAT_s011
                                 + 1.007 w_k1_MSFT_s011 + 0.869028 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.37433 w_k1_ADM_s012
                                 + 1.57318 w_k1_AXP_s012
                                 + 1.67556 w_k1_DIS_s012
                                 + 1.39135 w_k1_FDX_s012
                                 + 0.933675 w_k1_IBM_s012
                                 + 1.04362 w_k1_INTC_s012
                                 + 1.19224 w_k1_KO_s012 + 1.01165 w_k1_MAT_s012
                                 + 1.41716 w_k1_MSFT_s012
                                 + 1.23026 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.894066 w_k1_ADM_s012
                                 + 1.0157 w_k1_AXP_s012 + 1.0634 w_k1_DIS_s012
                                 + 0.883759 w_k1_FDX_s012
                                 + 1.45768 w_k1_IBM_s012
                                 + 0.953565 w_k1_INTC_s012
                                 + 1.13859 w_k1_KO_s012 + 1.33017 w_k1_MAT_s012
                                 + 0.903046 w_k1_MSFT_s012
                                 + 1.05917 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.999658 w_k1_ADM_s012
                                 + 0.8858 w_k1_AXP_s012
                                 + 0.826383 w_k1_DIS_s012
                                 + 0.877193 w_k1_FDX_s012
                                 + 1.12145 w_k1_IBM_s012
                                 + 1.03678 w_k1_INTC_s012
                                 + 1.00285 w_k1_KO_s012 + 1.11525 w_k1_MAT_s012
                                 + 0.945508 w_k1_MSFT_s012
                                 + 0.94051 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.0289 w_k1_ADM_s012
                                 + 0.603981 w_k1_AXP_s012
                                 + 0.897048 w_k1_DIS_s012
                                 + 0.821259 w_k1_FDX_s012
                                 + 0.825605 w_k1_IBM_s012
                                 + 0.754127 w_k1_INTC_s012
                                 + 0.926123 w_k1_KO_s012
                                 + 0.577342 w_k1_MAT_s012
                                 + 0.982282 w_k1_MSFT_s012
                                 + 0.956086 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.803646 w_k1_ADM_s012
                                 + 1.00653 w_k1_AXP_s012
                                 + 0.803175 w_k1_DIS_s012
                                 + 0.709307 w_k1_FDX_s012
                                 + 1.69856 w_k1_IBM_s012
                                 + 0.832955 w_k1_INTC_s012
                                 + 1.01811 w_k1_KO_s012 + 1.75438 w_k1_MAT_s012
                                 + 0.895202 w_k1_MSFT_s012
                                 + 0.915095 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.1648 w_k1_ADM_s012
                                 + 1.39736 w_k1_AXP_s012
                                 + 1.50431 w_k1_DIS_s012
                                 + 1.17732 w_k1_FDX_s012
                                 + 1.33305 w_k1_IBM_s012
                                 + 1.09091 w_k1_INTC_s012
                                 + 1.16277 w_k1_KO_s012 + 1.8357 w_k1_MAT_s012
                                 + 1.22262 w_k1_MSFT_s012 + 1.1707 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.931383 w_k1_ADM_s012
                                 + 1.00271 w_k1_AXP_s012
                                 + 0.781461 w_k1_DIS_s012
                                 + 0.952993 w_k1_FDX_s012
                                 + 0.657364 w_k1_IBM_s012
                                 + 0.910095 w_k1_INTC_s012
                                 + 0.680728 w_k1_KO_s012
                                 + 0.63431 w_k1_MAT_s012
                                 + 0.980119 w_k1_MSFT_s012
                                 + 0.848041 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.804963 w_k1_ADM_s012
                                 + 0.906864 w_k1_AXP_s012
                                 + 0.740568 w_k1_DIS_s012
                                 + 0.841475 w_k1_FDX_s012
                                 + 0.910645 w_k1_IBM_s012
                                 + 0.830303 w_k1_INTC_s012
                                 + 0.884141 w_k1_KO_s012
                                 + 1.08613 w_k1_MAT_s012
                                 + 0.924403 w_k1_MSFT_s012
                                 + 0.886349 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.834245 w_k1_ADM_s013
                                 + 0.695539 w_k1_AXP_s013
                                 + 0.728403 w_k1_DIS_s013
                                 + 0.8256 w_k1_FDX_s013
                                 + 0.740517 w_k1_IBM_s013
                                 + 0.949011 w_k1_INTC_s013
                                 + 0.801097 w_k1_KO_s013
                                 + 0.611797 w_k1_MAT_s013
                                 + 0.89394 w_k1_MSFT_s013
                                 + 0.898977 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.961787 w_k1_ADM_s013
                                 + 1.41465 w_k1_AXP_s013
                                 + 1.65301 w_k1_DIS_s013
                                 + 1.24361 w_k1_FDX_s013
                                 + 1.76262 w_k1_IBM_s013
                                 + 1.2505 w_k1_INTC_s013 + 1.48148 w_k1_KO_s013
                                 + 1.83859 w_k1_MAT_s013
                                 + 1.25064 w_k1_MSFT_s013
                                 + 1.04066 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.88269 w_k1_ADM_s013
                                 + 0.946123 w_k1_AXP_s013
                                 + 1.31223 w_k1_DIS_s013
                                 + 1.10056 w_k1_FDX_s013
                                 + 0.870086 w_k1_IBM_s013
                                 + 1.08981 w_k1_INTC_s013
                                 + 1.19734 w_k1_KO_s013
                                 + 0.801271 w_k1_MAT_s013
                                 + 1.01829 w_k1_MSFT_s013
                                 + 0.958851 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.36914 w_k1_ADM_s013
                                 + 1.57256 w_k1_AXP_s013
                                 + 1.46235 w_k1_DIS_s013
                                 + 1.35341 w_k1_FDX_s013
                                 + 1.21796 w_k1_IBM_s013
                                 + 1.31687 w_k1_INTC_s013
                                 + 1.43996 w_k1_KO_s013 + 1.40808 w_k1_MAT_s013
                                 + 1.29268 w_k1_MSFT_s013
                                 + 1.22021 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.885305 w_k1_ADM_s013
                                 + 0.972925 w_k1_AXP_s013
                                 + 1.07236 w_k1_DIS_s013
                                 + 0.931733 w_k1_FDX_s013
                                 + 1.29681 w_k1_IBM_s013
                                 + 1.07452 w_k1_INTC_s013
                                 + 1.21213 w_k1_KO_s013 + 1.09659 w_k1_MAT_s013
                                 + 0.98035 w_k1_MSFT_s013
                                 + 1.01275 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.69863 w_k1_ADM_s013
                                 + 0.844701 w_k1_AXP_s013
                                 + 0.812855 w_k1_DIS_s013
                                 + 0.816825 w_k1_FDX_s013
                                 + 0.923586 w_k1_IBM_s013
                                 + 0.901506 w_k1_INTC_s013
                                 + 0.915097 w_k1_KO_s013
                                 + 0.891246 w_k1_MAT_s013
                                 + 0.848359 w_k1_MSFT_s013
                                 + 0.843653 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.807416 w_k1_ADM_s013
                                 + 1.34511 w_k1_AXP_s013
                                 + 1.45213 w_k1_DIS_s013
                                 + 1.06695 w_k1_FDX_s013 + 1.4921 w_k1_IBM_s013
                                 + 1.14457 w_k1_INTC_s013
                                 + 1.33784 w_k1_KO_s013 + 1.50794 w_k1_MAT_s013
                                 + 1.08551 w_k1_MSFT_s013
                                 + 1.00916 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.09336 w_k1_ADM_s013
                                 + 1.40621 w_k1_AXP_s013
                                 + 1.08625 w_k1_DIS_s013
                                 + 1.12701 w_k1_FDX_s013
                                 + 0.997017 w_k1_IBM_s013
                                 + 1.20147 w_k1_INTC_s013
                                 + 1.05684 w_k1_KO_s013 + 1.19153 w_k1_MAT_s013
                                 + 1.0548 w_k1_MSFT_s013 + 1.02752 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.21781 w_k1_ADM_s014
                                 + 1.15306 w_k1_AXP_s014
                                 + 1.56158 w_k1_DIS_s014
                                 + 1.20459 w_k1_FDX_s014
                                 + 1.06299 w_k1_IBM_s014
                                 + 1.20037 w_k1_INTC_s014
                                 + 1.38194 w_k1_KO_s014 + 1.17811 w_k1_MAT_s014
                                 + 1.17714 w_k1_MSFT_s014
                                 + 1.13561 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.17796 w_k1_ADM_s014
                                 + 0.888454 w_k1_AXP_s014
                                 + 1.21717 w_k1_DIS_s014 + 1.1985 w_k1_FDX_s014
                                 + 0.696802 w_k1_IBM_s014
                                 + 1.11839 w_k1_INTC_s014
                                 + 0.919891 w_k1_KO_s014
                                 + 0.446975 w_k1_MAT_s014
                                 + 1.06861 w_k1_MSFT_s014
                                 + 1.03327 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.975973 w_k1_ADM_s014
                                 + 1.24163 w_k1_AXP_s014
                                 + 1.37574 w_k1_DIS_s014
                                 + 1.13605 w_k1_FDX_s014
                                 + 1.39947 w_k1_IBM_s014
                                 + 0.989855 w_k1_INTC_s014 + 1.212 w_k1_KO_s014
                                 + 1.43913 w_k1_MAT_s014
                                 + 0.962825 w_k1_MSFT_s014
                                 + 1.11031 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.873436 w_k1_ADM_s014
                                 + 0.877122 w_k1_AXP_s014
                                 + 0.822517 w_k1_DIS_s014
                                 + 0.862588 w_k1_FDX_s014
                                 + 1.18173 w_k1_IBM_s014
                                 + 0.72605 w_k1_INTC_s014
                                 + 1.08499 w_k1_KO_s014 + 1.44027 w_k1_MAT_s014
                                 + 0.864844 w_k1_MSFT_s014
                                 + 0.950072 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.902698 w_k1_ADM_s014
                                 + 0.781953 w_k1_AXP_s014
                                 + 0.858562 w_k1_DIS_s014
                                 + 0.898907 w_k1_FDX_s014
                                 + 0.718114 w_k1_IBM_s014
                                 + 0.846332 w_k1_INTC_s014
                                 + 0.871751 w_k1_KO_s014
                                 + 0.903263 w_k1_MAT_s014
                                 + 0.909228 w_k1_MSFT_s014
                                 + 0.838006 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.736064 w_k1_ADM_s014
                                 + 0.578252 w_k1_AXP_s014
                                 + 0.620192 w_k1_DIS_s014
                                 + 0.69949 w_k1_FDX_s014
                                 + 0.807632 w_k1_IBM_s014
                                 + 0.758155 w_k1_INTC_s014
                                 + 0.849596 w_k1_KO_s014
                                 + 0.984204 w_k1_MAT_s014
                                 + 0.711774 w_k1_MSFT_s014
                                 + 0.859634 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.993657 w_k1_ADM_s014
                                 + 0.855643 w_k1_AXP_s014
                                 + 1.12022 w_k1_DIS_s014 + 1.0193 w_k1_FDX_s014
                                 + 0.934338 w_k1_IBM_s014
                                 + 1.06424 w_k1_INTC_s014
                                 + 1.10573 w_k1_KO_s014 + 0.7188 w_k1_MAT_s014
                                 + 0.999188 w_k1_MSFT_s014
                                 + 1.04915 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.817991 w_k1_ADM_s014
                                 + 1.11127 w_k1_AXP_s014
                                 + 0.704304 w_k1_DIS_s014
                                 + 0.960094 w_k1_FDX_s014
                                 + 0.924587 w_k1_IBM_s014
                                 + 0.939934 w_k1_INTC_s014
                                 + 0.859387 w_k1_KO_s014
                                 + 1.48716 w_k1_MAT_s014
                                 + 0.813281 w_k1_MSFT_s014
                                 + 0.935734 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.90976 w_k1_ADM_s015
                                 + 0.734969 w_k1_AXP_s015
                                 + 1.0203 w_k1_DIS_s015
                                 + 0.996523 w_k1_FDX_s015
                                 + 0.754977 w_k1_IBM_s015
                                 + 0.839874 w_k1_INTC_s015
                                 + 0.869958 w_k1_KO_s015
                                 + 0.955872 w_k1_MAT_s015
                                 + 0.888017 w_k1_MSFT_s015
                                 + 0.972206 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.11775 w_k1_ADM_s015
                                 + 1.23162 w_k1_AXP_s015
                                 + 0.949659 w_k1_DIS_s015
                                 + 1.1886 w_k1_FDX_s015 + 0.95254 w_k1_IBM_s015
                                 + 1.06753 w_k1_INTC_s015
                                 + 0.942535 w_k1_KO_s015 + 1.3576 w_k1_MAT_s015
                                 + 0.991272 w_k1_MSFT_s015
                                 + 1.04714 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.05067 w_k1_ADM_s015
                                 + 1.16592 w_k1_AXP_s015
                                 + 1.67773 w_k1_DIS_s015
                                 + 1.35346 w_k1_FDX_s015
                                 + 1.62198 w_k1_IBM_s015
                                 + 1.16928 w_k1_INTC_s015
                                 + 1.32588 w_k1_KO_s015 + 2.0188 w_k1_MAT_s015
                                 + 1.29496 w_k1_MSFT_s015
                                 + 1.33838 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.784877 w_k1_ADM_s015
                                 + 1.01139 w_k1_AXP_s015
                                 + 0.929446 w_k1_DIS_s015
                                 + 1.00961 w_k1_FDX_s015
                                 + 1.05683 w_k1_IBM_s015
                                 + 0.959546 w_k1_INTC_s015
                                 + 0.998799 w_k1_KO_s015
                                 + 1.48864 w_k1_MAT_s015
                                 + 0.919035 w_k1_MSFT_s015
                                 + 1.02891 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.619934 w_k1_ADM_s015
                                 + 0.915296 w_k1_AXP_s015
                                 + 0.931676 w_k1_DIS_s015
                                 + 0.851031 w_k1_FDX_s015
                                 + 1.39752 w_k1_IBM_s015
                                 + 0.887699 w_k1_INTC_s015
                                 + 0.935926 w_k1_KO_s015
                                 + 1.74467 w_k1_MAT_s015
                                 + 0.862526 w_k1_MSFT_s015
                                 + 0.988551 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.681051 w_k1_ADM_s015
                                 + 0.670911 w_k1_AXP_s015
                                 + 0.640411 w_k1_DIS_s015
                                 + 0.76328 w_k1_FDX_s015
                                 + 0.778924 w_k1_IBM_s015
                                 + 0.745691 w_k1_INTC_s015
                                 + 0.747084 w_k1_KO_s015
                                 + 0.908975 w_k1_MAT_s015
                                 + 0.71786 w_k1_MSFT_s015
                                 + 0.820159 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.02007 w_k1_ADM_s015
                                 + 1.05883 w_k1_AXP_s015
                                 + 1.31997 w_k1_DIS_s015
                                 + 1.10743 w_k1_FDX_s015
                                 + 1.24002 w_k1_IBM_s015
                                 + 1.00796 w_k1_INTC_s015
                                 + 1.13949 w_k1_KO_s015 + 1.79932 w_k1_MAT_s015
                                 + 0.964033 w_k1_MSFT_s015
                                 + 1.01552 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.29092 w_k1_ADM_s015
                                 + 1.56134 w_k1_AXP_s015
                                 + 1.35391 w_k1_DIS_s015
                                 + 1.38475 w_k1_FDX_s015
                                 + 1.08797 w_k1_IBM_s015
                                 + 1.19909 w_k1_INTC_s015
                                 + 1.15444 w_k1_KO_s015 + 1.46114 w_k1_MAT_s015
                                 + 1.22725 w_k1_MSFT_s015
                                 + 1.12053 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.23472 w_k1_ADM_s016
                                 + 1.60861 w_k1_AXP_s016
                                 + 1.41267 w_k1_DIS_s016
                                 + 1.46459 w_k1_FDX_s016
                                 + 0.974701 w_k1_IBM_s016
                                 + 1.01411 w_k1_INTC_s016
                                 + 0.997192 w_k1_KO_s016
                                 + 1.13486 w_k1_MAT_s016 + 1.037 w_k1_MSFT_s016
                                 + 1.08098 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.03711 w_k1_ADM_s016
                                 + 0.97768 w_k1_AXP_s016
                                 + 1.34592 w_k1_DIS_s016
                                 + 1.20304 w_k1_FDX_s016
                                 + 1.06283 w_k1_IBM_s016
                                 + 1.08086 w_k1_INTC_s016
                                 + 1.26174 w_k1_KO_s016 + 1.1507 w_k1_MAT_s016
                                 + 1.21598 w_k1_MSFT_s016
                                 + 1.11162 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.00087 w_k1_ADM_s016
                                 + 1.00613 w_k1_AXP_s016 + 1.3605 w_k1_DIS_s016
                                 + 1.00997 w_k1_FDX_s016
                                 + 1.57398 w_k1_IBM_s016
                                 + 0.91992 w_k1_INTC_s016
                                 + 1.43501 w_k1_KO_s016 + 1.56072 w_k1_MAT_s016
                                 + 1.08932 w_k1_MSFT_s016
                                 + 1.19016 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.852955 w_k1_ADM_s016
                                 + 0.99399 w_k1_AXP_s016
                                 + 1.23441 w_k1_DIS_s016
                                 + 1.02824 w_k1_FDX_s016
                                 + 1.00167 w_k1_IBM_s016
                                 + 0.92818 w_k1_INTC_s016
                                 + 1.03356 w_k1_KO_s016 + 0.90442 w_k1_MAT_s016
                                 + 1.03502 w_k1_MSFT_s016
                                 + 1.00834 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.761948 w_k1_ADM_s016
                                 + 0.801666 w_k1_AXP_s016
                                 + 0.868674 w_k1_DIS_s016
                                 + 0.912809 w_k1_FDX_s016
                                 + 0.734579 w_k1_IBM_s016
                                 + 0.822606 w_k1_INTC_s016
                                 + 0.786493 w_k1_KO_s016
                                 + 0.761826 w_k1_MAT_s016
                                 + 0.988569 w_k1_MSFT_s016
                                 + 0.84259 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.767176 w_k1_ADM_s016
                                 + 0.750842 w_k1_AXP_s016
                                 + 0.925943 w_k1_DIS_s016
                                 + 0.751001 w_k1_FDX_s016
                                 + 1.08043 w_k1_IBM_s016
                                 + 0.681797 w_k1_INTC_s016
                                 + 0.987148 w_k1_KO_s016
                                 + 1.13144 w_k1_MAT_s016
                                 + 0.73519 w_k1_MSFT_s016
                                 + 0.941595 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.2075 w_k1_ADM_s016
                                 + 1.09679 w_k1_AXP_s016
                                 + 0.98106 w_k1_DIS_s016
                                 + 1.02507 w_k1_FDX_s016
                                 + 1.21753 w_k1_IBM_s016
                                 + 0.861446 w_k1_INTC_s016
                                 + 1.02717 w_k1_KO_s016 + 1.41782 w_k1_MAT_s016
                                 + 0.925576 w_k1_MSFT_s016
                                 + 1.21137 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.33545 w_k1_ADM_s016
                                 + 1.27072 w_k1_AXP_s016
                                 + 1.91824 w_k1_DIS_s016
                                 + 1.35898 w_k1_FDX_s016
                                 + 1.42307 w_k1_IBM_s016
                                 + 1.29573 w_k1_INTC_s016
                                 + 1.57743 w_k1_KO_s016 + 1.56459 w_k1_MAT_s016
                                 + 1.50656 w_k1_MSFT_s016
                                 + 1.25387 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.07907 w_k1_ADM_s017
                                 + 1.21104 w_k1_AXP_s017
                                 + 1.11637 w_k1_DIS_s017
                                 + 1.10316 w_k1_FDX_s017
                                 + 1.07729 w_k1_IBM_s017
                                 + 1.03501 w_k1_INTC_s017 + 1.0218 w_k1_KO_s017
                                 + 1.33328 w_k1_MAT_s017
                                 + 0.975874 w_k1_MSFT_s017
                                 + 1.0267 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.965788 w_k1_ADM_s017
                                 + 0.979447 w_k1_AXP_s017
                                 + 0.850454 w_k1_DIS_s017
                                 + 0.898575 w_k1_FDX_s017
                                 + 0.806001 w_k1_IBM_s017
                                 + 0.832776 w_k1_INTC_s017
                                 + 0.871486 w_k1_KO_s017
                                 + 1.01623 w_k1_MAT_s017
                                 + 0.885019 w_k1_MSFT_s017
                                 + 0.851212 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.12715 w_k1_ADM_s017
                                 + 1.09311 w_k1_AXP_s017
                                 + 0.777186 w_k1_DIS_s017
                                 + 1.17482 w_k1_FDX_s017
                                 + 0.710465 w_k1_IBM_s017
                                 + 0.973102 w_k1_INTC_s017
                                 + 0.742718 w_k1_KO_s017
                                 + 0.674407 w_k1_MAT_s017
                                 + 1.00527 w_k1_MSFT_s017
                                 + 0.907364 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.998502 w_k1_ADM_s017
                                 + 1.1071 w_k1_AXP_s017
                                 + 0.856754 w_k1_DIS_s017
                                 + 0.865749 w_k1_FDX_s017
                                 + 1.63745 w_k1_IBM_s017
                                 + 1.11976 w_k1_INTC_s017
                                 + 1.17281 w_k1_KO_s017 + 1.67313 w_k1_MAT_s017
                                 + 0.926009 w_k1_MSFT_s017
                                 + 1.05607 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.783648 w_k1_ADM_s017
                                 + 0.583533 w_k1_AXP_s017
                                 + 0.546632 w_k1_DIS_s017
                                 + 0.674918 w_k1_FDX_s017
                                 + 0.81124 w_k1_IBM_s017
                                 + 0.703054 w_k1_INTC_s017
                                 + 0.646262 w_k1_KO_s017
                                 + 0.705873 w_k1_MAT_s017
                                 + 0.678652 w_k1_MSFT_s017
                                 + 0.733688 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.33687 w_k1_ADM_s017
                                 + 1.57817 w_k1_AXP_s017 + 1.585 w_k1_DIS_s017
                                 + 1.2938 w_k1_FDX_s017 + 1.27809 w_k1_IBM_s017
                                 + 1.09955 w_k1_INTC_s017
                                 + 1.22255 w_k1_KO_s017 + 1.41556 w_k1_MAT_s017
                                 + 1.22652 w_k1_MSFT_s017
                                 + 1.20323 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.857226 w_k1_ADM_s017
                                 + 1.08858 w_k1_AXP_s017
                                 + 0.743093 w_k1_DIS_s017
                                 + 0.807072 w_k1_FDX_s017
                                 + 1.15587 w_k1_IBM_s017
                                 + 0.849437 w_k1_INTC_s017
                                 + 0.967549 w_k1_KO_s017
                                 + 1.70703 w_k1_MAT_s017
                                 + 0.894488 w_k1_MSFT_s017
                                 + 1.00834 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.95321 w_k1_ADM_s017
                                 + 1.06343 w_k1_AXP_s017
                                 + 1.09531 w_k1_DIS_s017
                                 + 1.13042 w_k1_FDX_s017
                                 + 0.563607 w_k1_IBM_s017
                                 + 0.980043 w_k1_INTC_s017
                                 + 0.821166 w_k1_KO_s017
                                 + 0.729762 w_k1_MAT_s017
                                 + 0.992135 w_k1_MSFT_s017
                                 + 0.886919 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.06761 w_k1_ADM_s100
                                 + 1.66542 w_k1_AXP_s100
                                 + 1.71403 w_k1_DIS_s100
                                 + 1.26943 w_k1_FDX_s100
                                 + 1.22386 w_k1_IBM_s100
                                 + 1.22154 w_k1_INTC_s100 + 1.4133 w_k1_KO_s100
                                 + 1.39628 w_k1_MAT_s100
                                 + 1.33275 w_k1_MSFT_s100
                                 + 1.20058 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.01793 w_k1_ADM_s100
                                 + 0.86038 w_k1_AXP_s100
                                 + 0.980688 w_k1_DIS_s100
                                 + 0.954593 w_k1_FDX_s100
                                 + 0.795867 w_k1_IBM_s100
                                 + 1.08552 w_k1_INTC_s100
                                 + 0.858631 w_k1_KO_s100
                                 + 0.636969 w_k1_MAT_s100
                                 + 1.09804 w_k1_MSFT_s100
                                 + 0.931012 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.19784 w_k1_ADM_s100
                                 + 1.26396 w_k1_AXP_s100
                                 + 1.07952 w_k1_DIS_s100
                                 + 1.08309 w_k1_FDX_s100 + 1.0063 w_k1_IBM_s100
                                 + 1.21161 w_k1_INTC_s100
                                 + 1.07302 w_k1_KO_s100
                                 + 0.717648 w_k1_MAT_s100
                                 + 1.10726 w_k1_MSFT_s100
                                 + 1.05275 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.877985 w_k1_ADM_s100
                                 + 0.773656 w_k1_AXP_s100
                                 + 0.690387 w_k1_DIS_s100
                                 + 0.697249 w_k1_FDX_s100
                                 + 0.925546 w_k1_IBM_s100
                                 + 0.802244 w_k1_INTC_s100
                                 + 0.801209 w_k1_KO_s100
                                 + 0.692404 w_k1_MAT_s100
                                 + 0.779663 w_k1_MSFT_s100
                                 + 0.76645 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.02738 w_k1_ADM_s100
                                 + 0.776499 w_k1_AXP_s100
                                 + 0.708941 w_k1_DIS_s100
                                 + 0.875676 w_k1_FDX_s100
                                 + 0.594967 w_k1_IBM_s100
                                 + 0.811877 w_k1_INTC_s100
                                 + 0.740835 w_k1_KO_s100
                                 + 0.594883 w_k1_MAT_s100
                                 + 0.908632 w_k1_MSFT_s100
                                 + 0.813141 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.09619 w_k1_ADM_s100
                                 + 1.22537 w_k1_AXP_s100
                                 + 1.09825 w_k1_DIS_s100
                                 + 0.891935 w_k1_FDX_s100
                                 + 1.42684 w_k1_IBM_s100
                                 + 0.868804 w_k1_INTC_s100
                                 + 1.22435 w_k1_KO_s100 + 1.52973 w_k1_MAT_s100
                                 + 0.884541 w_k1_MSFT_s100
                                 + 1.09252 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.17259 w_k1_ADM_s100
                                 + 1.14224 w_k1_AXP_s100
                                 + 1.04529 w_k1_DIS_s100
                                 + 1.23591 w_k1_FDX_s100
                                 + 0.642562 w_k1_IBM_s100
                                 + 1.25717 w_k1_INTC_s100
                                 + 0.899273 w_k1_KO_s100
                                 + 0.635396 w_k1_MAT_s100
                                 + 1.26279 w_k1_MSFT_s100
                                 + 1.06685 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.07836 w_k1_ADM_s100
                                 + 0.945889 w_k1_AXP_s100
                                 + 0.933769 w_k1_DIS_s100
                                 + 0.928835 w_k1_FDX_s100
                                 + 0.969786 w_k1_IBM_s100
                                 + 0.978257 w_k1_INTC_s100
                                 + 0.964923 w_k1_KO_s100
                                 + 0.820824 w_k1_MAT_s100
                                 + 0.974839 w_k1_MSFT_s100
                                 + 0.98196 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.966947 w_k1_ADM_s101
                                 + 0.647216 w_k1_AXP_s101
                                 + 0.624889 w_k1_DIS_s101
                                 + 0.7424 w_k1_FDX_s101
                                 + 0.780947 w_k1_IBM_s101
                                 + 0.691019 w_k1_INTC_s101
                                 + 0.764806 w_k1_KO_s101
                                 + 0.584376 w_k1_MAT_s101
                                 + 0.660729 w_k1_MSFT_s101
                                 + 0.830642 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.859381 w_k1_ADM_s101
                                 + 0.747508 w_k1_AXP_s101
                                 + 0.759081 w_k1_DIS_s101
                                 + 0.85422 w_k1_FDX_s101
                                 + 0.798429 w_k1_IBM_s101
                                 + 0.779827 w_k1_INTC_s101
                                 + 0.750643 w_k1_KO_s101
                                 + 0.953824 w_k1_MAT_s101
                                 + 0.802704 w_k1_MSFT_s101
                                 + 0.863142 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.996943 w_k1_ADM_s101
                                 + 1.1074 w_k1_AXP_s101 + 1.5917 w_k1_DIS_s101
                                 + 1.35015 w_k1_FDX_s101
                                 + 0.973793 w_k1_IBM_s101
                                 + 1.22789 w_k1_INTC_s101
                                 + 1.21152 w_k1_KO_s101
                                 + 0.698384 w_k1_MAT_s101
                                 + 1.16453 w_k1_MSFT_s101
                                 + 1.12922 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.831275 w_k1_ADM_s101
                                 + 1.01464 w_k1_AXP_s101
                                 + 0.917441 w_k1_DIS_s101
                                 + 0.852574 w_k1_FDX_s101
                                 + 1.49884 w_k1_IBM_s101
                                 + 0.889417 w_k1_INTC_s101
                                 + 1.09766 w_k1_KO_s101 + 1.49325 w_k1_MAT_s101
                                 + 0.821796 w_k1_MSFT_s101
                                 + 0.973964 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.890001 w_k1_ADM_s101
                                 + 0.700033 w_k1_AXP_s101
                                 + 0.788525 w_k1_DIS_s101
                                 + 0.919482 w_k1_FDX_s101
                                 + 0.553472 w_k1_IBM_s101
                                 + 0.838269 w_k1_INTC_s101
                                 + 0.764292 w_k1_KO_s101
                                 + 0.666444 w_k1_MAT_s101
                                 + 0.893405 w_k1_MSFT_s101
                                 + 0.972794 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.916149 w_k1_ADM_s101
                                 + 0.877602 w_k1_AXP_s101
                                 + 1.14025 w_k1_DIS_s101
                                 + 1.03892 w_k1_FDX_s101
                                 + 1.30746 w_k1_IBM_s101
                                 + 1.02238 w_k1_INTC_s101 + 1.2698 w_k1_KO_s101
                                 + 1.57018 w_k1_MAT_s101
                                 + 1.07157 w_k1_MSFT_s101
                                 + 1.11064 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.930643 w_k1_ADM_s101
                                 + 1.0067 w_k1_AXP_s101
                                 + 0.949647 w_k1_DIS_s101
                                 + 1.14835 w_k1_FDX_s101
                                 + 0.662653 w_k1_IBM_s101
                                 + 0.990032 w_k1_INTC_s101
                                 + 0.813055 w_k1_KO_s101
                                 + 0.621172 w_k1_MAT_s101
                                 + 0.975532 w_k1_MSFT_s101
                                 + 0.89801 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.966486 w_k1_ADM_s101
                                 + 1.17564 w_k1_AXP_s101
                                 + 1.00503 w_k1_DIS_s101
                                 + 1.10412 w_k1_FDX_s101
                                 + 1.09567 w_k1_IBM_s101
                                 + 1.1324 w_k1_INTC_s101 + 1.02603 w_k1_KO_s101
                                 + 1.17386 w_k1_MAT_s101
                                 + 0.881996 w_k1_MSFT_s101
                                 + 1.0235 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.961362 w_k1_ADM_s102
                                 + 1.02106 w_k1_AXP_s102
                                 + 0.754473 w_k1_DIS_s102
                                 + 0.856062 w_k1_FDX_s102
                                 + 1.01399 w_k1_IBM_s102
                                 + 0.91927 w_k1_INTC_s102
                                 + 0.820625 w_k1_KO_s102
                                 + 0.982947 w_k1_MAT_s102
                                 + 0.796881 w_k1_MSFT_s102
                                 + 0.960013 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.985132 w_k1_ADM_s102
                                 + 1.28836 w_k1_AXP_s102
                                 + 1.36973 w_k1_DIS_s102
                                 + 1.15451 w_k1_FDX_s102
                                 + 1.36213 w_k1_IBM_s102
                                 + 1.2169 w_k1_INTC_s102 + 1.32363 w_k1_KO_s102
                                 + 1.41519 w_k1_MAT_s102
                                 + 1.19317 w_k1_MSFT_s102
                                 + 1.11599 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.14218 w_k1_ADM_s102
                                 + 1.04429 w_k1_AXP_s102
                                 + 1.31371 w_k1_DIS_s102
                                 + 1.26368 w_k1_FDX_s102
                                 + 0.87874 w_k1_IBM_s102
                                 + 1.15989 w_k1_INTC_s102
                                 + 1.03839 w_k1_KO_s102
                                 + 0.886204 w_k1_MAT_s102
                                 + 1.17519 w_k1_MSFT_s102
                                 + 1.14401 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.02735 w_k1_ADM_s102
                                 + 0.823006 w_k1_AXP_s102
                                 + 0.772669 w_k1_DIS_s102
                                 + 0.92339 w_k1_FDX_s102
                                 + 0.795751 w_k1_IBM_s102
                                 + 0.812217 w_k1_INTC_s102
                                 + 0.77743 w_k1_KO_s102
                                 + 0.804851 w_k1_MAT_s102
                                 + 0.837383 w_k1_MSFT_s102
                                 + 0.935734 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.07935 w_k1_ADM_s102
                                 + 1.58992 w_k1_AXP_s102
                                 + 1.35561 w_k1_DIS_s102
                                 + 1.05101 w_k1_FDX_s102 + 1.6573 w_k1_IBM_s102
                                 + 0.865089 w_k1_INTC_s102
                                 + 1.3377 w_k1_KO_s102 + 2.13598 w_k1_MAT_s102
                                 + 0.826678 w_k1_MSFT_s102
                                 + 1.23456 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.887062 w_k1_ADM_s102
                                 + 0.88877 w_k1_AXP_s102
                                 + 0.674612 w_k1_DIS_s102
                                 + 0.911023 w_k1_FDX_s102
                                 + 0.59588 w_k1_IBM_s102
                                 + 0.781333 w_k1_INTC_s102
                                 + 0.658493 w_k1_KO_s102
                                 + 0.638131 w_k1_MAT_s102
                                 + 0.75861 w_k1_MSFT_s102
                                 + 0.88912 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.22724 w_k1_ADM_s102
                                 + 0.966256 w_k1_AXP_s102
                                 + 1.01045 w_k1_DIS_s102
                                 + 1.00755 w_k1_FDX_s102
                                 + 1.14773 w_k1_IBM_s102
                                 + 0.941704 w_k1_INTC_s102
                                 + 1.0664 w_k1_KO_s102 + 1.027 w_k1_MAT_s102
                                 + 0.97894 w_k1_MSFT_s102
                                 + 1.16009 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.24076 w_k1_ADM_s102
                                 + 1.46297 w_k1_AXP_s102
                                 + 1.05341 w_k1_DIS_s102
                                 + 1.30456 w_k1_FDX_s102
                                 + 1.08547 w_k1_IBM_s102
                                 + 1.23973 w_k1_INTC_s102
                                 + 1.21004 w_k1_KO_s102 + 1.48419 w_k1_MAT_s102
                                 + 1.13457 w_k1_MSFT_s102
                                 + 1.11382 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.09751 w_k1_ADM_s103
                                 + 1.41902 w_k1_AXP_s103 + 1.3934 w_k1_DIS_s103
                                 + 1.076 w_k1_FDX_s103 + 1.74509 w_k1_IBM_s103
                                 + 1.29453 w_k1_INTC_s103
                                 + 1.58383 w_k1_KO_s103 + 1.5171 w_k1_MAT_s103
                                 + 1.2137 w_k1_MSFT_s103 + 1.20334 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.21726 w_k1_ADM_s103
                                 + 1.04059 w_k1_AXP_s103
                                 + 0.924509 w_k1_DIS_s103
                                 + 1.07657 w_k1_FDX_s103 + 0.6836 w_k1_IBM_s103
                                 + 1.00491 w_k1_INTC_s103
                                 + 0.904682 w_k1_KO_s103
                                 + 0.659958 w_k1_MAT_s103
                                 + 0.898097 w_k1_MSFT_s103
                                 + 0.935127 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.01812 w_k1_ADM_s103
                                 + 0.833287 w_k1_AXP_s103
                                 + 1.01253 w_k1_DIS_s103
                                 + 0.995854 w_k1_FDX_s103
                                 + 0.715652 w_k1_IBM_s103
                                 + 1.17015 w_k1_INTC_s103
                                 + 0.912416 w_k1_KO_s103
                                 + 0.613052 w_k1_MAT_s103
                                 + 1.04304 w_k1_MSFT_s103
                                 + 0.923891 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.01651 w_k1_ADM_s103
                                 + 0.994103 w_k1_AXP_s103
                                 + 0.913486 w_k1_DIS_s103
                                 + 0.969341 w_k1_FDX_s103
                                 + 1.21708 w_k1_IBM_s103
                                 + 1.26437 w_k1_INTC_s103
                                 + 1.31574 w_k1_KO_s103 + 1.32257 w_k1_MAT_s103
                                 + 1.07861 w_k1_MSFT_s103
                                 + 1.03237 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.21954 w_k1_ADM_s103
                                 + 1.25717 w_k1_AXP_s103
                                 + 1.24877 w_k1_DIS_s103 + 1.1213 w_k1_FDX_s103
                                 + 1.38066 w_k1_IBM_s103
                                 + 0.967358 w_k1_INTC_s103
                                 + 1.39412 w_k1_KO_s103 + 1.58467 w_k1_MAT_s103
                                 + 1.0512 w_k1_MSFT_s103 + 1.1669 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.708139 w_k1_ADM_s103
                                 + 0.657136 w_k1_AXP_s103
                                 + 0.659851 w_k1_DIS_s103
                                 + 0.769877 w_k1_FDX_s103
                                 + 0.681633 w_k1_IBM_s103
                                 + 1.00307 w_k1_INTC_s103
                                 + 0.778124 w_k1_KO_s103
                                 + 0.627441 w_k1_MAT_s103
                                 + 0.761365 w_k1_MSFT_s103
                                 + 0.751421 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.40072 w_k1_ADM_s103
                                 + 1.40678 w_k1_AXP_s103
                                 + 1.31738 w_k1_DIS_s103
                                 + 1.24754 w_k1_FDX_s103 + 0.9039 w_k1_IBM_s103
                                 + 1.20314 w_k1_INTC_s103
                                 + 1.15088 w_k1_KO_s103 + 1.03436 w_k1_MAT_s103
                                 + 1.15917 w_k1_MSFT_s103
                                 + 1.03294 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.08624 w_k1_ADM_s103
                                 + 0.908138 w_k1_AXP_s103
                                 + 0.839731 w_k1_DIS_s103
                                 + 0.916373 w_k1_FDX_s103
                                 + 0.958716 w_k1_IBM_s103
                                 + 1.16905 w_k1_INTC_s103
                                 + 1.04564 w_k1_KO_s103
                                 + 0.909348 w_k1_MAT_s103
                                 + 0.903183 w_k1_MSFT_s103
                                 + 0.85639 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.790184 w_k1_ADM_s104
                                 + 0.932419 w_k1_AXP_s104
                                 + 0.722228 w_k1_DIS_s104
                                 + 0.937537 w_k1_FDX_s104
                                 + 0.764722 w_k1_IBM_s104
                                 + 0.964675 w_k1_INTC_s104
                                 + 0.721183 w_k1_KO_s104
                                 + 0.762067 w_k1_MAT_s104
                                 + 0.850291 w_k1_MSFT_s104
                                 + 0.85255 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.941793 w_k1_ADM_s104
                                 + 0.736387 w_k1_AXP_s104
                                 + 0.618629 w_k1_DIS_s104
                                 + 0.78024 w_k1_FDX_s104
                                 + 0.796267 w_k1_IBM_s104
                                 + 0.854783 w_k1_INTC_s104
                                 + 0.799515 w_k1_KO_s104
                                 + 0.815791 w_k1_MAT_s104
                                 + 0.818907 w_k1_MSFT_s104
                                 + 0.889481 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.03301 w_k1_ADM_s104
                                 + 1.18669 w_k1_AXP_s104
                                 + 0.75935 w_k1_DIS_s104
                                 + 0.842667 w_k1_FDX_s104
                                 + 1.31622 w_k1_IBM_s104
                                 + 0.870259 w_k1_INTC_s104
                                 + 1.08282 w_k1_KO_s104 + 1.47008 w_k1_MAT_s104
                                 + 0.768471 w_k1_MSFT_s104
                                 + 0.916929 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.941162 w_k1_ADM_s104
                                 + 0.891711 w_k1_AXP_s104
                                 + 0.974971 w_k1_DIS_s104
                                 + 0.901993 w_k1_FDX_s104
                                 + 1.21282 w_k1_IBM_s104
                                 + 0.980738 w_k1_INTC_s104
                                 + 1.11432 w_k1_KO_s104 + 1.23396 w_k1_MAT_s104
                                 + 0.950023 w_k1_MSFT_s104
                                 + 1.05597 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.980214 w_k1_ADM_s104
                                 + 0.931716 w_k1_AXP_s104
                                 + 0.860933 w_k1_DIS_s104
                                 + 0.885487 w_k1_FDX_s104
                                 + 0.953493 w_k1_IBM_s104
                                 + 0.925896 w_k1_INTC_s104
                                 + 0.979506 w_k1_KO_s104
                                 + 0.781098 w_k1_MAT_s104
                                 + 0.879051 w_k1_MSFT_s104
                                 + 0.915329 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.31323 w_k1_ADM_s104
                                 + 1.08411 w_k1_AXP_s104
                                 + 1.33532 w_k1_DIS_s104
                                 + 1.16711 w_k1_FDX_s104
                                 + 0.878598 w_k1_IBM_s104
                                 + 1.16139 w_k1_INTC_s104
                                 + 1.20587 w_k1_KO_s104
                                 + 0.641722 w_k1_MAT_s104
                                 + 1.25533 w_k1_MSFT_s104
                                 + 1.12755 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.28474 w_k1_ADM_s104
                                 + 1.29323 w_k1_AXP_s104
                                 + 1.06772 w_k1_DIS_s104
                                 + 1.05685 w_k1_FDX_s104
                                 + 1.04247 w_k1_IBM_s104
                                 + 1.1154 w_k1_INTC_s104 + 1.06419 w_k1_KO_s104
                                 + 1.1484 w_k1_MAT_s104 + 1.0267 w_k1_MSFT_s104
                                 + 1.05083 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.34834 w_k1_ADM_s104
                                 + 1.53108 w_k1_AXP_s104
                                 + 1.54675 w_k1_DIS_s104
                                 + 1.22527 w_k1_FDX_s104
                                 + 1.41605 w_k1_IBM_s104
                                 + 1.25026 w_k1_INTC_s104
                                 + 1.46304 w_k1_KO_s104 + 1.52923 w_k1_MAT_s104
                                 + 1.22071 w_k1_MSFT_s104
                                 + 1.33743 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.758773 w_k1_ADM_s105
                                 + 0.82662 w_k1_AXP_s105
                                 + 0.928326 w_k1_DIS_s105
                                 + 0.808289 w_k1_FDX_s105
                                 + 1.40299 w_k1_IBM_s105
                                 + 0.759891 w_k1_INTC_s105
                                 + 1.06765 w_k1_KO_s105 + 1.32437 w_k1_MAT_s105
                                 + 0.776901 w_k1_MSFT_s105
                                 + 1.03234 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.50939 w_k1_ADM_s105
                                 + 1.50547 w_k1_AXP_s105
                                 + 1.61697 w_k1_DIS_s105
                                 + 1.32652 w_k1_FDX_s105
                                 + 1.05884 w_k1_IBM_s105
                                 + 1.33444 w_k1_INTC_s105
                                 + 1.26321 w_k1_KO_s105
                                 + 0.974737 w_k1_MAT_s105
                                 + 1.14464 w_k1_MSFT_s105
                                 + 1.16339 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.00924 w_k1_ADM_s105
                                 + 0.866731 w_k1_AXP_s105
                                 + 1.06241 w_k1_DIS_s105
                                 + 0.974225 w_k1_FDX_s105
                                 + 0.952123 w_k1_IBM_s105
                                 + 0.817302 w_k1_INTC_s105
                                 + 1.1947 w_k1_KO_s105 + 0.832014 w_k1_MAT_s105
                                 + 0.954308 w_k1_MSFT_s105
                                 + 0.971581 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.12866 w_k1_ADM_s105
                                 + 1.49713 w_k1_AXP_s105
                                 + 1.46534 w_k1_DIS_s105
                                 + 1.16631 w_k1_FDX_s105
                                 + 1.70073 w_k1_IBM_s105
                                 + 1.11216 w_k1_INTC_s105
                                 + 1.63161 w_k1_KO_s105 + 1.58635 w_k1_MAT_s105
                                 + 1.09392 w_k1_MSFT_s105
                                 + 1.36146 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.648867 w_k1_ADM_s105
                                 + 0.642873 w_k1_AXP_s105
                                 + 0.709596 w_k1_DIS_s105
                                 + 0.764244 w_k1_FDX_s105
                                 + 0.68838 w_k1_IBM_s105
                                 + 0.868897 w_k1_INTC_s105
                                 + 0.774729 w_k1_KO_s105
                                 + 0.718499 w_k1_MAT_s105
                                 + 0.743293 w_k1_MSFT_s105
                                 + 0.836572 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.3645 w_k1_ADM_s105
                                 + 1.37099 w_k1_AXP_s105
                                 + 1.47399 w_k1_DIS_s105
                                 + 1.27025 w_k1_FDX_s105
                                 + 1.23429 w_k1_IBM_s105
                                 + 1.04029 w_k1_INTC_s105
                                 + 1.35687 w_k1_KO_s105 + 1.27735 w_k1_MAT_s105
                                 + 1.10847 w_k1_MSFT_s105
                                 + 1.23299 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.977853 w_k1_ADM_s105
                                 + 1.0522 w_k1_AXP_s105 + 1.06815 w_k1_DIS_s105
                                 + 0.899134 w_k1_FDX_s105
                                 + 1.21188 w_k1_IBM_s105
                                 + 0.996683 w_k1_INTC_s105
                                 + 0.950048 w_k1_KO_s105
                                 + 1.29798 w_k1_MAT_s105
                                 + 0.883571 w_k1_MSFT_s105
                                 + 0.983039 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.02272 w_k1_ADM_s105
                                 + 1.03384 w_k1_AXP_s105
                                 + 0.92837 w_k1_DIS_s105
                                 + 1.05829 w_k1_FDX_s105
                                 + 0.772057 w_k1_IBM_s105
                                 + 1.06404 w_k1_INTC_s105
                                 + 0.832397 w_k1_KO_s105
                                 + 0.785169 w_k1_MAT_s105
                                 + 0.965963 w_k1_MSFT_s105
                                 + 1.00774 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.94048 w_k1_ADM_s106
                                 + 0.666304 w_k1_AXP_s106
                                 + 0.806805 w_k1_DIS_s106
                                 + 0.912904 w_k1_FDX_s106
                                 + 0.580977 w_k1_IBM_s106
                                 + 1.04654 w_k1_INTC_s106
                                 + 0.71906 w_k1_KO_s106
                                 + 0.555424 w_k1_MAT_s106
                                 + 0.901438 w_k1_MSFT_s106
                                 + 0.939474 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.889557 w_k1_ADM_s106
                                 + 0.95081 w_k1_AXP_s106
                                 + 1.03844 w_k1_DIS_s106
                                 + 0.949189 w_k1_FDX_s106
                                 + 1.04815 w_k1_IBM_s106
                                 + 1.19416 w_k1_INTC_s106
                                 + 1.06432 w_k1_KO_s106
                                 + 0.870887 w_k1_MAT_s106
                                 + 0.990669 w_k1_MSFT_s106
                                 + 1.06075 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.03965 w_k1_ADM_s106
                                 + 1.03797 w_k1_AXP_s106
                                 + 0.941446 w_k1_DIS_s106
                                 + 0.837238 w_k1_FDX_s106
                                 + 1.37399 w_k1_IBM_s106
                                 + 1.06258 w_k1_INTC_s106
                                 + 1.17956 w_k1_KO_s106 + 1.23634 w_k1_MAT_s106
                                 + 0.942848 w_k1_MSFT_s106
                                 + 1.02072 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.8394 w_k1_ADM_s106
                                 + 0.711933 w_k1_AXP_s106
                                 + 0.865203 w_k1_DIS_s106
                                 + 0.877533 w_k1_FDX_s106
                                 + 0.723219 w_k1_IBM_s106
                                 + 1.03923 w_k1_INTC_s106
                                 + 0.891458 w_k1_KO_s106
                                 + 0.544937 w_k1_MAT_s106
                                 + 0.905858 w_k1_MSFT_s106
                                 + 1.00192 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.946597 w_k1_ADM_s106
                                 + 1.31012 w_k1_AXP_s106
                                 + 1.55652 w_k1_DIS_s106
                                 + 1.11666 w_k1_FDX_s106
                                 + 1.47992 w_k1_IBM_s106
                                 + 1.28858 w_k1_INTC_s106
                                 + 1.29354 w_k1_KO_s106 + 1.39139 w_k1_MAT_s106
                                 + 1.12383 w_k1_MSFT_s106
                                 + 1.19711 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.11028 w_k1_ADM_s106
                                 + 1.34616 w_k1_AXP_s106
                                 + 1.33403 w_k1_DIS_s106 + 1.2852 w_k1_FDX_s106
                                 + 0.902888 w_k1_IBM_s106
                                 + 1.35763 w_k1_INTC_s106 + 1.0266 w_k1_KO_s106
                                 + 1.00312 w_k1_MAT_s106
                                 + 1.23478 w_k1_MSFT_s106
                                 + 1.23962 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.940532 w_k1_ADM_s106
                                 + 0.90102 w_k1_AXP_s106
                                 + 0.938688 w_k1_DIS_s106
                                 + 0.936401 w_k1_FDX_s106
                                 + 0.846444 w_k1_IBM_s106
                                 + 0.963859 w_k1_INTC_s106
                                 + 0.940303 w_k1_KO_s106
                                 + 0.783993 w_k1_MAT_s106
                                 + 0.984161 w_k1_MSFT_s106
                                 + 1.05789 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.916898 w_k1_ADM_s106
                                 + 1.19222 w_k1_AXP_s106
                                 + 0.823814 w_k1_DIS_s106
                                 + 0.956453 w_k1_FDX_s106
                                 + 0.948069 w_k1_IBM_s106
                                 + 0.999678 w_k1_INTC_s106
                                 + 0.92309 w_k1_KO_s106 + 1.0475 w_k1_MAT_s106
                                 + 0.82307 w_k1_MSFT_s106
                                 + 1.07478 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.909832 w_k1_ADM_s107
                                 + 0.64435 w_k1_AXP_s107
                                 + 0.569241 w_k1_DIS_s107
                                 + 0.781418 w_k1_FDX_s107
                                 + 0.787271 w_k1_IBM_s107
                                 + 0.984214 w_k1_INTC_s107
                                 + 0.6676 w_k1_KO_s107 + 0.711714 w_k1_MAT_s107
                                 + 0.861994 w_k1_MSFT_s107
                                 + 0.809974 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.07084 w_k1_ADM_s107
                                 + 1.09649 w_k1_AXP_s107
                                 + 1.07523 w_k1_DIS_s107
                                 + 1.11268 w_k1_FDX_s107
                                 + 1.15085 w_k1_IBM_s107
                                 + 1.05269 w_k1_INTC_s107
                                 + 0.930387 w_k1_KO_s107
                                 + 1.48972 w_k1_MAT_s107
                                 + 0.991478 w_k1_MSFT_s107
                                 + 1.00311 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.13386 w_k1_ADM_s107
                                 + 0.867178 w_k1_AXP_s107
                                 + 1.01424 w_k1_DIS_s107
                                 + 1.08497 w_k1_FDX_s107
                                 + 0.728875 w_k1_IBM_s107
                                 + 1.22403 w_k1_INTC_s107
                                 + 0.748161 w_k1_KO_s107
                                 + 0.597253 w_k1_MAT_s107
                                 + 1.11646 w_k1_MSFT_s107
                                 + 0.978957 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.06106 w_k1_ADM_s107
                                 + 0.889374 w_k1_AXP_s107
                                 + 0.902273 w_k1_DIS_s107
                                 + 0.884378 w_k1_FDX_s107
                                 + 1.2387 w_k1_IBM_s107
                                 + 1.05233 w_k1_INTC_s107
                                 + 0.884078 w_k1_KO_s107
                                 + 1.39437 w_k1_MAT_s107
                                 + 1.03508 w_k1_MSFT_s107
                                 + 1.01946 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.880822 w_k1_ADM_s107
                                 + 0.855182 w_k1_AXP_s107
                                 + 0.971204 w_k1_DIS_s107
                                 + 0.992892 w_k1_FDX_s107
                                 + 0.925345 w_k1_IBM_s107
                                 + 1.04997 w_k1_INTC_s107
                                 + 0.875479 w_k1_KO_s107
                                 + 1.37276 w_k1_MAT_s107
                                 + 1.02447 w_k1_MSFT_s107
                                 + 1.01448 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.08074 w_k1_ADM_s107
                                 + 1.26741 w_k1_AXP_s107
                                 + 1.57901 w_k1_DIS_s107
                                 + 1.22023 w_k1_FDX_s107
                                 + 1.44099 w_k1_IBM_s107
                                 + 1.34951 w_k1_INTC_s107
                                 + 1.19699 w_k1_KO_s107 + 1.79902 w_k1_MAT_s107
                                 + 1.25442 w_k1_MSFT_s107
                                 + 1.22423 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.50001 w_k1_ADM_s107
                                 + 1.42382 w_k1_AXP_s107
                                 + 1.66431 w_k1_DIS_s107
                                 + 1.47605 w_k1_FDX_s107
                                 + 1.04286 w_k1_IBM_s107
                                 + 1.4702 w_k1_INTC_s107 + 1.07149 w_k1_KO_s107
                                 + 1.50749 w_k1_MAT_s107
                                 + 1.4739 w_k1_MSFT_s107 + 1.10467 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.48165 w_k1_ADM_s107
                                 + 1.29351 w_k1_AXP_s107
                                 + 0.910043 w_k1_DIS_s107
                                 + 1.21821 w_k1_FDX_s107
                                 + 0.811134 w_k1_IBM_s107
                                 + 1.11975 w_k1_INTC_s107
                                 + 0.787495 w_k1_KO_s107
                                 + 1.15064 w_k1_MAT_s107
                                 + 1.09741 w_k1_MSFT_s107 + 1.0563 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.23867 w_k1_ADM_s110
                                 + 1.33663 w_k1_AXP_s110
                                 + 0.78317 w_k1_DIS_s110
                                 + 1.01605 w_k1_FDX_s110
                                 + 1.07891 w_k1_IBM_s110
                                 + 0.953045 w_k1_INTC_s110
                                 + 0.814884 w_k1_KO_s110
                                 + 1.14018 w_k1_MAT_s110
                                 + 0.848564 w_k1_MSFT_s110
                                 + 0.925292 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.936109 w_k1_ADM_s110
                                 + 0.86166 w_k1_AXP_s110
                                 + 0.949084 w_k1_DIS_s110
                                 + 0.963146 w_k1_FDX_s110
                                 + 1.14878 w_k1_IBM_s110
                                 + 1.1103 w_k1_INTC_s110
                                 + 0.955623 w_k1_KO_s110
                                 + 0.909906 w_k1_MAT_s110
                                 + 0.919768 w_k1_MSFT_s110
                                 + 0.896025 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.851515 w_k1_ADM_s110
                                 + 0.699467 w_k1_AXP_s110
                                 + 0.791911 w_k1_DIS_s110
                                 + 0.925972 w_k1_FDX_s110
                                 + 0.728474 w_k1_IBM_s110
                                 + 0.90223 w_k1_INTC_s110
                                 + 0.652921 w_k1_KO_s110
                                 + 0.601762 w_k1_MAT_s110
                                 + 0.892638 w_k1_MSFT_s110
                                 + 0.82911 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.03317 w_k1_ADM_s110
                                 + 0.771011 w_k1_AXP_s110
                                 + 0.668562 w_k1_DIS_s110
                                 + 0.775571 w_k1_FDX_s110
                                 + 0.770544 w_k1_IBM_s110
                                 + 0.819681 w_k1_INTC_s110
                                 + 0.71268 w_k1_KO_s110
                                 + 0.710816 w_k1_MAT_s110
                                 + 0.86465 w_k1_MSFT_s110
                                 + 0.818587 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.31368 w_k1_ADM_s110
                                 + 1.4164 w_k1_AXP_s110 + 1.36902 w_k1_DIS_s110
                                 + 1.13252 w_k1_FDX_s110
                                 + 1.50255 w_k1_IBM_s110
                                 + 0.962054 w_k1_INTC_s110
                                 + 1.24151 w_k1_KO_s110 + 1.32922 w_k1_MAT_s110
                                 + 1.04209 w_k1_MSFT_s110
                                 + 1.05332 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.17781 w_k1_ADM_s110
                                 + 1.17608 w_k1_AXP_s110
                                 + 0.83019 w_k1_DIS_s110
                                 + 1.17421 w_k1_FDX_s110
                                 + 0.754747 w_k1_IBM_s110
                                 + 1.09287 w_k1_INTC_s110
                                 + 0.743316 w_k1_KO_s110
                                 + 0.799532 w_k1_MAT_s110
                                 + 1.07752 w_k1_MSFT_s110
                                 + 0.967768 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.798415 w_k1_ADM_s110
                                 + 0.706109 w_k1_AXP_s110
                                 + 0.576714 w_k1_DIS_s110
                                 + 0.724835 w_k1_FDX_s110
                                 + 1.07667 w_k1_IBM_s110
                                 + 0.94508 w_k1_INTC_s110
                                 + 0.741417 w_k1_KO_s110
                                 + 1.07141 w_k1_MAT_s110
                                 + 0.755574 w_k1_MSFT_s110
                                 + 0.854704 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.29065 w_k1_ADM_s110
                                 + 1.02782 w_k1_AXP_s110
                                 + 1.19077 w_k1_DIS_s110
                                 + 1.14729 w_k1_FDX_s110
                                 + 0.924258 w_k1_IBM_s110
                                 + 0.955205 w_k1_INTC_s110
                                 + 1.04383 w_k1_KO_s110
                                 + 0.868328 w_k1_MAT_s110
                                 + 0.998215 w_k1_MSFT_s110
                                 + 1.04683 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.783173 w_k1_ADM_s111
                                 + 0.924931 w_k1_AXP_s111
                                 + 1.03268 w_k1_DIS_s111
                                 + 0.988218 w_k1_FDX_s111
                                 + 1.3295 w_k1_IBM_s111
                                 + 1.01588 w_k1_INTC_s111
                                 + 1.09637 w_k1_KO_s111 + 1.47264 w_k1_MAT_s111
                                 + 1.13331 w_k1_MSFT_s111
                                 + 1.04652 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.06095 w_k1_ADM_s111
                                 + 0.867825 w_k1_AXP_s111
                                 + 1.00456 w_k1_DIS_s111
                                 + 1.06922 w_k1_FDX_s111
                                 + 0.780082 w_k1_IBM_s111
                                 + 1.10219 w_k1_INTC_s111
                                 + 1.02385 w_k1_KO_s111
                                 + 0.856825 w_k1_MAT_s111
                                 + 1.18757 w_k1_MSFT_s111
                                 + 0.895774 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.19437 w_k1_ADM_s111
                                 + 1.42823 w_k1_AXP_s111
                                 + 1.50927 w_k1_DIS_s111
                                 + 1.41356 w_k1_FDX_s111
                                 + 0.998111 w_k1_IBM_s111
                                 + 1.24262 w_k1_INTC_s111
                                 + 1.17097 w_k1_KO_s111 + 1.10775 w_k1_MAT_s111
                                 + 1.25706 w_k1_MSFT_s111
                                 + 1.20175 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.865869 w_k1_ADM_s111
                                 + 1.02671 w_k1_AXP_s111
                                 + 1.57465 w_k1_DIS_s111
                                 + 1.22586 w_k1_FDX_s111
                                 + 1.23195 w_k1_IBM_s111
                                 + 1.11902 w_k1_INTC_s111
                                 + 1.37564 w_k1_KO_s111 + 1.5284 w_k1_MAT_s111
                                 + 1.31807 w_k1_MSFT_s111
                                 + 1.16174 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.95649 w_k1_ADM_s111
                                 + 0.876422 w_k1_AXP_s111
                                 + 1.15117 w_k1_DIS_s111 + 1.0239 w_k1_FDX_s111
                                 + 0.986408 w_k1_IBM_s111
                                 + 0.83676 w_k1_INTC_s111
                                 + 1.08353 w_k1_KO_s111 + 1.06613 w_k1_MAT_s111
                                 + 1.10499 w_k1_MSFT_s111
                                 + 0.963379 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.905233 w_k1_ADM_s111
                                 + 0.582723 w_k1_AXP_s111
                                 + 0.613117 w_k1_DIS_s111
                                 + 0.728029 w_k1_FDX_s111
                                 + 0.701134 w_k1_IBM_s111
                                 + 0.688006 w_k1_INTC_s111
                                 + 0.704113 w_k1_KO_s111
                                 + 0.627036 w_k1_MAT_s111
                                 + 0.7336 w_k1_MSFT_s111
                                 + 0.826926 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.988699 w_k1_ADM_s111
                                 + 1.42813 w_k1_AXP_s111
                                 + 1.83725 w_k1_DIS_s111
                                 + 1.37157 w_k1_FDX_s111
                                 + 1.54763 w_k1_IBM_s111
                                 + 1.57592 w_k1_INTC_s111
                                 + 1.78994 w_k1_KO_s111 + 1.79897 w_k1_MAT_s111
                                 + 1.54803 w_k1_MSFT_s111
                                 + 1.22796 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.696558 w_k1_ADM_s111
                                 + 0.639925 w_k1_AXP_s111
                                 + 0.91408 w_k1_DIS_s111
                                 + 0.743317 w_k1_FDX_s111
                                 + 1.00716 w_k1_IBM_s111
                                 + 0.905362 w_k1_INTC_s111
                                 + 0.933369 w_k1_KO_s111
                                 + 0.857882 w_k1_MAT_s111
                                 + 0.881193 w_k1_MSFT_s111
                                 + 0.855333 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.02392 w_k1_ADM_s112
                                 + 1.25642 w_k1_AXP_s112
                                 + 1.04069 w_k1_DIS_s112
                                 + 1.15223 w_k1_FDX_s112
                                 + 0.780981 w_k1_IBM_s112
                                 + 1.1852 w_k1_INTC_s112
                                 + 0.983789 w_k1_KO_s112
                                 + 1.39382 w_k1_MAT_s112
                                 + 1.02284 w_k1_MSFT_s112
                                 + 0.925197 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17147 w_k1_ADM_s112
                                 + 1.66204 w_k1_AXP_s112
                                 + 1.39016 w_k1_DIS_s112
                                 + 1.34945 w_k1_FDX_s112
                                 + 1.07044 w_k1_IBM_s112
                                 + 1.2735 w_k1_INTC_s112 + 1.07701 w_k1_KO_s112
                                 + 1.61778 w_k1_MAT_s112 + 1.249 w_k1_MSFT_s112
                                 + 1.00474 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.9515 w_k1_ADM_s112
                                 + 0.821046 w_k1_AXP_s112
                                 + 1.02387 w_k1_DIS_s112
                                 + 0.984738 w_k1_FDX_s112
                                 + 0.773217 w_k1_IBM_s112
                                 + 1.11511 w_k1_INTC_s112
                                 + 0.908782 w_k1_KO_s112
                                 + 0.781387 w_k1_MAT_s112
                                 + 1.13362 w_k1_MSFT_s112
                                 + 0.875653 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.32263 w_k1_ADM_s112
                                 + 1.11356 w_k1_AXP_s112
                                 + 1.71013 w_k1_DIS_s112
                                 + 1.25804 w_k1_FDX_s112
                                 + 1.12523 w_k1_IBM_s112
                                 + 1.11805 w_k1_INTC_s112
                                 + 1.28608 w_k1_KO_s112 + 1.77526 w_k1_MAT_s112
                                 + 1.42963 w_k1_MSFT_s112 + 1.0988 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.821593 w_k1_ADM_s112
                                 + 0.928663 w_k1_AXP_s112
                                 + 0.783888 w_k1_DIS_s112
                                 + 0.956935 w_k1_FDX_s112
                                 + 0.757852 w_k1_IBM_s112
                                 + 1.09152 w_k1_INTC_s112
                                 + 0.76952 w_k1_KO_s112 + 1.06574 w_k1_MAT_s112
                                 + 0.997797 w_k1_MSFT_s112
                                 + 0.81747 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.866979 w_k1_ADM_s112
                                 + 0.988271 w_k1_AXP_s112
                                 + 1.01385 w_k1_DIS_s112
                                 + 0.98061 w_k1_FDX_s112
                                 + 1.26505 w_k1_IBM_s112
                                 + 1.24968 w_k1_INTC_s112
                                 + 1.04764 w_k1_KO_s112 + 1.22586 w_k1_MAT_s112
                                 + 1.06281 w_k1_MSFT_s112
                                 + 0.95668 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.665275 w_k1_ADM_s112
                                 + 0.721209 w_k1_AXP_s112
                                 + 0.73996 w_k1_DIS_s112
                                 + 0.738944 w_k1_FDX_s112
                                 + 0.819221 w_k1_IBM_s112
                                 + 0.89498 w_k1_INTC_s112
                                 + 0.773153 w_k1_KO_s112
                                 + 0.984724 w_k1_MAT_s112
                                 + 0.810244 w_k1_MSFT_s112
                                 + 0.769591 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.39569 w_k1_ADM_s112
                                 + 1.32822 w_k1_AXP_s112
                                 + 1.44217 w_k1_DIS_s112
                                 + 1.08419 w_k1_FDX_s112
                                 + 1.59977 w_k1_IBM_s112
                                 + 1.12716 w_k1_INTC_s112
                                 + 1.45685 w_k1_KO_s112 + 1.92315 w_k1_MAT_s112
                                 + 1.15842 w_k1_MSFT_s112
                                 + 1.13079 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.879747 w_k1_ADM_s113
                                 + 0.718088 w_k1_AXP_s113
                                 + 0.921669 w_k1_DIS_s113
                                 + 0.915621 w_k1_FDX_s113
                                 + 0.775375 w_k1_IBM_s113
                                 + 0.766978 w_k1_INTC_s113
                                 + 0.849745 w_k1_KO_s113
                                 + 0.737108 w_k1_MAT_s113
                                 + 0.979304 w_k1_MSFT_s113
                                 + 0.851109 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.934468 w_k1_ADM_s113
                                 + 0.763629 w_k1_AXP_s113
                                 + 1.19413 w_k1_DIS_s113
                                 + 0.988226 w_k1_FDX_s113
                                 + 1.21044 w_k1_IBM_s113
                                 + 1.06409 w_k1_INTC_s113
                                 + 1.11636 w_k1_KO_s113 + 1.11397 w_k1_MAT_s113
                                 + 1.25196 w_k1_MSFT_s113
                                 + 1.06689 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.982856 w_k1_ADM_s113
                                 + 0.907011 w_k1_AXP_s113
                                 + 1.0278 w_k1_DIS_s113 + 1.03557 w_k1_FDX_s113
                                 + 0.910251 w_k1_IBM_s113
                                 + 0.924849 w_k1_INTC_s113
                                 + 0.963619 w_k1_KO_s113
                                 + 0.845144 w_k1_MAT_s113
                                 + 1.03292 w_k1_MSFT_s113
                                 + 0.892728 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.893113 w_k1_ADM_s113
                                 + 0.89585 w_k1_AXP_s113
                                 + 0.816702 w_k1_DIS_s113
                                 + 0.80327 w_k1_FDX_s113
                                 + 1.26572 w_k1_IBM_s113
                                 + 0.864702 w_k1_INTC_s113
                                 + 0.878225 w_k1_KO_s113
                                 + 1.16547 w_k1_MAT_s113
                                 + 0.926363 w_k1_MSFT_s113
                                 + 0.862576 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.994296 w_k1_ADM_s113
                                 + 1.33943 w_k1_AXP_s113
                                 + 1.59259 w_k1_DIS_s113
                                 + 1.28133 w_k1_FDX_s113
                                 + 1.13358 w_k1_IBM_s113
                                 + 1.0692 w_k1_INTC_s113 + 1.25108 w_k1_KO_s113
                                 + 1.71982 w_k1_MAT_s113
                                 + 1.32273 w_k1_MSFT_s113
                                 + 1.22702 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.21985 w_k1_ADM_s113
                                 + 1.17398 w_k1_AXP_s113
                                 + 1.13241 w_k1_DIS_s113
                                 + 0.996882 w_k1_FDX_s113
                                 + 1.21893 w_k1_IBM_s113
                                 + 0.86372 w_k1_INTC_s113
                                 + 1.02799 w_k1_KO_s113 + 1.21241 w_k1_MAT_s113
                                 + 1.02025 w_k1_MSFT_s113
                                 + 1.12833 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.04281 w_k1_ADM_s113
                                 + 1.20173 w_k1_AXP_s113
                                 + 0.892426 w_k1_DIS_s113
                                 + 1.12563 w_k1_FDX_s113
                                 + 0.67653 w_k1_IBM_s113
                                 + 0.765415 w_k1_INTC_s113
                                 + 0.727305 w_k1_KO_s113
                                 + 1.23636 w_k1_MAT_s113
                                 + 0.903154 w_k1_MSFT_s113
                                 + 1.05491 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.846907 w_k1_ADM_s113
                                 + 0.668122 w_k1_AXP_s113
                                 + 0.697321 w_k1_DIS_s113
                                 + 0.741762 w_k1_FDX_s113
                                 + 0.718169 w_k1_IBM_s113
                                 + 0.595915 w_k1_INTC_s113
                                 + 0.692765 w_k1_KO_s113
                                 + 0.830897 w_k1_MAT_s113
                                 + 0.805803 w_k1_MSFT_s113
                                 + 0.837241 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.737402 w_k1_ADM_s114
                                 + 0.718354 w_k1_AXP_s114
                                 + 0.661646 w_k1_DIS_s114
                                 + 0.831048 w_k1_FDX_s114
                                 + 0.76856 w_k1_IBM_s114
                                 + 0.805882 w_k1_INTC_s114
                                 + 0.689059 w_k1_KO_s114
                                 + 0.551183 w_k1_MAT_s114
                                 + 0.739799 w_k1_MSFT_s114
                                 + 0.79731 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.28092 w_k1_ADM_s114
                                 + 1.02629 w_k1_AXP_s114
                                 + 1.44714 w_k1_DIS_s114 + 1.2613 w_k1_FDX_s114
                                 + 1.20025 w_k1_IBM_s114
                                 + 0.928296 w_k1_INTC_s114
                                 + 1.26756 w_k1_KO_s114 + 0.9226 w_k1_MAT_s114
                                 + 1.27242 w_k1_MSFT_s114
                                 + 1.13805 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.31532 w_k1_ADM_s114
                                 + 1.4161 w_k1_AXP_s114 + 1.48858 w_k1_DIS_s114
                                 + 1.29282 w_k1_FDX_s114
                                 + 1.77494 w_k1_IBM_s114 + 1.226 w_k1_INTC_s114
                                 + 1.48213 w_k1_KO_s114 + 1.55794 w_k1_MAT_s114
                                 + 1.26005 w_k1_MSFT_s114
                                 + 1.24578 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.1828 w_k1_ADM_s114
                                 + 1.0086 w_k1_AXP_s114
                                 + 0.946959 w_k1_DIS_s114
                                 + 1.04648 w_k1_FDX_s114
                                 + 0.796638 w_k1_IBM_s114
                                 + 0.845963 w_k1_INTC_s114
                                 + 0.794513 w_k1_KO_s114
                                 + 0.730007 w_k1_MAT_s114
                                 + 0.901688 w_k1_MSFT_s114
                                 + 0.922943 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.18878 w_k1_ADM_s114
                                 + 1.09099 w_k1_AXP_s114
                                 + 0.917168 w_k1_DIS_s114
                                 + 0.988931 w_k1_FDX_s114
                                 + 1.14954 w_k1_IBM_s114
                                 + 0.890084 w_k1_INTC_s114
                                 + 0.963139 w_k1_KO_s114
                                 + 0.956945 w_k1_MAT_s114
                                 + 0.935265 w_k1_MSFT_s114
                                 + 0.951536 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.961801 w_k1_ADM_s114
                                 + 0.804944 w_k1_AXP_s114
                                 + 0.958771 w_k1_DIS_s114
                                 + 0.929343 w_k1_FDX_s114
                                 + 1.36613 w_k1_IBM_s114
                                 + 1.01357 w_k1_INTC_s114
                                 + 1.25679 w_k1_KO_s114
                                 + 0.976236 w_k1_MAT_s114
                                 + 1.00449 w_k1_MSFT_s114
                                 + 1.01958 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.17379 w_k1_ADM_s114
                                 + 1.5351 w_k1_AXP_s114 + 1.52521 w_k1_DIS_s114
                                 + 1.4648 w_k1_FDX_s114 + 1.08361 w_k1_IBM_s114
                                 + 1.26945 w_k1_INTC_s114
                                 + 1.02453 w_k1_KO_s114 + 1.1407 w_k1_MAT_s114
                                 + 1.15661 w_k1_MSFT_s114
                                 + 1.16108 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.905633 w_k1_ADM_s114
                                 + 0.646658 w_k1_AXP_s114
                                 + 1.07691 w_k1_DIS_s114
                                 + 0.902881 w_k1_FDX_s114
                                 + 0.951514 w_k1_IBM_s114
                                 + 0.831068 w_k1_INTC_s114
                                 + 1.01059 w_k1_KO_s114
                                 + 0.792064 w_k1_MAT_s114
                                 + 1.03662 w_k1_MSFT_s114
                                 + 1.12094 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.989146 w_k1_ADM_s115
                                 + 0.747835 w_k1_AXP_s115
                                 + 0.896496 w_k1_DIS_s115
                                 + 0.788311 w_k1_FDX_s115
                                 + 1.20218 w_k1_IBM_s115
                                 + 0.955258 w_k1_INTC_s115
                                 + 0.992194 w_k1_KO_s115
                                 + 0.980003 w_k1_MAT_s115
                                 + 0.851114 w_k1_MSFT_s115
                                 + 0.893661 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.953584 w_k1_ADM_s115
                                 + 1.28243 w_k1_AXP_s115
                                 + 1.05161 w_k1_DIS_s115
                                 + 1.06867 w_k1_FDX_s115
                                 + 1.37409 w_k1_IBM_s115
                                 + 1.04574 w_k1_INTC_s115
                                 + 0.973788 w_k1_KO_s115
                                 + 1.77231 w_k1_MAT_s115
                                 + 0.954822 w_k1_MSFT_s115
                                 + 0.968826 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.960584 w_k1_ADM_s115
                                 + 0.917765 w_k1_AXP_s115
                                 + 1.11352 w_k1_DIS_s115 + 1.0506 w_k1_FDX_s115
                                 + 0.804562 w_k1_IBM_s115
                                 + 1.12298 w_k1_INTC_s115
                                 + 1.04467 w_k1_KO_s115 + 1.02714 w_k1_MAT_s115
                                 + 1.26158 w_k1_MSFT_s115
                                 + 1.05776 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.901821 w_k1_ADM_s115
                                 + 1.75991 w_k1_AXP_s115
                                 + 1.60819 w_k1_DIS_s115
                                 + 1.48308 w_k1_FDX_s115
                                 + 1.06155 w_k1_IBM_s115
                                 + 1.43219 w_k1_INTC_s115
                                 + 1.42562 w_k1_KO_s115 + 1.76951 w_k1_MAT_s115
                                 + 1.43947 w_k1_MSFT_s115
                                 + 1.31885 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.642022 w_k1_ADM_s115
                                 + 0.704486 w_k1_AXP_s115
                                 + 0.594783 w_k1_DIS_s115
                                 + 0.704486 w_k1_FDX_s115
                                 + 0.868698 w_k1_IBM_s115
                                 + 0.947204 w_k1_INTC_s115
                                 + 0.767462 w_k1_KO_s115
                                 + 0.844177 w_k1_MAT_s115
                                 + 0.739965 w_k1_MSFT_s115
                                 + 0.771228 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.821298 w_k1_ADM_s115
                                 + 0.941758 w_k1_AXP_s115
                                 + 0.848834 w_k1_DIS_s115
                                 + 0.95508 w_k1_FDX_s115
                                 + 0.976348 w_k1_IBM_s115
                                 + 1.09426 w_k1_INTC_s115
                                 + 0.973256 w_k1_KO_s115
                                 + 1.02117 w_k1_MAT_s115
                                 + 1.01027 w_k1_MSFT_s115
                                 + 0.847114 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.662871 w_k1_ADM_s115
                                 + 0.997685 w_k1_AXP_s115
                                 + 0.682463 w_k1_DIS_s115
                                 + 0.857799 w_k1_FDX_s115
                                 + 0.710546 w_k1_IBM_s115
                                 + 0.944338 w_k1_INTC_s115
                                 + 0.832439 w_k1_KO_s115
                                 + 0.963748 w_k1_MAT_s115
                                 + 0.939319 w_k1_MSFT_s115
                                 + 0.944384 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.847381 w_k1_ADM_s115
                                 + 1.23161 w_k1_AXP_s115
                                 + 1.55754 w_k1_DIS_s115
                                 + 1.18378 w_k1_FDX_s115
                                 + 1.52275 w_k1_IBM_s115
                                 + 1.13108 w_k1_INTC_s115
                                 + 1.33453 w_k1_KO_s115 + 1.61659 w_k1_MAT_s115
                                 + 1.24223 w_k1_MSFT_s115
                                 + 1.01798 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.14805 w_k1_ADM_s116
                                 + 0.881274 w_k1_AXP_s116
                                 + 0.819222 w_k1_DIS_s116
                                 + 0.940659 w_k1_FDX_s116
                                 + 0.894966 w_k1_IBM_s116
                                 + 1.11352 w_k1_INTC_s116
                                 + 0.931782 w_k1_KO_s116
                                 + 0.817281 w_k1_MAT_s116
                                 + 0.932211 w_k1_MSFT_s116 + 0.957 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.34502 w_k1_ADM_s116
                                 + 1.15789 w_k1_AXP_s116
                                 + 1.15176 w_k1_DIS_s116
                                 + 1.23443 w_k1_FDX_s116
                                 + 0.873356 w_k1_IBM_s116
                                 + 1.305 w_k1_INTC_s116 + 1.04645 w_k1_KO_s116
                                 + 0.84139 w_k1_MAT_s116
                                 + 1.07172 w_k1_MSFT_s116
                                 + 1.10391 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.39732 w_k1_ADM_s116
                                 + 1.63039 w_k1_AXP_s116
                                 + 1.88526 w_k1_DIS_s116
                                 + 1.41352 w_k1_FDX_s116
                                 + 1.65944 w_k1_IBM_s116
                                 + 1.60224 w_k1_INTC_s116
                                 + 1.64611 w_k1_KO_s116 + 1.98273 w_k1_MAT_s116
                                 + 1.47271 w_k1_MSFT_s116
                                 + 1.34427 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.08619 w_k1_ADM_s116
                                 + 1.05433 w_k1_AXP_s116
                                 + 0.979276 w_k1_DIS_s116
                                 + 0.938139 w_k1_FDX_s116
                                 + 1.0243 w_k1_IBM_s116
                                 + 0.997776 w_k1_INTC_s116
                                 + 1.19601 w_k1_KO_s116 + 1.29118 w_k1_MAT_s116
                                 + 1.01654 w_k1_MSFT_s116
                                 + 1.11137 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.863131 w_k1_ADM_s116
                                 + 0.650578 w_k1_AXP_s116
                                 + 0.701839 w_k1_DIS_s116
                                 + 0.861479 w_k1_FDX_s116
                                 + 0.582424 w_k1_IBM_s116
                                 + 0.885033 w_k1_INTC_s116
                                 + 0.781947 w_k1_KO_s116
                                 + 0.650146 w_k1_MAT_s116
                                 + 0.849645 w_k1_MSFT_s116
                                 + 0.834412 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.827566 w_k1_ADM_s116
                                 + 0.839199 w_k1_AXP_s116
                                 + 0.65507 w_k1_DIS_s116
                                 + 0.751861 w_k1_FDX_s116
                                 + 1.15358 w_k1_IBM_s116
                                 + 0.805716 w_k1_INTC_s116
                                 + 0.794421 w_k1_KO_s116
                                 + 0.823162 w_k1_MAT_s116
                                 + 0.722848 w_k1_MSFT_s116
                                 + 0.872416 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.09133 w_k1_ADM_s116
                                 + 0.997602 w_k1_AXP_s116
                                 + 0.961561 w_k1_DIS_s116
                                 + 0.866064 w_k1_FDX_s116
                                 + 1.58241 w_k1_IBM_s116
                                 + 0.916781 w_k1_INTC_s116 + 1.138 w_k1_KO_s116
                                 + 1.31899 w_k1_MAT_s116
                                 + 0.829954 w_k1_MSFT_s116
                                 + 1.04684 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.57325 w_k1_ADM_s116
                                 + 1.18083 w_k1_AXP_s116
                                 + 1.16748 w_k1_DIS_s116
                                 + 1.12777 w_k1_FDX_s116
                                 + 1.37704 w_k1_IBM_s116
                                 + 1.1687 w_k1_INTC_s116 + 1.24244 w_k1_KO_s116
                                 + 1.50371 w_k1_MAT_s116
                                 + 1.16451 w_k1_MSFT_s116
                                 + 1.11281 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.967178 w_k1_ADM_s117
                                 + 0.911621 w_k1_AXP_s117
                                 + 0.885456 w_k1_DIS_s117
                                 + 0.96089 w_k1_FDX_s117
                                 + 1.04694 w_k1_IBM_s117
                                 + 0.878947 w_k1_INTC_s117
                                 + 0.994682 w_k1_KO_s117
                                 + 1.37163 w_k1_MAT_s117
                                 + 1.01914 w_k1_MSFT_s117
                                 + 1.02856 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.616 w_k1_ADM_s117
                                 + 1.10821 w_k1_AXP_s117
                                 + 1.67079 w_k1_DIS_s117
                                 + 1.47087 w_k1_FDX_s117
                                 + 1.12983 w_k1_IBM_s117
                                 + 1.20583 w_k1_INTC_s117
                                 + 1.27753 w_k1_KO_s117
                                 + 0.964517 w_k1_MAT_s117
                                 + 1.40836 w_k1_MSFT_s117
                                 + 1.05065 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.95531 w_k1_ADM_s117
                                 + 0.748885 w_k1_AXP_s117
                                 + 0.722245 w_k1_DIS_s117
                                 + 0.823886 w_k1_FDX_s117
                                 + 0.735047 w_k1_IBM_s117
                                 + 0.836975 w_k1_INTC_s117
                                 + 0.645901 w_k1_KO_s117
                                 + 0.852343 w_k1_MAT_s117
                                 + 0.934397 w_k1_MSFT_s117
                                 + 0.791503 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.34934 w_k1_ADM_s117
                                 + 1.18721 w_k1_AXP_s117
                                 + 0.918808 w_k1_DIS_s117
                                 + 1.22158 w_k1_FDX_s117
                                 + 0.881695 w_k1_IBM_s117
                                 + 1.09477 w_k1_INTC_s117
                                 + 0.936247 w_k1_KO_s117
                                 + 1.36593 w_k1_MAT_s117
                                 + 1.14626 w_k1_MSFT_s117 + 0.934 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.878463 w_k1_ADM_s117
                                 + 0.811816 w_k1_AXP_s117
                                 + 0.981839 w_k1_DIS_s117
                                 + 0.871945 w_k1_FDX_s117
                                 + 1.27184 w_k1_IBM_s117
                                 + 0.695066 w_k1_INTC_s117
                                 + 1.08787 w_k1_KO_s117 + 1.40313 w_k1_MAT_s117
                                 + 0.973701 w_k1_MSFT_s117
                                 + 0.959843 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.5604 w_k1_ADM_s117
                                 + 1.436 w_k1_AXP_s117 + 1.4469 w_k1_DIS_s117
                                 + 1.2579 w_k1_FDX_s117 + 1.75753 w_k1_IBM_s117
                                 + 1.3753 w_k1_INTC_s117 + 1.58014 w_k1_KO_s117
                                 + 2.10882 w_k1_MAT_s117
                                 + 1.13731 w_k1_MSFT_s117
                                 + 1.10275 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.30379 w_k1_ADM_s117
                                 + 0.884598 w_k1_AXP_s117
                                 + 1.30357 w_k1_DIS_s117
                                 + 1.15713 w_k1_FDX_s117
                                 + 1.00771 w_k1_IBM_s117
                                 + 1.28282 w_k1_INTC_s117 + 1.1354 w_k1_KO_s117
                                 + 0.853616 w_k1_MAT_s117
                                 + 1.1472 w_k1_MSFT_s117 + 1.04093 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.947807 w_k1_ADM_s117
                                 + 1.40924 w_k1_AXP_s117
                                 + 1.11445 w_k1_DIS_s117
                                 + 1.12809 w_k1_FDX_s117
                                 + 1.33542 w_k1_IBM_s117
                                 + 1.0144 w_k1_INTC_s117 + 1.12169 w_k1_KO_s117
                                 + 1.85928 w_k1_MAT_s117
                                 + 1.10424 w_k1_MSFT_s117
                                 + 1.08361 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0130001139675154 w_k1_ADM_s000
                                 + 0.0134050860386105 w_k1_ADM_s001
                                 + 0.0217994240236169 w_k1_ADM_s002
                                 + 0.0131919969769699 w_k1_ADM_s003
                                 + 0.0279014794336751 w_k1_ADM_s004
                                 + 0.0203282966187962 w_k1_ADM_s005
                                 + 0.018167072395957 w_k1_ADM_s006
                                 + 0.0296032938329806 w_k1_ADM_s007
                                 + 0.0330186531458603 w_k1_ADM_s010
                                 + 0.0516872567302261 w_k1_ADM_s011
                                 + 0.0331008802122213 w_k1_ADM_s012
                                 + 0.031367503449158 w_k1_ADM_s013
                                 + 0.0395464992358266 w_k1_ADM_s014
                                 + 0.0300549724038386 w_k1_ADM_s015
                                 + 0.0262794780576248 w_k1_ADM_s016
                                 + 0.0366145027222058 w_k1_ADM_s017
                                 + 0.0293739256807019 w_k1_ADM_s100
                                 + 0.0409994797801605 w_k1_ADM_s101
                                 + 0.0275538119540007 w_k1_ADM_s102
                                 + 0.0494024443220679 w_k1_ADM_s103
                                 + 0.0482005593416958 w_k1_ADM_s104
                                 + 0.0474016006750962 w_k1_ADM_s105
                                 + 0.0318065837968844 w_k1_ADM_s106
                                 + 0.0408742483105522 w_k1_ADM_s107
                                 + 0.026159498693641 w_k1_ADM_s110
                                 + 0.0120500385700245 w_k1_ADM_s111
                                 + 0.0543741196867535 w_k1_ADM_s112
                                 + 0.0566426536315074 w_k1_ADM_s113
                                 + 0.0300088948990269 w_k1_ADM_s114
                                 + 0.0125180651854799 w_k1_ADM_s115
                                 + 0.0231516688547665 w_k1_ADM_s116
                                 + 0.038883919174371 w_k1_ADM_s117
                                 + 0.0129387464534855 w_k1_AXP_s000
                                 + 0.014171508300357 w_k1_AXP_s001
                                 + 0.0230983110426125 w_k1_AXP_s002
                                 + 0.0118143689647011 w_k1_AXP_s003
                                 + 0.0286413711215459 w_k1_AXP_s004
                                 + 0.0207163696721707 w_k1_AXP_s005
                                 + 0.0184747167661203 w_k1_AXP_s006
                                 + 0.0288379769397129 w_k1_AXP_s007
                                 + 0.0390231132882627 w_k1_AXP_s010
                                 + 0.0581789880863976 w_k1_AXP_s011
                                 + 0.0345702337854038 w_k1_AXP_s012
                                 + 0.0366018146247677 w_k1_AXP_s013
                                 + 0.0386367035874944 w_k1_AXP_s014
                                 + 0.0329684296999882 w_k1_AXP_s015
                                 + 0.0273930271848215 w_k1_AXP_s016
                                 + 0.0401883349169598 w_k1_AXP_s017
                                 + 0.0311125392521452 w_k1_AXP_s100
                                 + 0.0428733653318542 w_k1_AXP_s101
                                 + 0.0266675846420266 w_k1_AXP_s102
                                 + 0.0471646583084023 w_k1_AXP_s103
                                 + 0.0468065854799608 w_k1_AXP_s104
                                 + 0.0477286167932825 w_k1_AXP_s105
                                 + 0.0352977572137133 w_k1_AXP_s106
                                 + 0.0368665263070925 w_k1_AXP_s107
                                 + 0.0243848013181246 w_k1_AXP_s110
                                 + 0.0117561490445063 w_k1_AXP_s111
                                 + 0.0579775827306262 w_k1_AXP_s112
                                 + 0.0576424457230314 w_k1_AXP_s113
                                 + 0.0278470266784148 w_k1_AXP_s114
                                 + 0.0157629200601521 w_k1_AXP_s115
                                 + 0.0201337171868459 w_k1_AXP_s116
                                 + 0.0346164708846166 w_k1_AXP_s117
                                 + 0.0142455792247923 w_k1_DIS_s000
                                 + 0.011950076008499 w_k1_DIS_s001
                                 + 0.021780485995257 w_k1_DIS_s002
                                 + 0.0110996633800593 w_k1_DIS_s003
                                 + 0.030555647996438 w_k1_DIS_s004
                                 + 0.0236849895018559 w_k1_DIS_s005
                                 + 0.0196053417662014 w_k1_DIS_s006
                                 + 0.0288754964985581 w_k1_DIS_s007
                                 + 0.0375079126529595 w_k1_DIS_s010
                                 + 0.0551910339026089 w_k1_DIS_s011
                                 + 0.0335440890398275 w_k1_DIS_s012
                                 + 0.0367052908573282 w_k1_DIS_s013
                                 + 0.0430692692391996 w_k1_DIS_s014
                                 + 0.0358984803110657 w_k1_DIS_s015
                                 + 0.0317967579426671 w_k1_DIS_s016
                                 + 0.0369479861087276 w_k1_DIS_s017
                                 + 0.0291110480761451 w_k1_DIS_s100
                                 + 0.0471115790280865 w_k1_DIS_s101
                                 + 0.0247490806001169 w_k1_DIS_s102
                                 + 0.0449972157081471 w_k1_DIS_s103
                                 + 0.043332732541305 w_k1_DIS_s104
                                 + 0.0510181450012419 w_k1_DIS_s105
                                 + 0.0359557649305028 w_k1_DIS_s106
                                 + 0.0382138855013916 w_k1_DIS_s107
                                 + 0.0231066105069925 w_k1_DIS_s110
                                 + 0.0146521114179621 w_k1_DIS_s111
                                 + 0.0604429842688624 w_k1_DIS_s112
                                 + 0.0642666104445251 w_k1_DIS_s113
                                 + 0.0293491257647522 w_k1_DIS_s114
                                 + 0.0153503643986057 w_k1_DIS_s115
                                 + 0.0196954172572551 w_k1_DIS_s116
                                 + 0.0353661493145485 w_k1_DIS_s117
                                 + 0.0140102926888366 w_k1_FDX_s000
                                 + 0.011404215127537 w_k1_FDX_s001
                                 + 0.0211638732373517 w_k1_FDX_s002
                                 + 0.0116620478911583 w_k1_FDX_s003
                                 + 0.0282858270154268 w_k1_FDX_s004
                                 + 0.022776795745659 w_k1_FDX_s005
                                 + 0.0194205490842828 w_k1_FDX_s006
                                 + 0.0276222685830312 w_k1_FDX_s007
                                 + 0.0388128850603074 w_k1_FDX_s010
                                 + 0.0560171255762373 w_k1_FDX_s011
                                 + 0.0320224640824331 w_k1_FDX_s012
                                 + 0.0343917772944825 w_k1_FDX_s013
                                 + 0.0408789492739018 w_k1_FDX_s014
                                 + 0.0350078559900712 w_k1_FDX_s015
                                 + 0.0285482914775402 w_k1_FDX_s016
                                 + 0.0364792165306089 w_k1_FDX_s017
                                 + 0.0270229146368568 w_k1_FDX_s100
                                 + 0.0466340728431364 w_k1_FDX_s101
                                 + 0.0262153544299363 w_k1_FDX_s102
                                 + 0.0450600845315236 w_k1_FDX_s103
                                 + 0.0424091144321677 w_k1_FDX_s104
                                 + 0.047096083781817 w_k1_FDX_s105
                                 + 0.0336813330214156 w_k1_FDX_s106
                                 + 0.0391317704477557 w_k1_FDX_s107
                                 + 0.0239650071136347 w_k1_FDX_s110
                                 + 0.0133829249516066 w_k1_FDX_s111
                                 + 0.057490490264423 w_k1_FDX_s112
                                 + 0.058447094549578 w_k1_FDX_s113
                                 + 0.0294262485424847 w_k1_FDX_s114
                                 + 0.0150011197299479 w_k1_FDX_s115
                                 + 0.0199865838312101 w_k1_FDX_s116
                                 + 0.0364206676322376 w_k1_FDX_s117
                                 + 0.0130282280053548 w_k1_IBM_s000
                                 + 0.011779605369317 w_k1_IBM_s001
                                 + 0.0210129908810041 w_k1_IBM_s002
                                 + 0.0114202249919042 w_k1_IBM_s003
                                 + 0.0281731884616449 w_k1_IBM_s004
                                 + 0.0223832062289013 w_k1_IBM_s005
                                 + 0.0188821998752592 w_k1_IBM_s006
                                 + 0.0298087924870276 w_k1_IBM_s007
                                 + 0.0375637590553062 w_k1_IBM_s010
                                 + 0.0571135855027162 w_k1_IBM_s011
                                 + 0.0349687387728081 w_k1_IBM_s012
                                 + 0.0354855152213491 w_k1_IBM_s013
                                 + 0.0406504428509645 w_k1_IBM_s014
                                 + 0.0362555785526545 w_k1_IBM_s015
                                 + 0.0291778581494397 w_k1_IBM_s016
                                 + 0.0344688690497428 w_k1_IBM_s017
                                 + 0.027845331955871 w_k1_IBM_s100
                                 + 0.0451772205660409 w_k1_IBM_s101
                                 + 0.0256658503733147 w_k1_IBM_s102
                                 + 0.0444203660925459 w_k1_IBM_s103
                                 + 0.0443056104515277 w_k1_IBM_s104
                                 + 0.0490950449920269 w_k1_IBM_s105
                                 + 0.0330323849107047 w_k1_IBM_s106
                                 + 0.038818427399062 w_k1_IBM_s107
                                 + 0.0245819488725803 w_k1_IBM_s110
                                 + 0.0130511595032687 w_k1_IBM_s111
                                 + 0.0578227932513019 w_k1_IBM_s112
                                 + 0.0600517516929728 w_k1_IBM_s113
                                 + 0.0300305660178352 w_k1_IBM_s114
                                 + 0.0155251537257487 w_k1_IBM_s115
                                 + 0.0212862479632804 w_k1_IBM_s116
                                 + 0.0374563364016376 w_k1_IBM_s117
                                 + 0.0135404590270417 w_k1_INTC_s000
                                 + 0.0113813232070968 w_k1_INTC_s001
                                 + 0.0233074254080571 w_k1_INTC_s002
                                 + 0.0127463147296066 w_k1_INTC_s003
                                 + 0.0278844983350181 w_k1_INTC_s004
                                 + 0.0242084713940924 w_k1_INTC_s005
                                 + 0.0190712529568388 w_k1_INTC_s006
                                 + 0.0300459158983773 w_k1_INTC_s007
                                 + 0.041524045521079 w_k1_INTC_s010
                                 + 0.0541861121675695 w_k1_INTC_s011
                                 + 0.0314101327917439 w_k1_INTC_s012
                                 + 0.0368313118581004 w_k1_INTC_s013
                                 + 0.0388887159579714 w_k1_INTC_s014
                                 + 0.0319869290002418 w_k1_INTC_s015
                                 + 0.0249750945174265 w_k1_INTC_s016
                                 + 0.0336829976753349 w_k1_INTC_s017
                                 + 0.0277989018373628 w_k1_INTC_s100
                                 + 0.0444368954469369 w_k1_INTC_s101
                                 + 0.0244838959101899 w_k1_INTC_s102
                                 + 0.0499581445992684 w_k1_INTC_s103
                                 + 0.0443795666249883 w_k1_INTC_s104
                                 + 0.0455763532447494 w_k1_INTC_s105
                                 + 0.0378147704254117 w_k1_INTC_s106
                                 + 0.0422858232005895 w_k1_INTC_s107
                                 + 0.0229842537340819 w_k1_INTC_s110
                                 + 0.0131921753165563 w_k1_INTC_s111
                                 + 0.0641198175116743 w_k1_INTC_s112
                                 + 0.0516207042501165 w_k1_INTC_s113
                                 + 0.0264617554283936 w_k1_INTC_s114
                                 + 0.0159721303052055 w_k1_INTC_s115
                                 + 0.0215487500601689 w_k1_INTC_s116
                                 + 0.0339641255235472 w_k1_INTC_s117
                                 + 0.0112652647087953 w_k1_KO_s000
                                 + 0.0117564294817131 w_k1_KO_s001
                                 + 0.0205376202730977 w_k1_KO_s002
                                 + 0.0116637587456804 w_k1_KO_s003
                                 + 0.0304729145045149 w_k1_KO_s004
                                 + 0.0246603901727368 w_k1_KO_s005
                                 + 0.0186017519696106 w_k1_KO_s006
                                 + 0.0283177687965353 w_k1_KO_s007
                                 + 0.0385533406527911 w_k1_KO_s010
                                 + 0.0538174218145604 w_k1_KO_s011
                                 + 0.0323564209226458 w_k1_KO_s012
                                 + 0.037163254204273 w_k1_KO_s013
                                 + 0.0434479053428711 w_k1_KO_s014
                                 + 0.033437194297378 w_k1_KO_s015
                                 + 0.0292089626804077 w_k1_KO_s016
                                 + 0.033579358846993 w_k1_KO_s017
                                 + 0.0285888222694224 w_k1_KO_s100
                                 + 0.0454055643667942 w_k1_KO_s101
                                 + 0.0247322732020751 w_k1_KO_s102
                                 + 0.049500203783266 w_k1_KO_s103
                                 + 0.0454771310334641 w_k1_KO_s104
                                 + 0.0513628252875276 w_k1_KO_s105
                                 + 0.0336222838865641 w_k1_KO_s106
                                 + 0.03311116798699 w_k1_KO_s107
                                 + 0.0216751482382265 w_k1_KO_s110
                                 + 0.0141775811606156 w_k1_KO_s111
                                 + 0.0568963690172948 w_k1_KO_s112
                                 + 0.0572983116115087 w_k1_KO_s113
                                 + 0.0279035252002196 w_k1_KO_s114
                                 + 0.015283537590282 w_k1_KO_s115
                                 + 0.0210535089694018 w_k1_KO_s116
                                 + 0.0353580649627898 w_k1_KO_s117
                                 + 0.0135725132051396 w_k1_MAT_s000
                                 + 0.0122323056271674 w_k1_MAT_s001
                                 + 0.0217914272210542 w_k1_MAT_s002
                                 + 0.0113685027603026 w_k1_MAT_s003
                                 + 0.0326589206499956 w_k1_MAT_s004
                                 + 0.0275558599334261 w_k1_MAT_s005
                                 + 0.0199539154824934 w_k1_MAT_s006
                                 + 0.0260217926630556 w_k1_MAT_s007
                                 + 0.0455373946338782 w_k1_MAT_s010
                                 + 0.0574780668060796 w_k1_MAT_s011
                                 + 0.0377285243843134 w_k1_MAT_s012
                                 + 0.0353186058010659 w_k1_MAT_s013
                                 + 0.0449900086956784 w_k1_MAT_s014
                                 + 0.0476299924895341 w_k1_MAT_s015
                                 + 0.030644136655968 w_k1_MAT_s016
                                 + 0.0402688310931353 w_k1_MAT_s017
                                 + 0.0263390087643429 w_k1_MAT_s100
                                 + 0.0446955410382749 w_k1_MAT_s101
                                 + 0.0268937191860701 w_k1_MAT_s102
                                 + 0.0455452618487359 w_k1_MAT_s103
                                 + 0.0445422014447668 w_k1_MAT_s104
                                 + 0.0474130311493327 w_k1_MAT_s105
                                 + 0.0317231770059871 w_k1_MAT_s106
                                 + 0.0471253112302051 w_k1_MAT_s107
                                 + 0.0222275880380382 w_k1_MAT_s110
                                 + 0.0140492703939941 w_k1_MAT_s111
                                 + 0.0715334926087793 w_k1_MAT_s112
                                 + 0.0679326615731461 w_k1_MAT_s113
                                 + 0.0250263106191866 w_k1_MAT_s114
                                 + 0.0183577087406258 w_k1_MAT_s115
                                 + 0.0213895047807346 w_k1_MAT_s116
                                 + 0.0454103357417087 w_k1_MAT_s117
                                 + 0.0143502728399027 w_k1_MSFT_s000
                                 + 0.0121192993959673 w_k1_MSFT_s001
                                 + 0.0226537483809223 w_k1_MSFT_s002
                                 + 0.011442685631296 w_k1_MSFT_s003
                                 + 0.0325866737657057 w_k1_MSFT_s004
                                 + 0.0244131155506164 w_k1_MSFT_s005
                                 + 0.0207908909739444 w_k1_MSFT_s006
                                 + 0.0265331515474823 w_k1_MSFT_s007
                                 + 0.0384827027212269 w_k1_MSFT_s010
                                 + 0.0551887651058904 w_k1_MSFT_s011
                                 + 0.0342539666178696 w_k1_MSFT_s012
                                 + 0.0341920507846613 w_k1_MSFT_s013
                                 + 0.0389000368083812 w_k1_MSFT_s014
                                 + 0.0319727053239423 w_k1_MSFT_s015
                                 + 0.0282838644332571 w_k1_MSFT_s016
                                 + 0.0341657150111351 w_k1_MSFT_s017
                                 + 0.0280289918621974 w_k1_MSFT_s100
                                 + 0.0418209402887453 w_k1_MSFT_s101
                                 + 0.024066947898108 w_k1_MSFT_s102
                                 + 0.0441150804746937 w_k1_MSFT_s103
                                 + 0.0425710476697663 w_k1_MSFT_s104
                                 + 0.0443923839489314 w_k1_MSFT_s105
                                 + 0.0335209118239808 w_k1_MSFT_s106
                                 + 0.0401186733094214 w_k1_MSFT_s107
                                 + 0.0223477166002264 w_k1_MSFT_s110
                                 + 0.014520022202733 w_k1_MSFT_s111
                                 + 0.0609207229277071 w_k1_MSFT_s112
                                 + 0.0619515510320058 w_k1_MSFT_s113
                                 + 0.0278151668964844 w_k1_MSFT_s114
                                 + 0.0157215627880588 w_k1_MSFT_s115
                                 + 0.0196177289533114 w_k1_MSFT_s116
                                 + 0.0374078352111398 w_k1_MSFT_s117
                                 + 0.012857766649068 w_k1_PG_s000
                                 + 0.0124188657915378 w_k1_PG_s001
                                 + 0.0226434247420887 w_k1_PG_s002
                                 + 0.0124182519203392 w_k1_PG_s003
                                 + 0.0309620048124531 w_k1_PG_s004
                                 + 0.0233667161185981 w_k1_PG_s005
                                 + 0.0186842947104192 w_k1_PG_s006
                                 + 0.0286996699431028 w_k1_PG_s007
                                 + 0.0396443515797469 w_k1_PG_s010
                                 + 0.0516643623761105 w_k1_PG_s011
                                 + 0.0328511947132073 w_k1_PG_s012
                                 + 0.0332434764312451 w_k1_PG_s013
                                 + 0.0406371534051216 w_k1_PG_s014
                                 + 0.0342255120547429 w_k1_PG_s015
                                 + 0.0284014540737418 w_k1_PG_s016
                                 + 0.0340860070025236 w_k1_PG_s017
                                 + 0.0275010074757032 w_k1_PG_s100
                                 + 0.0444566785517229 w_k1_PG_s101
                                 + 0.026784674340311 w_k1_PG_s102
                                 + 0.042642771697377 w_k1_PG_s103
                                 + 0.0442068385366451 w_k1_PG_s104
                                 + 0.0486301096876643 w_k1_PG_s105
                                 + 0.0362159019667527 w_k1_PG_s106
                                 + 0.0377289348040421 w_k1_PG_s107
                                 + 0.0221932898581288 w_k1_PG_s110
                                 + 0.0126877595181887 w_k1_PG_s111
                                 + 0.0528077413293643 w_k1_PG_s112
                                 + 0.0587271681159309 w_k1_PG_s113
                                 + 0.0283507903366627 w_k1_PG_s114
                                 + 0.0142676504630221 w_k1_PG_s115
                                 + 0.0204990289951879 w_k1_PG_s116
                                 + 0.034164996836673 w_k1_PG_s117 >= 1.209
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
