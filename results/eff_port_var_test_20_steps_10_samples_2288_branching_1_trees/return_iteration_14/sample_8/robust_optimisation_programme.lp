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
 w_k1_ADM_s0_return_constraint:  1.13345 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.902878 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.13628 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.98241 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.976572 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15846 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.20245 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.14445 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.37177 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.10258 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.952808 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.824613 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.830451 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.00846 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.873883 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.02542 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.917093 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.13979 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.956173 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.917255 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.39068 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.740796 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.36373 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.780826 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.600103 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.33301 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.18662 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.902544 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.953541 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.829505 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.896177 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.15875 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.948037 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.04818 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.01506 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.29878 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.41632 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.42366 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18129 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.884295 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.38314 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.950092 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.940645 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.35349 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.32425 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.03846 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.44519 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.26422 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.03341 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.693397 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.954722 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.823827 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.890747 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.16161 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.70001 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.28524 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.991993 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.20885 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.41347 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.809688 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.25375 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.854533 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.507383 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.47431 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.12023 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.756463 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.878944 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.818532 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.731937 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.959222 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.929784 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.889215 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.02399 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.26927 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.40711 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.39273 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25813 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.867756 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.39186 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.881349 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.894955 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.37958 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.43869 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.959032 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.70297 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.12587 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.24314 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.684055 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.806661 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.988083 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.986512 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.989779 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.772682 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.3471 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.14394 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.16265 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.6138 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.744245 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.37788 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.702626 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.467404 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.58485 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.02686 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.7555 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.90836 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.766103 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.760549 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.926048 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.888056 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.808746 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.22737 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.21671 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.82588 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.51803 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18826 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.896497 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.15471 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.929779 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.949082 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.21436 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.48918 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.16795 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.42299 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.12472 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.01821 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.76784 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.769264 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.9328 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.952211 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.0446 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.858069 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.1639 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.92364 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.935795 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.41065 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.771704 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.30531 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.735094 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.624633 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.33987 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.14455 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.946665 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.937153 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.790044 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.761967 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.07441 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.829288 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.948563 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.16709 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.09116 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.58483 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.12495 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14341 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.944567 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.37342 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.891706 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.874952 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.45642 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.791005 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.66399 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.28764 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.11914 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.54344 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.830573 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.43034 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.87998 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.857445 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.08529 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.609357 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.21566 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.40487 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.82613 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.01578 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.12282 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.03905 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.46252 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.680113 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.48855 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.841148 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.516459 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.897785 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.1051 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.902499 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.602105 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.37927 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.762209 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.810698 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.1947 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.902347 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.83795 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12883 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.92826 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.17605 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.871089 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.93853 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.02092 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.3085 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.980213 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.15815 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.988077 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.07312 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.911759 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.757909 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.976956 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.21527 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.1633 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.989887 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.26599 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.02314 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.938087 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.1816 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.834836 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.11784 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.804394 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.714924 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.21947 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.04637 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.899688 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.995617 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.951296 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.91977 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.09318 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.12294 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.952044 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.26709 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.20678 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.65793 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.3258 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.17092 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.9285 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.23304 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.875288 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.863959 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.35119 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.00454 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.854044 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.45795 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.12892 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.13043 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.756841 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.00998 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.955567 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.01015 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.15337 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.762704 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.21212 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.28975 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.23911 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.23069 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.9064 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.11336 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.02963 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.633024 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.42447 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.00757 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.733591 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.964075 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.951325 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.824947 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.815107 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.03487 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.740519 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.03787 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.15205 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.2527 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.52086 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.19331 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.839539 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.50896 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.01103 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.818258 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.35389 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.868039 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.726608 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.30419 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.1725 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.43734 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.682869 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.46085 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.73944 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.725956 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.929359 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.403399 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.07132 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.30634 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.91154 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.956646 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.83442 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.1004 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.7002 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.646825 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.12856 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.03522 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.680641 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.24369 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24692 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.923197 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.855938 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.58738 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.945476 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.11934 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.49543 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.22964 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.38017 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.17731 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.885231 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.07126 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.914996 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.991988 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.21722 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.3839 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.1385 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.32729 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.0278 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.11064 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.853272 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.843209 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.08167 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.07341 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.951465 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.89893 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.1594 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.969606 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.993365 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.19726 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.8323 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.23926 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.708503 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.648573 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.21804 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.02233 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.937181 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.930797 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.809238 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.832597 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.00641 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.888415 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.860236 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.18262 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.05425 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.50625 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.20363 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08882 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.938389 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.18549 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.937549 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.952769 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.14435 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.1403 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.00519 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.20398 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.99709 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05856 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.800311 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.966971 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.943435 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.974719 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.05552 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.878622 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.15845 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.0992 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11108 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.17913 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.92482 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.18066 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.869817 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.731421 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.18669 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.03168 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.856191 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.958955 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.885276 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.857943 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.962297 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.992382 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.887964 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.11622 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.12945 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.3347 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.35085 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0910002112581666 z_k1_s0000
                                 - 0.078000195307027 z_k1_s0001
                                 - 0.0940001689698805 z_k1_s0002
                                 - 0.138000046742633 z_k1_s0003
                                 - 0.0660001652597921 z_k1_s0004
                                 - 0.121000286376254 z_k1_s0005
                                 - 0.263000276734505 z_k1_s0006
                                 - 0.144999948069965 z_k1_s0007
                                 - 0.35300141472434 z_k1_s0010
                                 - 0.229000761909017 z_k1_s0011
                                 - 0.191000595268953 z_k1_s0012
                                 - 0.432001724812768 z_k1_s0013
                                 - 0.188000618460653 z_k1_s0014
                                 - 0.326001163048043 z_k1_s0015
                                 - 0.385001627747804 z_k1_s0016
                                 - 0.198000771355785 z_k1_s0017
                                 - 0.226000806388401 z_k1_s0020
                                 - 0.0770000844905258 z_k1_s0021
                                 - 0.135000189562656 z_k1_s0022
                                 - 0.232999616204383 z_k1_s0023
                                 - 0.257000291007349 z_k1_s0024
                                 - 0.247000009839447 z_k1_s0025
                                 - 0.113000125793891 z_k1_s0026
                                 - 0.0990001482592917 z_k1_s0027
                                 - 0.452001188903319 z_k1_s0030
                                 - 0.385001132105933 z_k1_s0031
                                 - 0.165000292387886 z_k1_s0032
                                 - 0.120000300282268 z_k1_s0033
                                 - 0.232000541985643 z_k1_s0034
                                 - 0.104000183124484 z_k1_s0035
                                 - 0.303999912393447 z_k1_s0036
                                 - 0.16600034791034 z_k1_s0037
                                 - 0.246999822576497 z_k1_s0040
                                 - 0.104000155348234 z_k1_s0041
                                 - 0.144000627251822 z_k1_s0042
                                 - 0.191999896334802 z_k1_s0043
                                 - 0.246000131554235 z_k1_s0044
                                 - 0.191999896334802 z_k1_s0045
                                 - 0.19600007042529 z_k1_s0046
                                 - 0.0890001370095505 z_k1_s0047
                                 - 0.176000035177363 z_k1_s0050
                                 - 0.381999352739047 z_k1_s0051
                                 - 0.365999668614034 z_k1_s0052
                                 - 0.366999962247045 z_k1_s0053
                                 - 0.362001001083574 z_k1_s0054
                                 - 0.42900061848266 z_k1_s0055
                                 - 0.258000547269403 z_k1_s0056
                                 - 0.167000150182004 z_k1_s0057
                                 - 0.216999232655118 z_k1_s0060
                                 - 0.181000303711388 z_k1_s0061
                                 - 0.229999539634103 z_k1_s0062
                                 - 0.25299982659718 z_k1_s0063
                                 - 0.165999834235751 z_k1_s0064
                                 - 0.299999399522743 z_k1_s0065
                                 - 0.0989999683423386 z_k1_s0066
                                 - 0.219000560650602 z_k1_s0067
                                 - 0.579002450814022 z_k1_s0070
                                 - 0.136000412035246 z_k1_s0071
                                 - 0.375001705760752 z_k1_s0072
                                 - 0.41500175224147 z_k1_s0073
                                 - 0.133000408549192 z_k1_s0074
                                 - 0.427000750182473 z_k1_s0075
                                 - 0.176000458515965 z_k1_s0076
                                 - 0.299002125448993 z_k1_s0077
                                 - 0.450998752337873 z_k1_s0100
                                 - 0.633998087100747 z_k1_s0101
                                 - 0.630996256536653 z_k1_s0102
                                 - 0.861998590407682 z_k1_s0103
                                 - 0.64499683252747 z_k1_s0104
                                 - 0.686998560499923 z_k1_s0105
                                 - 0.6079997488235 z_k1_s0106
                                 - 0.261999102436292 z_k1_s0107
                                 - 0.342998923346311 z_k1_s0110
                                 - 0.441999318291035 z_k1_s0111
                                 - 0.176999728114282 z_k1_s0112
                                 - 0.505999573608836 z_k1_s0113
                                 - 0.16899994069931 z_k1_s0114
                                 - 0.327998863506202 z_k1_s0115
                                 - 0.359998991165102 z_k1_s0116
                                 - 0.119999989721371 z_k1_s0117
                                 - 0.377999190611691 z_k1_s0120
                                 - 0.941998148895559 z_k1_s0121
                                 - 0.563998958283868 z_k1_s0122
                                 - 0.817997529381901 z_k1_s0123
                                 - 1.37299758556392 z_k1_s0124
                                 - 0.637996255378749 z_k1_s0125
                                 - 0.397999590300554 z_k1_s0126
                                 - 0.696999641497684 z_k1_s0127
                                 - 0.366000504993685 z_k1_s0130
                                 - 0.309998783881743 z_k1_s0131
                                 - 0.36799952674885 z_k1_s0132
                                 - 0.456000151927094 z_k1_s0133
                                 - 0.159999857525516 z_k1_s0134
                                 - 0.374998528889205 z_k1_s0135
                                 - 0.234999805903084 z_k1_s0136
                                 - 0.155999873217364 z_k1_s0137
                                 - 0.237999838347683 z_k1_s0140
                                 - 0.231000526218944 z_k1_s0141
                                 - 0.218999335429824 z_k1_s0142
                                 - 0.336000162137411 z_k1_s0143
                                 - 0.0879998844647602 z_k1_s0144
                                 - 0.171999924999113 z_k1_s0145
                                 - 0.0779999428391619 z_k1_s0146
                                 - 0.297000439567514 z_k1_s0147
                                 - 0.123000034340133 z_k1_s0150
                                 - 0.272999117916046 z_k1_s0151
                                 - 0.390999449590276 z_k1_s0152
                                 - 0.100000056017177 z_k1_s0153
                                 - 0.273999567769152 z_k1_s0154
                                 - 0.290999939273019 z_k1_s0155
                                 - 0.146000012663089 z_k1_s0156
                                 - 0.220999372551072 z_k1_s0157
                                 - 0.674001731249123 z_k1_s0160
                                 - 0.467000667953855 z_k1_s0161
                                 - 0.249000314093117 z_k1_s0162
                                 - 1.02700121660485 z_k1_s0163
                                 - 0.89100336887814 z_k1_s0164
                                 - 0.268000351525942 z_k1_s0165
                                 - 0.453000438687314 z_k1_s0166
                                 - 0.761002608547664 z_k1_s0167
                                 - 0.287000464107295 z_k1_s0170
                                 - 0.423000666781453 z_k1_s0171
                                 - 0.493999114497424 z_k1_s0172
                                 - 0.931001966597638 z_k1_s0173
                                 - 0.369000725309603 z_k1_s0174
                                 - 0.863999963751168 z_k1_s0175
                                 - 0.725999063087942 z_k1_s0176
                                 - 0.407000934179119 z_k1_s0177
                                 - 1.17499866967544 z_k1_s1000
                                 - 1.09199779101025 z_k1_s1001
                                 - 0.42499950602092 z_k1_s1002
                                 - 0.484999655653005 z_k1_s1003
                                 - 0.594999428729134 z_k1_s1004
                                 - 0.440999385523015 z_k1_s1005
                                 - 0.706996179246879 z_k1_s1006
                                 - 1.09499927216497 z_k1_s1007
                                 - 0.610000529843884 z_k1_s1010
                                 - 0.48199814726244 z_k1_s1011
                                 - 0.208999823184562 z_k1_s1012
                                 - 0.553999907839097 z_k1_s1013
                                 - 0.249999798152816 z_k1_s1014
                                 - 0.444998456937153 z_k1_s1015
                                 - 0.41699814593476 z_k1_s1016
                                 - 0.395999594181343 z_k1_s1017
                                 - 0.294999893871599 z_k1_s1020
                                 - 0.172000046167518 z_k1_s1021
                                 - 0.129999977241724 z_k1_s1022
                                 - 0.371999193885481 z_k1_s1023
                                 - 0.212999995402212 z_k1_s1024
                                 - 0.595000386198704 z_k1_s1025
                                 - 0.405999957982617 z_k1_s1026
                                 - 0.178000056014061 z_k1_s1027
                                 - 1.35399856653646 z_k1_s1030
                                 - 0.40899942424737 z_k1_s1031
                                 - 0.457999273155873 z_k1_s1032
                                 - 0.657998656455885 z_k1_s1033
                                 - 0.694000601646425 z_k1_s1034
                                 - 0.962999086786095 z_k1_s1035
                                 - 0.837996045229357 z_k1_s1036
                                 - 1.47999543642009 z_k1_s1037
                                 - 0.751997785344116 z_k1_s1040
                                 - 0.873993831460121 z_k1_s1041
                                 - 0.400998288996609 z_k1_s1042
                                 - 0.523997701335482 z_k1_s1043
                                 - 0.424998327981603 z_k1_s1044
                                 - 0.874996052287757 z_k1_s1045
                                 - 0.879995702473621 z_k1_s1046
                                 - 0.995995604552284 z_k1_s1047
                                 - 0.257999697586682 z_k1_s1050
                                 - 0.308999933286128 z_k1_s1051
                                 - 0.144999891597115 z_k1_s1052
                                 - 0.521000632786018 z_k1_s1053
                                 - 0.289999783194229 z_k1_s1054
                                 - 0.216999819514318 z_k1_s1055
                                 - 0.434999674791344 z_k1_s1056
                                 - 0.471999409812876 z_k1_s1057
                                 - 1.63000253764063 z_k1_s1060
                                 - 1.06600057096647 z_k1_s1061
                                 - 0.577000662775034 z_k1_s1062
                                 - 0.553000379424895 z_k1_s1063
                                 - 1.1610018702901 z_k1_s1064
                                 - 1.8390036899511 z_k1_s1065
                                 - 0.536000640814785 z_k1_s1066
                                 - 1.16900395530856 z_k1_s1067
                                 - 0.574000443885101 z_k1_s1070
                                 - 0.541000475396992 z_k1_s1071
                                 - 1.77300407295802 z_k1_s1072
                                 - 0.557000894118953 z_k1_s1073
                                 - 0.790998066668272 z_k1_s1074
                                 - 0.413000535624872 z_k1_s1075
                                 - 1.28900180712958 z_k1_s1076
                                 - 0.882004241778387 z_k1_s1077
                                 - 0.665001355753272 z_k1_s1100
                                 - 0.284000413614592 z_k1_s1101
                                 - 0.364000706475402 z_k1_s1102
                                 - 0.512000249985854 z_k1_s1103
                                 - 0.455000037092518 z_k1_s1104
                                 - 0.175000196482111 z_k1_s1105
                                 - 0.232000409375447 z_k1_s1106
                                 - 0.697000525375653 z_k1_s1107
                                 - 0.236000416899705 z_k1_s1110
                                 - 0.0970001887766032 z_k1_s1111
                                 - 0.0540001093313121 z_k1_s1112
                                 - 0.24500057218849 z_k1_s1113
                                 - 0.0570001268275155 z_k1_s1114
                                 - 0.0520001319339014 z_k1_s1115
                                 - 0.118000208449852 z_k1_s1116
                                 - 0.169000403085136 z_k1_s1117
                                 - 0.288001310739073 z_k1_s1120
                                 - 0.122000122851253 z_k1_s1121
                                 - 0.122000122851253 z_k1_s1122
                                 - 0.198001119758475 z_k1_s1123
                                 - 0.120000138040605 z_k1_s1124
                                 - 0.239999926280631 z_k1_s1125
                                 - 0.178999952305149 z_k1_s1126
                                 - 0.480001601564157 z_k1_s1127
                                 - 0.386999274594388 z_k1_s1130
                                 - 0.306999381662741 z_k1_s1131
                                 - 0.231999926601722 z_k1_s1132
                                 - 0.382999398497779 z_k1_s1133
                                 - 0.335999076113024 z_k1_s1134
                                 - 0.103999860811141 z_k1_s1135
                                 - 0.171999888353014 z_k1_s1136
                                 - 0.44999910136558 z_k1_s1137
                                 - 0.461000870912625 z_k1_s1140
                                 - 0.222000289701947 z_k1_s1141
                                 - 0.142000066802136 z_k1_s1142
                                 - 0.13599994046964 z_k1_s1143
                                 - 0.117000071883451 z_k1_s1144
                                 - 0.379000871635339 z_k1_s1145
                                 - 0.374000434191563 z_k1_s1146
                                 - 0.16200002287708 z_k1_s1147
                                 - 0.521001879465973 z_k1_s1150
                                 - 0.357000304018068 z_k1_s1151
                                 - 0.234000511184683 z_k1_s1152
                                 - 0.236000314848287 z_k1_s1153
                                 - 0.324001018563977 z_k1_s1154
                                 - 0.208000286052743 z_k1_s1155
                                 - 0.159000298785657 z_k1_s1156
                                 - 0.486000265343652 z_k1_s1157
                                 - 0.0930002723704715 z_k1_s1160
                                 - 0.223000433231919 z_k1_s1161
                                 - 0.0460000962433712 z_k1_s1162
                                 - 0.0500000749775097 z_k1_s1163
                                 - 0.108000618624464 z_k1_s1164
                                 - 0.223000433231919 z_k1_s1165
                                 - 0.0500000749775097 z_k1_s1166
                                 - 0.0590001123295159 z_k1_s1167
                                 - 0.0930001063325071 z_k1_s1170
                                 - 0.139000563035387 z_k1_s1171
                                 - 0.0870000467625663 z_k1_s1172
                                 - 0.0620001253551098 z_k1_s1173
                                 - 0.207000421159807 z_k1_s1174
                                 - 0.0740001627948338 z_k1_s1175
                                 - 0.0860000368342428 z_k1_s1176
                                 - 0.0690001131532165 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.24483 w_k1_ADM_s000
                                 + 0.983172 w_k1_AXP_s000
                                 + 1.34079 w_k1_DIS_s000
                                 + 1.10207 w_k1_FDX_s000
                                 + 0.825919 w_k1_IBM_s000
                                 + 0.782091 w_k1_INTC_s000
                                 + 1.12124 w_k1_KO_s000
                                 + 0.884369 w_k1_MAT_s000
                                 + 1.21168 w_k1_MSFT_s000
                                 + 1.13464 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.58292 w_k1_ADM_s000
                                 + 1.56276 w_k1_AXP_s000
                                 + 1.47971 w_k1_DIS_s000
                                 + 1.32889 w_k1_FDX_s000
                                 + 0.984213 w_k1_IBM_s000
                                 + 1.15802 w_k1_INTC_s000
                                 + 1.34101 w_k1_KO_s000 + 1.52045 w_k1_MAT_s000
                                 + 1.31601 w_k1_MSFT_s000
                                 + 1.18943 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.54854 w_k1_ADM_s000
                                 + 1.14722 w_k1_AXP_s000 + 0.9643 w_k1_DIS_s000
                                 + 1.02727 w_k1_FDX_s000
                                 + 0.968129 w_k1_IBM_s000
                                 + 0.757838 w_k1_INTC_s000
                                 + 1.0911 w_k1_KO_s000 + 1.1567 w_k1_MAT_s000
                                 + 0.870312 w_k1_MSFT_s000
                                 + 0.943073 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.15365 w_k1_ADM_s000
                                 + 1.02552 w_k1_AXP_s000
                                 + 1.22332 w_k1_DIS_s000
                                 + 1.03801 w_k1_FDX_s000
                                 + 1.36779 w_k1_IBM_s000
                                 + 0.8417 w_k1_INTC_s000 + 1.39615 w_k1_KO_s000
                                 + 1.31828 w_k1_MAT_s000
                                 + 1.06603 w_k1_MSFT_s000
                                 + 1.05668 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.47507 w_k1_ADM_s000
                                 + 1.68587 w_k1_AXP_s000
                                 + 1.97767 w_k1_DIS_s000
                                 + 1.54368 w_k1_FDX_s000
                                 + 1.42139 w_k1_IBM_s000
                                 + 1.45721 w_k1_INTC_s000
                                 + 1.71702 w_k1_KO_s000 + 1.38126 w_k1_MAT_s000
                                 + 1.54925 w_k1_MSFT_s000
                                 + 1.24312 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.26505 w_k1_ADM_s000
                                 + 0.983893 w_k1_AXP_s000
                                 + 1.03091 w_k1_DIS_s000
                                 + 1.07151 w_k1_FDX_s000
                                 + 0.689117 w_k1_IBM_s000
                                 + 0.899916 w_k1_INTC_s000
                                 + 0.935271 w_k1_KO_s000
                                 + 0.676755 w_k1_MAT_s000
                                 + 1.05136 w_k1_MSFT_s000
                                 + 1.07334 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.762591 w_k1_ADM_s000
                                 + 0.680618 w_k1_AXP_s000
                                 + 0.904232 w_k1_DIS_s000
                                 + 0.809345 w_k1_FDX_s000
                                 + 1.00954 w_k1_IBM_s000
                                 + 0.855234 w_k1_INTC_s000
                                 + 1.19542 w_k1_KO_s000
                                 + 0.913611 w_k1_MAT_s000
                                 + 0.938908 w_k1_MSFT_s000
                                 + 0.889909 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.981864 w_k1_ADM_s000
                                 + 1.03605 w_k1_AXP_s000
                                 + 0.623243 w_k1_DIS_s000
                                 + 0.81881 w_k1_FDX_s000
                                 + 0.829351 w_k1_IBM_s000
                                 + 0.757315 w_k1_INTC_s000
                                 + 0.798709 w_k1_KO_s000
                                 + 0.796964 w_k1_MAT_s000
                                 + 0.806571 w_k1_MSFT_s000
                                 + 0.873531 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.04159 w_k1_ADM_s001
                                 + 0.858418 w_k1_AXP_s001
                                 + 1.01438 w_k1_DIS_s001
                                 + 1.02127 w_k1_FDX_s001
                                 + 0.831335 w_k1_IBM_s001
                                 + 1.0772 w_k1_INTC_s001
                                 + 0.973664 w_k1_KO_s001
                                 + 0.869997 w_k1_MAT_s001
                                 + 0.955727 w_k1_MSFT_s001
                                 + 0.926288 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.755602 w_k1_ADM_s001
                                 + 0.772282 w_k1_AXP_s001
                                 + 0.596398 w_k1_DIS_s001
                                 + 0.697943 w_k1_FDX_s001
                                 + 0.899435 w_k1_IBM_s001
                                 + 0.745776 w_k1_INTC_s001
                                 + 0.741724 w_k1_KO_s001
                                 + 0.840663 w_k1_MAT_s001
                                 + 0.670812 w_k1_MSFT_s001
                                 + 0.896262 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.644078 w_k1_ADM_s001
                                 + 0.68979 w_k1_AXP_s001
                                 + 0.746992 w_k1_DIS_s001
                                 + 0.768152 w_k1_FDX_s001
                                 + 0.834611 w_k1_IBM_s001
                                 + 0.937158 w_k1_INTC_s001
                                 + 0.702313 w_k1_KO_s001
                                 + 0.771765 w_k1_MAT_s001
                                 + 0.86108 w_k1_MSFT_s001
                                 + 0.791929 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.12117 w_k1_ADM_s001
                                 + 1.16977 w_k1_AXP_s001
                                 + 1.45776 w_k1_DIS_s001
                                 + 1.11834 w_k1_FDX_s001
                                 + 1.26959 w_k1_IBM_s001
                                 + 1.24304 w_k1_INTC_s001
                                 + 1.36504 w_k1_KO_s001 + 1.40657 w_k1_MAT_s001
                                 + 1.13141 w_k1_MSFT_s001
                                 + 1.12486 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.22071 w_k1_ADM_s001
                                 + 1.13651 w_k1_AXP_s001
                                 + 1.44331 w_k1_DIS_s001
                                 + 1.33928 w_k1_FDX_s001
                                 + 0.973166 w_k1_IBM_s001
                                 + 1.2956 w_k1_INTC_s001 + 1.09975 w_k1_KO_s001
                                 + 0.766784 w_k1_MAT_s001
                                 + 1.15409 w_k1_MSFT_s001
                                 + 1.14035 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.0957 w_k1_ADM_s001
                                 + 1.33102 w_k1_AXP_s001
                                 + 1.10076 w_k1_DIS_s001
                                 + 1.06566 w_k1_FDX_s001
                                 + 1.44125 w_k1_IBM_s001
                                 + 1.14547 w_k1_INTC_s001
                                 + 1.24375 w_k1_KO_s001 + 1.14933 w_k1_MAT_s001
                                 + 1.02998 w_k1_MSFT_s001
                                 + 1.18883 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05614 w_k1_ADM_s001
                                 + 1.12548 w_k1_AXP_s001
                                 + 0.833991 w_k1_DIS_s001
                                 + 0.943158 w_k1_FDX_s001
                                 + 1.03722 w_k1_IBM_s001
                                 + 0.92048 w_k1_INTC_s001
                                 + 0.856555 w_k1_KO_s001
                                 + 1.00972 w_k1_MAT_s001
                                 + 0.833877 w_k1_MSFT_s001
                                 + 0.906659 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.0274 w_k1_ADM_s001
                                 + 0.829838 w_k1_AXP_s001
                                 + 0.857542 w_k1_DIS_s001
                                 + 1.01433 w_k1_FDX_s001
                                 + 0.630289 w_k1_IBM_s001
                                 + 0.825219 w_k1_INTC_s001
                                 + 0.743952 w_k1_KO_s001
                                 + 0.696619 w_k1_MAT_s001
                                 + 0.892924 w_k1_MSFT_s001
                                 + 0.832246 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.02491 w_k1_ADM_s002
                                 + 1.08135 w_k1_AXP_s002
                                 + 0.976139 w_k1_DIS_s002
                                 + 1.14884 w_k1_FDX_s002
                                 + 0.823954 w_k1_IBM_s002
                                 + 1.20439 w_k1_INTC_s002
                                 + 0.89333 w_k1_KO_s002
                                 + 0.893293 w_k1_MAT_s002
                                 + 1.18403 w_k1_MSFT_s002
                                 + 1.02996 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.83339 w_k1_ADM_s002
                                 + 2.06565 w_k1_AXP_s002
                                 + 2.01629 w_k1_DIS_s002
                                 + 1.59941 w_k1_FDX_s002
                                 + 1.36524 w_k1_IBM_s002
                                 + 1.63498 w_k1_INTC_s002 + 1.4654 w_k1_KO_s002
                                 + 1.43995 w_k1_MAT_s002
                                 + 1.56875 w_k1_MSFT_s002
                                 + 1.26223 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.23601 w_k1_ADM_s002
                                 + 1.23766 w_k1_AXP_s002
                                 + 1.46022 w_k1_DIS_s002 + 1.1043 w_k1_FDX_s002
                                 + 1.64886 w_k1_IBM_s002
                                 + 1.16027 w_k1_INTC_s002
                                 + 1.59929 w_k1_KO_s002 + 1.8347 w_k1_MAT_s002
                                 + 1.12931 w_k1_MSFT_s002
                                 + 1.29083 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.949362 w_k1_ADM_s002
                                 + 0.78767 w_k1_AXP_s002
                                 + 0.841547 w_k1_DIS_s002
                                 + 0.848246 w_k1_FDX_s002
                                 + 0.714952 w_k1_IBM_s002
                                 + 0.767381 w_k1_INTC_s002
                                 + 0.83276 w_k1_KO_s002
                                 + 0.762425 w_k1_MAT_s002
                                 + 0.901998 w_k1_MSFT_s002
                                 + 0.930641 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.089 w_k1_ADM_s002
                                 + 1.07291 w_k1_AXP_s002
                                 + 1.32222 w_k1_DIS_s002
                                 + 1.13645 w_k1_FDX_s002
                                 + 1.07028 w_k1_IBM_s002
                                 + 1.12381 w_k1_INTC_s002
                                 + 1.20431 w_k1_KO_s002 + 1.32981 w_k1_MAT_s002
                                 + 1.15611 w_k1_MSFT_s002
                                 + 1.13573 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.771826 w_k1_ADM_s002
                                 + 0.861445 w_k1_AXP_s002
                                 + 0.882707 w_k1_DIS_s002
                                 + 0.795936 w_k1_FDX_s002
                                 + 1.0435 w_k1_IBM_s002
                                 + 0.981984 w_k1_INTC_s002
                                 + 0.95311 w_k1_KO_s002
                                 + 0.964184 w_k1_MAT_s002
                                 + 0.934865 w_k1_MSFT_s002
                                 + 0.973939 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.36285 w_k1_ADM_s002
                                 + 1.22015 w_k1_AXP_s002
                                 + 1.52145 w_k1_DIS_s002
                                 + 1.30085 w_k1_FDX_s002
                                 + 0.906161 w_k1_IBM_s002
                                 + 1.19074 w_k1_INTC_s002
                                 + 1.20892 w_k1_KO_s002
                                 + 0.977975 w_k1_MAT_s002
                                 + 1.2553 w_k1_MSFT_s002 + 1.18699 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.657916 w_k1_ADM_s002
                                 + 0.541705 w_k1_AXP_s002
                                 + 0.597803 w_k1_DIS_s002
                                 + 0.755309 w_k1_FDX_s002
                                 + 0.606486 w_k1_IBM_s002
                                 + 0.716592 w_k1_INTC_s002
                                 + 0.713841 w_k1_KO_s002
                                 + 0.637851 w_k1_MAT_s002
                                 + 0.902844 w_k1_MSFT_s002
                                 + 0.849071 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.826689 w_k1_ADM_s003
                                 + 0.811417 w_k1_AXP_s003
                                 + 0.879109 w_k1_DIS_s003
                                 + 0.900785 w_k1_FDX_s003
                                 + 0.918945 w_k1_IBM_s003
                                 + 0.968581 w_k1_INTC_s003
                                 + 0.945688 w_k1_KO_s003
                                 + 0.854746 w_k1_MAT_s003
                                 + 1.04515 w_k1_MSFT_s003
                                 + 1.02242 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02307 w_k1_ADM_s003
                                 + 1.03453 w_k1_AXP_s003
                                 + 1.40124 w_k1_DIS_s003
                                 + 1.17674 w_k1_FDX_s003
                                 + 1.11945 w_k1_IBM_s003
                                 + 1.15147 w_k1_INTC_s003
                                 + 1.31721 w_k1_KO_s003 + 1.07628 w_k1_MAT_s003
                                 + 1.23743 w_k1_MSFT_s003
                                 + 1.23677 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.957321 w_k1_ADM_s003
                                 + 0.784568 w_k1_AXP_s003
                                 + 1.04924 w_k1_DIS_s003 + 1.0436 w_k1_FDX_s003
                                 + 0.662416 w_k1_IBM_s003
                                 + 0.947722 w_k1_INTC_s003
                                 + 0.906683 w_k1_KO_s003
                                 + 0.578749 w_k1_MAT_s003
                                 + 1.15812 w_k1_MSFT_s003
                                 + 1.01121 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.07337 w_k1_ADM_s003
                                 + 1.49028 w_k1_AXP_s003
                                 + 1.54155 w_k1_DIS_s003
                                 + 1.45503 w_k1_FDX_s003
                                 + 0.926683 w_k1_IBM_s003
                                 + 1.403 w_k1_INTC_s003 + 1.08151 w_k1_KO_s003
                                 + 1.12095 w_k1_MAT_s003
                                 + 1.45658 w_k1_MSFT_s003
                                 + 1.20657 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.18932 w_k1_ADM_s003
                                 + 1.01505 w_k1_AXP_s003
                                 + 0.908818 w_k1_DIS_s003
                                 + 1.03382 w_k1_FDX_s003
                                 + 0.813649 w_k1_IBM_s003
                                 + 1.08524 w_k1_INTC_s003
                                 + 0.874102 w_k1_KO_s003
                                 + 0.934685 w_k1_MAT_s003
                                 + 0.998162 w_k1_MSFT_s003
                                 + 1.23501 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.48018 w_k1_ADM_s003
                                 + 1.55353 w_k1_AXP_s003
                                 + 1.25905 w_k1_DIS_s003
                                 + 1.14825 w_k1_FDX_s003
                                 + 1.76631 w_k1_IBM_s003
                                 + 1.28647 w_k1_INTC_s003
                                 + 1.37484 w_k1_KO_s003 + 1.90271 w_k1_MAT_s003
                                 + 1.26378 w_k1_MSFT_s003
                                 + 1.36657 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.1295 w_k1_ADM_s003
                                 + 1.10198 w_k1_AXP_s003
                                 + 0.894223 w_k1_DIS_s003
                                 + 0.921066 w_k1_FDX_s003
                                 + 1.24906 w_k1_IBM_s003
                                 + 0.992118 w_k1_INTC_s003
                                 + 1.10982 w_k1_KO_s003 + 1.40433 w_k1_MAT_s003
                                 + 0.922512 w_k1_MSFT_s003
                                 + 1.21093 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.786683 w_k1_ADM_s003
                                 + 0.675597 w_k1_AXP_s003
                                 + 0.559143 w_k1_DIS_s003
                                 + 0.705985 w_k1_FDX_s003
                                 + 0.734254 w_k1_IBM_s003
                                 + 0.766044 w_k1_INTC_s003
                                 + 0.729309 w_k1_KO_s003
                                 + 0.897912 w_k1_MAT_s003
                                 + 0.822368 w_k1_MSFT_s003
                                 + 0.742675 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.39449 w_k1_ADM_s004
                                 + 1.03797 w_k1_AXP_s004
                                 + 1.10578 w_k1_DIS_s004
                                 + 1.06498 w_k1_FDX_s004
                                 + 1.16369 w_k1_IBM_s004
                                 + 0.980852 w_k1_INTC_s004
                                 + 1.29888 w_k1_KO_s004 + 1.07551 w_k1_MAT_s004
                                 + 1.22901 w_k1_MSFT_s004
                                 + 1.13506 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.952865 w_k1_ADM_s004
                                 + 0.795679 w_k1_AXP_s004
                                 + 0.606976 w_k1_DIS_s004
                                 + 0.738611 w_k1_FDX_s004
                                 + 0.772989 w_k1_IBM_s004
                                 + 0.929632 w_k1_INTC_s004
                                 + 0.858177 w_k1_KO_s004
                                 + 0.735433 w_k1_MAT_s004
                                 + 0.904652 w_k1_MSFT_s004
                                 + 0.881293 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.758951 w_k1_ADM_s004
                                 + 0.732393 w_k1_AXP_s004
                                 + 0.672882 w_k1_DIS_s004
                                 + 0.793779 w_k1_FDX_s004
                                 + 0.588599 w_k1_IBM_s004
                                 + 0.786545 w_k1_INTC_s004
                                 + 0.787074 w_k1_KO_s004
                                 + 0.517375 w_k1_MAT_s004
                                 + 0.754173 w_k1_MSFT_s004
                                 + 0.826267 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.38495 w_k1_ADM_s004
                                 + 1.34106 w_k1_AXP_s004
                                 + 1.36075 w_k1_DIS_s004
                                 + 1.16151 w_k1_FDX_s004
                                 + 1.62977 w_k1_IBM_s004
                                 + 1.03839 w_k1_INTC_s004
                                 + 1.67545 w_k1_KO_s004 + 1.52794 w_k1_MAT_s004
                                 + 1.10191 w_k1_MSFT_s004
                                 + 1.25855 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.38128 w_k1_ADM_s004
                                 + 0.957968 w_k1_AXP_s004
                                 + 1.36631 w_k1_DIS_s004
                                 + 1.22705 w_k1_FDX_s004 + 0.9647 w_k1_IBM_s004
                                 + 1.03283 w_k1_INTC_s004
                                 + 1.17779 w_k1_KO_s004
                                 + 0.960706 w_k1_MAT_s004
                                 + 1.21322 w_k1_MSFT_s004
                                 + 1.20498 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.872207 w_k1_ADM_s004
                                 + 0.821142 w_k1_AXP_s004
                                 + 0.945882 w_k1_DIS_s004
                                 + 1.01775 w_k1_FDX_s004
                                 + 0.721579 w_k1_IBM_s004
                                 + 0.860105 w_k1_INTC_s004
                                 + 0.94581 w_k1_KO_s004
                                 + 0.729663 w_k1_MAT_s004
                                 + 0.996885 w_k1_MSFT_s004
                                 + 0.89954 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.950683 w_k1_ADM_s004
                                 + 0.912577 w_k1_AXP_s004
                                 + 0.898312 w_k1_DIS_s004
                                 + 0.944787 w_k1_FDX_s004
                                 + 0.941666 w_k1_IBM_s004
                                 + 0.759035 w_k1_INTC_s004
                                 + 1.08439 w_k1_KO_s004
                                 + 0.729022 w_k1_MAT_s004
                                 + 0.873604 w_k1_MSFT_s004
                                 + 0.925368 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.780356 w_k1_ADM_s004
                                 + 0.764888 w_k1_AXP_s004
                                 + 0.634513 w_k1_DIS_s004
                                 + 0.805243 w_k1_FDX_s004
                                 + 0.919804 w_k1_IBM_s004
                                 + 0.987803 w_k1_INTC_s004
                                 + 1.05863 w_k1_KO_s004
                                 + 0.931394 w_k1_MAT_s004
                                 + 0.919592 w_k1_MSFT_s004
                                 + 0.825642 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.872027 w_k1_ADM_s005
                                 + 0.707012 w_k1_AXP_s005
                                 + 0.648458 w_k1_DIS_s005
                                 + 0.762978 w_k1_FDX_s005
                                 + 0.701786 w_k1_IBM_s005
                                 + 0.669094 w_k1_INTC_s005
                                 + 0.816864 w_k1_KO_s005
                                 + 0.60051 w_k1_MAT_s005
                                 + 0.751466 w_k1_MSFT_s005
                                 + 0.867809 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.796642 w_k1_ADM_s005
                                 + 1.06155 w_k1_AXP_s005
                                 + 0.79608 w_k1_DIS_s005
                                 + 0.890649 w_k1_FDX_s005
                                 + 1.2464 w_k1_IBM_s005
                                 + 0.920052 w_k1_INTC_s005
                                 + 1.14491 w_k1_KO_s005 + 1.35065 w_k1_MAT_s005
                                 + 0.948995 w_k1_MSFT_s005
                                 + 1.09962 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.879015 w_k1_ADM_s005
                                 + 0.764907 w_k1_AXP_s005
                                 + 0.610331 w_k1_DIS_s005
                                 + 0.780023 w_k1_FDX_s005
                                 + 0.92406 w_k1_IBM_s005
                                 + 0.771662 w_k1_INTC_s005
                                 + 0.76567 w_k1_KO_s005
                                 + 0.850321 w_k1_MAT_s005
                                 + 0.740985 w_k1_MSFT_s005
                                 + 0.869033 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.18861 w_k1_ADM_s005
                                 + 0.947028 w_k1_AXP_s005
                                 + 0.774981 w_k1_DIS_s005
                                 + 0.97406 w_k1_FDX_s005
                                 + 0.57419 w_k1_IBM_s005
                                 + 0.758102 w_k1_INTC_s005
                                 + 0.752391 w_k1_KO_s005
                                 + 0.780294 w_k1_MAT_s005
                                 + 0.899343 w_k1_MSFT_s005
                                 + 1.03348 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.977436 w_k1_ADM_s005
                                 + 1.2881 w_k1_AXP_s005 + 1.29759 w_k1_DIS_s005
                                 + 1.22445 w_k1_FDX_s005
                                 + 1.52351 w_k1_IBM_s005
                                 + 1.03624 w_k1_INTC_s005
                                 + 1.48901 w_k1_KO_s005 + 2.18102 w_k1_MAT_s005
                                 + 1.16432 w_k1_MSFT_s005 + 1.1732 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.0386 w_k1_ADM_s005
                                 + 0.993568 w_k1_AXP_s005
                                 + 0.957413 w_k1_DIS_s005
                                 + 1.04046 w_k1_FDX_s005
                                 + 0.807865 w_k1_IBM_s005
                                 + 0.814408 w_k1_INTC_s005
                                 + 0.968817 w_k1_KO_s005
                                 + 1.07523 w_k1_MAT_s005
                                 + 0.970687 w_k1_MSFT_s005
                                 + 1.05547 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.43795 w_k1_ADM_s005
                                 + 1.3125 w_k1_AXP_s005 + 1.33749 w_k1_DIS_s005
                                 + 1.42943 w_k1_FDX_s005
                                 + 0.884538 w_k1_IBM_s005
                                 + 0.996373 w_k1_INTC_s005
                                 + 1.1591 w_k1_KO_s005 + 0.786938 w_k1_MAT_s005
                                 + 1.08604 w_k1_MSFT_s005
                                 + 1.21286 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.20199 w_k1_ADM_s005
                                 + 1.25298 w_k1_AXP_s005
                                 + 1.15006 w_k1_DIS_s005
                                 + 1.21578 w_k1_FDX_s005
                                 + 1.09688 w_k1_IBM_s005
                                 + 1.05616 w_k1_INTC_s005
                                 + 1.16865 w_k1_KO_s005 + 1.23013 w_k1_MAT_s005
                                 + 1.1424 w_k1_MSFT_s005 + 1.10914 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.02501 w_k1_ADM_s006
                                 + 0.85554 w_k1_AXP_s006
                                 + 1.06048 w_k1_DIS_s006
                                 + 1.13164 w_k1_FDX_s006
                                 + 1.06083 w_k1_IBM_s006
                                 + 1.22246 w_k1_INTC_s006
                                 + 1.20084 w_k1_KO_s006 + 1.00278 w_k1_MAT_s006
                                 + 1.09677 w_k1_MSFT_s006
                                 + 1.00089 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.784 w_k1_ADM_s006
                                 + 0.849976 w_k1_AXP_s006
                                 + 0.664526 w_k1_DIS_s006
                                 + 0.883592 w_k1_FDX_s006
                                 + 0.757059 w_k1_IBM_s006
                                 + 0.849896 w_k1_INTC_s006
                                 + 0.767187 w_k1_KO_s006
                                 + 0.892784 w_k1_MAT_s006
                                 + 0.857956 w_k1_MSFT_s006
                                 + 0.87417 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.996231 w_k1_ADM_s006
                                 + 0.686649 w_k1_AXP_s006
                                 + 0.843707 w_k1_DIS_s006
                                 + 0.826123 w_k1_FDX_s006
                                 + 0.933638 w_k1_IBM_s006
                                 + 0.96375 w_k1_INTC_s006
                                 + 1.09869 w_k1_KO_s006
                                 + 0.601204 w_k1_MAT_s006
                                 + 0.907553 w_k1_MSFT_s006
                                 + 0.938055 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.10379 w_k1_ADM_s006
                                 + 1.18789 w_k1_AXP_s006 + 1.4317 w_k1_DIS_s006
                                 + 1.19144 w_k1_FDX_s006
                                 + 1.47625 w_k1_IBM_s006
                                 + 1.44543 w_k1_INTC_s006
                                 + 1.51968 w_k1_KO_s006 + 1.51246 w_k1_MAT_s006
                                 + 1.23382 w_k1_MSFT_s006
                                 + 1.02625 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.201 w_k1_ADM_s006
                                 + 1.05847 w_k1_AXP_s006
                                 + 1.27647 w_k1_DIS_s006
                                 + 1.32769 w_k1_FDX_s006
                                 + 0.852854 w_k1_IBM_s006
                                 + 0.957089 w_k1_INTC_s006
                                 + 1.17315 w_k1_KO_s006
                                 + 0.990344 w_k1_MAT_s006
                                 + 1.208 w_k1_MSFT_s006 + 1.02969 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.799834 w_k1_ADM_s006
                                 + 0.640439 w_k1_AXP_s006
                                 + 0.810414 w_k1_DIS_s006
                                 + 0.908076 w_k1_FDX_s006
                                 + 0.684151 w_k1_IBM_s006
                                 + 1.06349 w_k1_INTC_s006
                                 + 0.834139 w_k1_KO_s006
                                 + 0.707185 w_k1_MAT_s006
                                 + 0.962772 w_k1_MSFT_s006
                                 + 0.934505 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.779109 w_k1_ADM_s006
                                 + 0.811671 w_k1_AXP_s006
                                 + 0.989497 w_k1_DIS_s006
                                 + 0.821255 w_k1_FDX_s006
                                 + 1.66547 w_k1_IBM_s006
                                 + 1.05065 w_k1_INTC_s006
                                 + 1.30689 w_k1_KO_s006 + 1.55659 w_k1_MAT_s006
                                 + 0.842725 w_k1_MSFT_s006
                                 + 0.958038 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.738379 w_k1_ADM_s006
                                 + 0.945799 w_k1_AXP_s006
                                 + 0.822959 w_k1_DIS_s006
                                 + 0.955543 w_k1_FDX_s006
                                 + 1.24493 w_k1_IBM_s006
                                 + 1.13937 w_k1_INTC_s006 + 1.0877 w_k1_KO_s006
                                 + 1.20846 w_k1_MAT_s006
                                 + 0.903201 w_k1_MSFT_s006
                                 + 0.872915 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.871799 w_k1_ADM_s007
                                 + 0.889291 w_k1_AXP_s007
                                 + 0.91025 w_k1_DIS_s007
                                 + 0.918381 w_k1_FDX_s007
                                 + 1.16838 w_k1_IBM_s007
                                 + 0.87245 w_k1_INTC_s007
                                 + 1.09259 w_k1_KO_s007 + 1.07127 w_k1_MAT_s007
                                 + 0.826485 w_k1_MSFT_s007
                                 + 0.994644 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.02765 w_k1_ADM_s007
                                 + 1.00056 w_k1_AXP_s007
                                 + 1.08892 w_k1_DIS_s007
                                 + 1.20384 w_k1_FDX_s007
                                 + 0.731414 w_k1_IBM_s007
                                 + 0.888995 w_k1_INTC_s007
                                 + 0.886072 w_k1_KO_s007
                                 + 0.79465 w_k1_MAT_s007
                                 + 1.04719 w_k1_MSFT_s007
                                 + 1.07255 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.967489 w_k1_ADM_s007
                                 + 1.21294 w_k1_AXP_s007 + 1.3485 w_k1_DIS_s007
                                 + 1.30246 w_k1_FDX_s007
                                 + 0.916698 w_k1_IBM_s007
                                 + 1.08059 w_k1_INTC_s007
                                 + 1.06957 w_k1_KO_s007 + 1.07864 w_k1_MAT_s007
                                 + 1.26679 w_k1_MSFT_s007
                                 + 1.00666 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.800288 w_k1_ADM_s007
                                 + 0.843771 w_k1_AXP_s007
                                 + 0.763054 w_k1_DIS_s007
                                 + 0.972442 w_k1_FDX_s007
                                 + 0.586005 w_k1_IBM_s007
                                 + 0.907583 w_k1_INTC_s007
                                 + 0.692709 w_k1_KO_s007
                                 + 0.941205 w_k1_MAT_s007
                                 + 0.929868 w_k1_MSFT_s007
                                 + 0.948618 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.636829 w_k1_ADM_s007
                                 + 0.89921 w_k1_AXP_s007
                                 + 1.07186 w_k1_DIS_s007
                                 + 0.996931 w_k1_FDX_s007
                                 + 0.940028 w_k1_IBM_s007
                                 + 0.98909 w_k1_INTC_s007
                                 + 1.03268 w_k1_KO_s007
                                 + 0.855661 w_k1_MAT_s007
                                 + 0.949525 w_k1_MSFT_s007
                                 + 0.985658 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.768172 w_k1_ADM_s007
                                 + 0.668644 w_k1_AXP_s007
                                 + 0.760458 w_k1_DIS_s007
                                 + 0.890807 w_k1_FDX_s007
                                 + 0.809075 w_k1_IBM_s007
                                 + 1.00233 w_k1_INTC_s007
                                 + 0.899892 w_k1_KO_s007
                                 + 0.736923 w_k1_MAT_s007
                                 + 0.927767 w_k1_MSFT_s007
                                 + 0.930027 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.986386 w_k1_ADM_s007
                                 + 1.51069 w_k1_AXP_s007
                                 + 1.47444 w_k1_DIS_s007 + 1.4489 w_k1_FDX_s007
                                 + 1.11019 w_k1_IBM_s007
                                 + 1.22745 w_k1_INTC_s007
                                 + 1.15716 w_k1_KO_s007 + 1.48014 w_k1_MAT_s007
                                 + 1.24698 w_k1_MSFT_s007 + 1.056 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.12801 w_k1_ADM_s007
                                 + 1.26347 w_k1_AXP_s007
                                 + 1.22872 w_k1_DIS_s007 + 1.171 w_k1_FDX_s007
                                 + 1.4476 w_k1_IBM_s007
                                 + 1.09858 w_k1_INTC_s007
                                 + 1.47001 w_k1_KO_s007 + 1.53703 w_k1_MAT_s007
                                 + 1.12643 w_k1_MSFT_s007
                                 + 1.23965 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.04058 w_k1_ADM_s010
                                 + 0.963288 w_k1_AXP_s010
                                 + 1.02776 w_k1_DIS_s010
                                 + 0.945951 w_k1_FDX_s010
                                 + 1.48675 w_k1_IBM_s010
                                 + 0.822871 w_k1_INTC_s010
                                 + 1.18514 w_k1_KO_s010 + 1.55439 w_k1_MAT_s010
                                 + 0.929487 w_k1_MSFT_s010
                                 + 1.10864 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.886069 w_k1_ADM_s010
                                 + 0.709057 w_k1_AXP_s010
                                 + 0.843966 w_k1_DIS_s010
                                 + 0.88592 w_k1_FDX_s010
                                 + 0.702935 w_k1_IBM_s010
                                 + 0.812256 w_k1_INTC_s010
                                 + 0.719013 w_k1_KO_s010
                                 + 0.551151 w_k1_MAT_s010
                                 + 0.914641 w_k1_MSFT_s010
                                 + 0.874064 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.49592 w_k1_ADM_s010
                                 + 1.09404 w_k1_AXP_s010
                                 + 1.28133 w_k1_DIS_s010
                                 + 1.30824 w_k1_FDX_s010
                                 + 0.71207 w_k1_IBM_s010
                                 + 1.23979 w_k1_INTC_s010
                                 + 0.884218 w_k1_KO_s010
                                 + 0.612656 w_k1_MAT_s010
                                 + 1.12317 w_k1_MSFT_s010
                                 + 1.04988 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.05727 w_k1_ADM_s010
                                 + 0.900023 w_k1_AXP_s010
                                 + 0.862192 w_k1_DIS_s010
                                 + 0.96158 w_k1_FDX_s010
                                 + 0.977911 w_k1_IBM_s010
                                 + 0.841955 w_k1_INTC_s010
                                 + 0.900106 w_k1_KO_s010
                                 + 0.853218 w_k1_MAT_s010
                                 + 0.854991 w_k1_MSFT_s010
                                 + 0.962019 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.16709 w_k1_ADM_s010
                                 + 1.40087 w_k1_AXP_s010
                                 + 1.39535 w_k1_DIS_s010
                                 + 1.29941 w_k1_FDX_s010
                                 + 1.19811 w_k1_IBM_s010
                                 + 1.03335 w_k1_INTC_s010
                                 + 1.09305 w_k1_KO_s010 + 1.07507 w_k1_MAT_s010
                                 + 1.09683 w_k1_MSFT_s010
                                 + 1.14529 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.04081 w_k1_ADM_s010
                                 + 0.954654 w_k1_AXP_s010
                                 + 1.17354 w_k1_DIS_s010
                                 + 1.06826 w_k1_FDX_s010
                                 + 1.03215 w_k1_IBM_s010
                                 + 0.815492 w_k1_INTC_s010
                                 + 1.13333 w_k1_KO_s010 + 1.10968 w_k1_MAT_s010
                                 + 1.00777 w_k1_MSFT_s010
                                 + 1.06638 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.720178 w_k1_ADM_s010
                                 + 0.635268 w_k1_AXP_s010
                                 + 0.675896 w_k1_DIS_s010
                                 + 0.732682 w_k1_FDX_s010
                                 + 1.01736 w_k1_IBM_s010
                                 + 0.807254 w_k1_INTC_s010
                                 + 0.828123 w_k1_KO_s010
                                 + 0.786214 w_k1_MAT_s010
                                 + 0.737958 w_k1_MSFT_s010
                                 + 0.911827 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.42776 w_k1_ADM_s010
                                 + 1.26089 w_k1_AXP_s010
                                 + 1.94374 w_k1_DIS_s010
                                 + 1.27832 w_k1_FDX_s010
                                 + 1.54653 w_k1_IBM_s010
                                 + 0.92802 w_k1_INTC_s010 + 1.5974 w_k1_KO_s010
                                 + 1.15051 w_k1_MAT_s010
                                 + 1.10469 w_k1_MSFT_s010 + 1.3242 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.12888 w_k1_ADM_s011
                                 + 1.24756 w_k1_AXP_s011
                                 + 1.58347 w_k1_DIS_s011
                                 + 1.23535 w_k1_FDX_s011
                                 + 1.36229 w_k1_IBM_s011
                                 + 1.14144 w_k1_INTC_s011
                                 + 1.22333 w_k1_KO_s011 + 1.41275 w_k1_MAT_s011
                                 + 1.40549 w_k1_MSFT_s011
                                 + 1.14282 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.07989 w_k1_ADM_s011
                                 + 1.06471 w_k1_AXP_s011
                                 + 1.02816 w_k1_DIS_s011 + 1.1292 w_k1_FDX_s011
                                 + 0.715794 w_k1_IBM_s011
                                 + 0.989517 w_k1_INTC_s011
                                 + 0.832189 w_k1_KO_s011
                                 + 0.983955 w_k1_MAT_s011
                                 + 1.13659 w_k1_MSFT_s011
                                 + 1.09004 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.22989 w_k1_ADM_s011
                                 + 0.923191 w_k1_AXP_s011
                                 + 0.878535 w_k1_DIS_s011
                                 + 0.901863 w_k1_FDX_s011
                                 + 0.915779 w_k1_IBM_s011
                                 + 0.777717 w_k1_INTC_s011
                                 + 0.851096 w_k1_KO_s011
                                 + 0.982645 w_k1_MAT_s011
                                 + 0.939767 w_k1_MSFT_s011
                                 + 0.964353 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.05564 w_k1_ADM_s011
                                 + 0.986174 w_k1_AXP_s011
                                 + 1.11932 w_k1_DIS_s011
                                 + 1.06131 w_k1_FDX_s011
                                 + 1.02208 w_k1_IBM_s011
                                 + 0.949404 w_k1_INTC_s011
                                 + 0.92962 w_k1_KO_s011
                                 + 0.994601 w_k1_MAT_s011
                                 + 1.11988 w_k1_MSFT_s011
                                 + 1.03963 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.04502 w_k1_ADM_s011
                                 + 0.870968 w_k1_AXP_s011
                                 + 1.32011 w_k1_DIS_s011
                                 + 1.03467 w_k1_FDX_s011
                                 + 0.897843 w_k1_IBM_s011
                                 + 0.998101 w_k1_INTC_s011
                                 + 1.15001 w_k1_KO_s011 + 1.04931 w_k1_MAT_s011
                                 + 1.24675 w_k1_MSFT_s011
                                 + 1.17676 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.07998 w_k1_ADM_s011
                                 + 1.16449 w_k1_AXP_s011
                                 + 0.976266 w_k1_DIS_s011
                                 + 0.955447 w_k1_FDX_s011
                                 + 1.17874 w_k1_IBM_s011
                                 + 0.967433 w_k1_INTC_s011
                                 + 0.982412 w_k1_KO_s011
                                 + 1.33783 w_k1_MAT_s011
                                 + 0.984549 w_k1_MSFT_s011
                                 + 1.04827 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.834066 w_k1_ADM_s011
                                 + 0.82997 w_k1_AXP_s011
                                 + 0.785445 w_k1_DIS_s011
                                 + 0.857856 w_k1_FDX_s011
                                 + 0.761591 w_k1_IBM_s011
                                 + 0.886151 w_k1_INTC_s011
                                 + 0.670288 w_k1_KO_s011
                                 + 0.950886 w_k1_MAT_s011
                                 + 0.9409 w_k1_MSFT_s011
                                 + 0.847808 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.796364 w_k1_ADM_s011
                                 + 0.592661 w_k1_AXP_s011
                                 + 0.613455 w_k1_DIS_s011
                                 + 0.675436 w_k1_FDX_s011
                                 + 0.775601 w_k1_IBM_s011
                                 + 0.830477 w_k1_INTC_s011
                                 + 0.703929 w_k1_KO_s011
                                 + 0.846798 w_k1_MAT_s011
                                 + 0.894255 w_k1_MSFT_s011
                                 + 0.774674 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.1497 w_k1_ADM_s012
                                 + 1.35385 w_k1_AXP_s012
                                 + 1.08199 w_k1_DIS_s012
                                 + 1.24108 w_k1_FDX_s012
                                 + 1.28205 w_k1_IBM_s012
                                 + 0.983679 w_k1_INTC_s012
                                 + 1.15984 w_k1_KO_s012 + 1.85474 w_k1_MAT_s012
                                 + 1.03618 w_k1_MSFT_s012
                                 + 1.05456 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.809589 w_k1_ADM_s012
                                 + 0.928507 w_k1_AXP_s012
                                 + 0.909467 w_k1_DIS_s012
                                 + 0.857661 w_k1_FDX_s012
                                 + 1.40483 w_k1_IBM_s012
                                 + 0.889746 w_k1_INTC_s012
                                 + 1.14463 w_k1_KO_s012 + 1.85263 w_k1_MAT_s012
                                 + 0.856188 w_k1_MSFT_s012
                                 + 1.02485 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.646494 w_k1_ADM_s012
                                 + 0.587486 w_k1_AXP_s012
                                 + 0.504198 w_k1_DIS_s012
                                 + 0.644132 w_k1_FDX_s012
                                 + 0.770388 w_k1_IBM_s012
                                 + 0.743405 w_k1_INTC_s012
                                 + 0.716976 w_k1_KO_s012
                                 + 0.820308 w_k1_MAT_s012
                                 + 0.694786 w_k1_MSFT_s012
                                 + 0.761154 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.959871 w_k1_ADM_s012
                                 + 1.2144 w_k1_AXP_s012
                                 + 0.966372 w_k1_DIS_s012
                                 + 1.08952 w_k1_FDX_s012
                                 + 1.03883 w_k1_IBM_s012
                                 + 1.1956 w_k1_INTC_s012 + 1.05767 w_k1_KO_s012
                                 + 1.14682 w_k1_MAT_s012
                                 + 0.947779 w_k1_MSFT_s012
                                 + 1.03545 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.756495 w_k1_ADM_s012
                                 + 0.785578 w_k1_AXP_s012
                                 + 0.791198 w_k1_DIS_s012
                                 + 0.850858 w_k1_FDX_s012
                                 + 0.894214 w_k1_IBM_s012
                                 + 0.799746 w_k1_INTC_s012
                                 + 0.877802 w_k1_KO_s012
                                 + 1.08036 w_k1_MAT_s012
                                 + 0.819007 w_k1_MSFT_s012
                                 + 0.868457 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.37087 w_k1_ADM_s012
                                 + 1.53593 w_k1_AXP_s012
                                 + 1.68632 w_k1_DIS_s012
                                 + 1.48665 w_k1_FDX_s012
                                 + 1.19859 w_k1_IBM_s012
                                 + 1.26195 w_k1_INTC_s012
                                 + 1.15333 w_k1_KO_s012 + 1.68187 w_k1_MAT_s012
                                 + 1.37297 w_k1_MSFT_s012
                                 + 1.25295 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.08436 w_k1_ADM_s012
                                 + 1.14828 w_k1_AXP_s012 + 1.3117 w_k1_DIS_s012
                                 + 1.2077 w_k1_FDX_s012 + 1.39023 w_k1_IBM_s012
                                 + 1.09551 w_k1_INTC_s012
                                 + 1.22757 w_k1_KO_s012 + 1.82274 w_k1_MAT_s012
                                 + 1.12369 w_k1_MSFT_s012
                                 + 1.16714 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.17272 w_k1_ADM_s012
                                 + 1.0605 w_k1_AXP_s012 + 1.06987 w_k1_DIS_s012
                                 + 1.16911 w_k1_FDX_s012
                                 + 0.711398 w_k1_IBM_s012
                                 + 1.00627 w_k1_INTC_s012
                                 + 0.93221 w_k1_KO_s012
                                 + 0.879229 w_k1_MAT_s012
                                 + 1.1037 w_k1_MSFT_s012 + 1.14576 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.792023 w_k1_ADM_s013
                                 + 0.62186 w_k1_AXP_s013
                                 + 0.87587 w_k1_DIS_s013
                                 + 0.814446 w_k1_FDX_s013
                                 + 0.959961 w_k1_IBM_s013
                                 + 1.06898 w_k1_INTC_s013
                                 + 0.964316 w_k1_KO_s013
                                 + 0.843609 w_k1_MAT_s013
                                 + 1.04179 w_k1_MSFT_s013
                                 + 0.906865 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.802463 w_k1_ADM_s013
                                 + 0.843125 w_k1_AXP_s013
                                 + 0.973172 w_k1_DIS_s013
                                 + 0.988731 w_k1_FDX_s013
                                 + 0.737648 w_k1_IBM_s013
                                 + 1.03853 w_k1_INTC_s013
                                 + 0.835654 w_k1_KO_s013
                                 + 0.665211 w_k1_MAT_s013
                                 + 0.994869 w_k1_MSFT_s013
                                 + 0.844508 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.815076 w_k1_ADM_s013
                                 + 0.802653 w_k1_AXP_s013
                                 + 0.642967 w_k1_DIS_s013
                                 + 0.776967 w_k1_FDX_s013
                                 + 0.851493 w_k1_IBM_s013
                                 + 0.824492 w_k1_INTC_s013
                                 + 0.707751 w_k1_KO_s013
                                 + 0.779484 w_k1_MAT_s013
                                 + 0.737098 w_k1_MSFT_s013
                                 + 0.75833 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.11768 w_k1_ADM_s013
                                 + 1.15716 w_k1_AXP_s013
                                 + 1.05919 w_k1_DIS_s013
                                 + 0.991629 w_k1_FDX_s013
                                 + 1.25552 w_k1_IBM_s013
                                 + 0.92165 w_k1_INTC_s013
                                 + 1.15531 w_k1_KO_s013 + 1.2953 w_k1_MAT_s013
                                 + 0.944486 w_k1_MSFT_s013
                                 + 0.986571 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.35946 w_k1_ADM_s013
                                 + 1.18666 w_k1_AXP_s013 + 1.3624 w_k1_DIS_s013
                                 + 1.27234 w_k1_FDX_s013
                                 + 1.03047 w_k1_IBM_s013
                                 + 1.33614 w_k1_INTC_s013
                                 + 1.32544 w_k1_KO_s013 + 1.03132 w_k1_MAT_s013
                                 + 1.27199 w_k1_MSFT_s013
                                 + 1.09693 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.04095 w_k1_ADM_s013
                                 + 1.04881 w_k1_AXP_s013
                                 + 1.06617 w_k1_DIS_s013
                                 + 1.09203 w_k1_FDX_s013
                                 + 1.00918 w_k1_IBM_s013
                                 + 0.86506 w_k1_INTC_s013
                                 + 1.14094 w_k1_KO_s013 + 1.36606 w_k1_MAT_s013
                                 + 0.946057 w_k1_MSFT_s013
                                 + 1.05607 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.05737 w_k1_ADM_s013
                                 + 1.32385 w_k1_AXP_s013
                                 + 1.68315 w_k1_DIS_s013
                                 + 1.36606 w_k1_FDX_s013
                                 + 1.51841 w_k1_IBM_s013
                                 + 1.36416 w_k1_INTC_s013 + 1.5492 w_k1_KO_s013
                                 + 1.53746 w_k1_MAT_s013
                                 + 1.49221 w_k1_MSFT_s013
                                 + 1.28056 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.06337 w_k1_ADM_s013
                                 + 0.884109 w_k1_AXP_s013
                                 + 1.16105 w_k1_DIS_s013
                                 + 0.946311 w_k1_FDX_s013
                                 + 1.69526 w_k1_IBM_s013
                                 + 0.933156 w_k1_INTC_s013
                                 + 1.43282 w_k1_KO_s013 + 1.64321 w_k1_MAT_s013
                                 + 1.1429 w_k1_MSFT_s013 + 1.07897 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.08816 w_k1_ADM_s014
                                 + 1.00354 w_k1_AXP_s014
                                 + 0.965935 w_k1_DIS_s014
                                 + 1.04922 w_k1_FDX_s014
                                 + 1.00391 w_k1_IBM_s014
                                 + 1.11172 w_k1_INTC_s014
                                 + 0.925379 w_k1_KO_s014
                                 + 0.903565 w_k1_MAT_s014
                                 + 1.1029 w_k1_MSFT_s014 + 0.99001 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.49788 w_k1_ADM_s014
                                 + 1.65826 w_k1_AXP_s014
                                 + 1.30685 w_k1_DIS_s014
                                 + 1.39637 w_k1_FDX_s014
                                 + 1.09885 w_k1_IBM_s014
                                 + 1.27634 w_k1_INTC_s014
                                 + 1.09145 w_k1_KO_s014 + 1.08546 w_k1_MAT_s014
                                 + 1.20175 w_k1_MSFT_s014
                                 + 1.09771 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.23437 w_k1_ADM_s014
                                 + 1.13548 w_k1_AXP_s014
                                 + 1.45912 w_k1_DIS_s014
                                 + 1.23751 w_k1_FDX_s014
                                 + 1.07564 w_k1_IBM_s014
                                 + 1.28873 w_k1_INTC_s014
                                 + 1.12118 w_k1_KO_s014
                                 + 0.844099 w_k1_MAT_s014
                                 + 1.27937 w_k1_MSFT_s014
                                 + 1.05789 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.990362 w_k1_ADM_s014
                                 + 0.84617 w_k1_AXP_s014
                                 + 0.834569 w_k1_DIS_s014
                                 + 0.946918 w_k1_FDX_s014
                                 + 0.786727 w_k1_IBM_s014
                                 + 0.935847 w_k1_INTC_s014
                                 + 0.867965 w_k1_KO_s014
                                 + 0.735796 w_k1_MAT_s014
                                 + 0.967389 w_k1_MSFT_s014
                                 + 0.922807 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.983546 w_k1_ADM_s014
                                 + 0.844592 w_k1_AXP_s014
                                 + 0.742805 w_k1_DIS_s014
                                 + 0.935092 w_k1_FDX_s014
                                 + 1.17208 w_k1_IBM_s014
                                 + 0.98475 w_k1_INTC_s014
                                 + 0.892817 w_k1_KO_s014
                                 + 1.10166 w_k1_MAT_s014
                                 + 0.902638 w_k1_MSFT_s014
                                 + 0.950082 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.818038 w_k1_ADM_s014
                                 + 0.597603 w_k1_AXP_s014
                                 + 0.626608 w_k1_DIS_s014
                                 + 0.78051 w_k1_FDX_s014
                                 + 0.729503 w_k1_IBM_s014
                                 + 0.681311 w_k1_INTC_s014
                                 + 0.664472 w_k1_KO_s014
                                 + 0.588819 w_k1_MAT_s014
                                 + 0.754248 w_k1_MSFT_s014
                                 + 0.797662 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.83332 w_k1_ADM_s014
                                 + 0.775108 w_k1_AXP_s014
                                 + 0.73213 w_k1_DIS_s014
                                 + 0.775344 w_k1_FDX_s014
                                 + 1.14764 w_k1_IBM_s014
                                 + 0.920808 w_k1_INTC_s014
                                 + 0.859182 w_k1_KO_s014
                                 + 1.01206 w_k1_MAT_s014
                                 + 0.85994 w_k1_MSFT_s014
                                 + 0.858949 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.12746 w_k1_ADM_s014
                                 + 1.12662 w_k1_AXP_s014
                                 + 1.12244 w_k1_DIS_s014
                                 + 1.04909 w_k1_FDX_s014
                                 + 1.43163 w_k1_IBM_s014
                                 + 1.27348 w_k1_INTC_s014
                                 + 1.16278 w_k1_KO_s014
                                 + 0.712769 w_k1_MAT_s014
                                 + 1.09852 w_k1_MSFT_s014
                                 + 0.987538 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.67994 w_k1_ADM_s015
                                 + 1.62421 w_k1_AXP_s015
                                 + 1.59671 w_k1_DIS_s015 + 1.5089 w_k1_FDX_s015
                                 + 1.56271 w_k1_IBM_s015
                                 + 1.60091 w_k1_INTC_s015
                                 + 1.68716 w_k1_KO_s015 + 1.8003 w_k1_MAT_s015
                                 + 1.4944 w_k1_MSFT_s015 + 1.45816 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.789938 w_k1_ADM_s015
                                 + 0.770127 w_k1_AXP_s015
                                 + 0.677427 w_k1_DIS_s015
                                 + 0.846216 w_k1_FDX_s015
                                 + 0.671541 w_k1_IBM_s015
                                 + 0.860005 w_k1_INTC_s015
                                 + 0.789856 w_k1_KO_s015
                                 + 0.640992 w_k1_MAT_s015
                                 + 0.753298 w_k1_MSFT_s015
                                 + 0.869426 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.988067 w_k1_ADM_s015
                                 + 0.935935 w_k1_AXP_s015
                                 + 1.03579 w_k1_DIS_s015
                                 + 0.990798 w_k1_FDX_s015
                                 + 0.930361 w_k1_IBM_s015
                                 + 1.11215 w_k1_INTC_s015 + 1.0412 w_k1_KO_s015
                                 + 0.77076 w_k1_MAT_s015
                                 + 0.925129 w_k1_MSFT_s015
                                 + 0.958558 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.2368 w_k1_ADM_s015
                                 + 1.22044 w_k1_AXP_s015
                                 + 1.27344 w_k1_DIS_s015
                                 + 1.37654 w_k1_FDX_s015
                                 + 0.878395 w_k1_IBM_s015
                                 + 1.28441 w_k1_INTC_s015
                                 + 1.08494 w_k1_KO_s015
                                 + 0.760436 w_k1_MAT_s015
                                 + 1.13759 w_k1_MSFT_s015 + 1.0006 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.11708 w_k1_ADM_s015
                                 + 1.11879 w_k1_AXP_s015
                                 + 0.981629 w_k1_DIS_s015
                                 + 1.01301 w_k1_FDX_s015
                                 + 1.16327 w_k1_IBM_s015
                                 + 1.32043 w_k1_INTC_s015
                                 + 1.26003 w_k1_KO_s015
                                 + 0.997611 w_k1_MAT_s015
                                 + 0.986904 w_k1_MSFT_s015
                                 + 1.1117 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.95519 w_k1_ADM_s015
                                 + 0.734667 w_k1_AXP_s015
                                 + 0.666165 w_k1_DIS_s015
                                 + 0.667422 w_k1_FDX_s015
                                 + 1.18227 w_k1_IBM_s015
                                 + 0.838709 w_k1_INTC_s015
                                 + 1.06705 w_k1_KO_s015
                                 + 0.789202 w_k1_MAT_s015
                                 + 0.742595 w_k1_MSFT_s015
                                 + 0.983825 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.11946 w_k1_ADM_s015
                                 + 1.49694 w_k1_AXP_s015
                                 + 0.963636 w_k1_DIS_s015
                                 + 1.12912 w_k1_FDX_s015
                                 + 0.810174 w_k1_IBM_s015
                                 + 1.26465 w_k1_INTC_s015
                                 + 0.992123 w_k1_KO_s015
                                 + 1.11659 w_k1_MAT_s015
                                 + 1.12609 w_k1_MSFT_s015
                                 + 1.04575 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.43981 w_k1_ADM_s015
                                 + 1.36721 w_k1_AXP_s015
                                 + 1.53211 w_k1_DIS_s015
                                 + 1.37746 w_k1_FDX_s015 + 1.2891 w_k1_IBM_s015
                                 + 1.18061 w_k1_INTC_s015
                                 + 1.27318 w_k1_KO_s015 + 1.30348 w_k1_MAT_s015
                                 + 1.22623 w_k1_MSFT_s015
                                 + 1.22267 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.839576 w_k1_ADM_s016
                                 + 0.984524 w_k1_AXP_s016
                                 + 0.782581 w_k1_DIS_s016
                                 + 0.877966 w_k1_FDX_s016
                                 + 0.918521 w_k1_IBM_s016
                                 + 1.07593 w_k1_INTC_s016
                                 + 1.01561 w_k1_KO_s016
                                 + 0.668459 w_k1_MAT_s016
                                 + 0.957935 w_k1_MSFT_s016
                                 + 1.00069 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.03653 w_k1_ADM_s016
                                 + 1.18384 w_k1_AXP_s016
                                 + 1.24335 w_k1_DIS_s016
                                 + 1.01635 w_k1_FDX_s016
                                 + 1.66661 w_k1_IBM_s016
                                 + 1.10188 w_k1_INTC_s016
                                 + 1.32964 w_k1_KO_s016 + 1.57196 w_k1_MAT_s016
                                 + 1.12292 w_k1_MSFT_s016
                                 + 1.09021 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.625711 w_k1_ADM_s016
                                 + 0.570531 w_k1_AXP_s016
                                 + 0.610461 w_k1_DIS_s016
                                 + 0.711815 w_k1_FDX_s016
                                 + 0.636661 w_k1_IBM_s016
                                 + 0.755059 w_k1_INTC_s016
                                 + 0.737017 w_k1_KO_s016
                                 + 0.747094 w_k1_MAT_s016
                                 + 0.853816 w_k1_MSFT_s016
                                 + 0.812838 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.00311 w_k1_ADM_s016
                                 + 1.05226 w_k1_AXP_s016 + 1.1273 w_k1_DIS_s016
                                 + 1.03256 w_k1_FDX_s016
                                 + 1.15519 w_k1_IBM_s016 + 1.111 w_k1_INTC_s016
                                 + 1.19673 w_k1_KO_s016 + 1.35172 w_k1_MAT_s016
                                 + 1.054 w_k1_MSFT_s016 + 1.13568 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.956133 w_k1_ADM_s016
                                 + 0.929194 w_k1_AXP_s016
                                 + 0.953784 w_k1_DIS_s016
                                 + 1.01082 w_k1_FDX_s016
                                 + 0.698006 w_k1_IBM_s016
                                 + 1.21222 w_k1_INTC_s016
                                 + 0.874646 w_k1_KO_s016
                                 + 0.65982 w_k1_MAT_s016
                                 + 1.12509 w_k1_MSFT_s016
                                 + 0.977207 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.786406 w_k1_ADM_s016
                                 + 0.68061 w_k1_AXP_s016
                                 + 0.573265 w_k1_DIS_s016
                                 + 0.807094 w_k1_FDX_s016
                                 + 0.691454 w_k1_IBM_s016
                                 + 0.857847 w_k1_INTC_s016
                                 + 0.658972 w_k1_KO_s016
                                 + 0.751029 w_k1_MAT_s016
                                 + 0.812539 w_k1_MSFT_s016
                                 + 0.843166 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.745879 w_k1_ADM_s016
                                 + 0.562782 w_k1_AXP_s016
                                 + 0.788663 w_k1_DIS_s016
                                 + 0.755739 w_k1_FDX_s016
                                 + 0.99109 w_k1_IBM_s016
                                 + 0.999996 w_k1_INTC_s016
                                 + 0.852781 w_k1_KO_s016
                                 + 0.632903 w_k1_MAT_s016
                                 + 0.971481 w_k1_MSFT_s016
                                 + 0.903552 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.24384 w_k1_ADM_s016
                                 + 1.43628 w_k1_AXP_s016
                                 + 1.58974 w_k1_DIS_s016
                                 + 1.29038 w_k1_FDX_s016
                                 + 1.12493 w_k1_IBM_s016
                                 + 1.41573 w_k1_INTC_s016
                                 + 1.20236 w_k1_KO_s016 + 1.02992 w_k1_MAT_s016
                                 + 1.48201 w_k1_MSFT_s016
                                 + 1.21911 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.40497 w_k1_ADM_s017
                                 + 1.28567 w_k1_AXP_s017
                                 + 0.998864 w_k1_DIS_s017
                                 + 1.16845 w_k1_FDX_s017
                                 + 0.753799 w_k1_IBM_s017
                                 + 1.15575 w_k1_INTC_s017
                                 + 0.776486 w_k1_KO_s017
                                 + 0.865575 w_k1_MAT_s017
                                 + 1.02742 w_k1_MSFT_s017
                                 + 0.946724 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.78866 w_k1_ADM_s017
                                 + 0.708281 w_k1_AXP_s017
                                 + 0.581146 w_k1_DIS_s017
                                 + 0.673182 w_k1_FDX_s017
                                 + 0.876374 w_k1_IBM_s017
                                 + 0.931646 w_k1_INTC_s017
                                 + 0.914077 w_k1_KO_s017
                                 + 0.917814 w_k1_MAT_s017
                                 + 0.78363 w_k1_MSFT_s017
                                 + 0.798708 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.05581 w_k1_ADM_s017
                                 + 0.826949 w_k1_AXP_s017
                                 + 0.806302 w_k1_DIS_s017
                                 + 0.880696 w_k1_FDX_s017
                                 + 0.825882 w_k1_IBM_s017
                                 + 0.914872 w_k1_INTC_s017
                                 + 0.804739 w_k1_KO_s017
                                 + 1.07348 w_k1_MAT_s017
                                 + 0.856739 w_k1_MSFT_s017
                                 + 0.833811 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.11521 w_k1_ADM_s017
                                 + 1.18709 w_k1_AXP_s017
                                 + 1.15139 w_k1_DIS_s017
                                 + 1.07803 w_k1_FDX_s017
                                 + 1.05592 w_k1_IBM_s017
                                 + 1.3995 w_k1_INTC_s017 + 1.05773 w_k1_KO_s017
                                 + 1.31986 w_k1_MAT_s017
                                 + 1.04118 w_k1_MSFT_s017
                                 + 1.03515 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.01343 w_k1_ADM_s017
                                 + 0.926891 w_k1_AXP_s017
                                 + 1.16709 w_k1_DIS_s017
                                 + 1.12264 w_k1_FDX_s017
                                 + 0.885488 w_k1_IBM_s017
                                 + 1.36377 w_k1_INTC_s017
                                 + 0.953032 w_k1_KO_s017
                                 + 0.972794 w_k1_MAT_s017
                                 + 1.10773 w_k1_MSFT_s017
                                 + 1.01366 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.26255 w_k1_ADM_s017
                                 + 1.49467 w_k1_AXP_s017
                                 + 1.53857 w_k1_DIS_s017
                                 + 1.29806 w_k1_FDX_s017
                                 + 1.43704 w_k1_IBM_s017
                                 + 1.52332 w_k1_INTC_s017
                                 + 1.72459 w_k1_KO_s017 + 1.95158 w_k1_MAT_s017
                                 + 1.17037 w_k1_MSFT_s017
                                 + 1.25745 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.884911 w_k1_ADM_s017
                                 + 0.940765 w_k1_AXP_s017
                                 + 0.794974 w_k1_DIS_s017
                                 + 0.849584 w_k1_FDX_s017
                                 + 1.23372 w_k1_IBM_s017
                                 + 0.979368 w_k1_INTC_s017
                                 + 0.980587 w_k1_KO_s017
                                 + 1.06813 w_k1_MAT_s017
                                 + 0.873345 w_k1_MSFT_s017
                                 + 0.937674 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.27581 w_k1_ADM_s017
                                 + 0.813903 w_k1_AXP_s017
                                 + 0.687577 w_k1_DIS_s017
                                 + 0.855402 w_k1_FDX_s017
                                 + 0.608215 w_k1_IBM_s017
                                 + 0.894504 w_k1_INTC_s017
                                 + 0.758069 w_k1_KO_s017
                                 + 0.822224 w_k1_MAT_s017
                                 + 0.840909 w_k1_MSFT_s017
                                 + 0.838457 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.86613 w_k1_ADM_s100
                                 + 0.927953 w_k1_AXP_s100
                                 + 0.944665 w_k1_DIS_s100
                                 + 0.90994 w_k1_FDX_s100
                                 + 0.892637 w_k1_IBM_s100
                                 + 0.83999 w_k1_INTC_s100
                                 + 0.996941 w_k1_KO_s100
                                 + 1.34367 w_k1_MAT_s100
                                 + 0.915814 w_k1_MSFT_s100
                                 + 0.872241 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.01587 w_k1_ADM_s100
                                 + 1.02795 w_k1_AXP_s100 + 1.2655 w_k1_DIS_s100
                                 + 1.04813 w_k1_FDX_s100
                                 + 1.45306 w_k1_IBM_s100
                                 + 1.17158 w_k1_INTC_s100
                                 + 1.40744 w_k1_KO_s100 + 1.24538 w_k1_MAT_s100
                                 + 1.14045 w_k1_MSFT_s100
                                 + 1.06997 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.547302 w_k1_ADM_s100
                                 + 0.731697 w_k1_AXP_s100
                                 + 0.62084 w_k1_DIS_s100
                                 + 0.735203 w_k1_FDX_s100
                                 + 0.740406 w_k1_IBM_s100
                                 + 0.900596 w_k1_INTC_s100
                                 + 0.720939 w_k1_KO_s100
                                 + 0.959753 w_k1_MAT_s100
                                 + 0.76008 w_k1_MSFT_s100
                                 + 0.802732 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.784274 w_k1_ADM_s100
                                 + 0.632791 w_k1_AXP_s100
                                 + 0.749024 w_k1_DIS_s100
                                 + 0.734657 w_k1_FDX_s100
                                 + 0.886776 w_k1_IBM_s100
                                 + 0.774046 w_k1_INTC_s100
                                 + 0.844971 w_k1_KO_s100
                                 + 0.903409 w_k1_MAT_s100
                                 + 0.780412 w_k1_MSFT_s100
                                 + 0.796876 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.30917 w_k1_ADM_s100
                                 + 1.32676 w_k1_AXP_s100
                                 + 1.53611 w_k1_DIS_s100
                                 + 1.45431 w_k1_FDX_s100
                                 + 1.01745 w_k1_IBM_s100
                                 + 1.53021 w_k1_INTC_s100
                                 + 1.40482 w_k1_KO_s100 + 1.44996 w_k1_MAT_s100
                                 + 1.41264 w_k1_MSFT_s100 + 1.116 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.03832 w_k1_ADM_s100
                                 + 0.750439 w_k1_AXP_s100
                                 + 0.948489 w_k1_DIS_s100
                                 + 1.06794 w_k1_FDX_s100
                                 + 0.592004 w_k1_IBM_s100
                                 + 1.02471 w_k1_INTC_s100
                                 + 0.82529 w_k1_KO_s100 + 0.43323 w_k1_MAT_s100
                                 + 1.03052 w_k1_MSFT_s100
                                 + 0.922982 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.08303 w_k1_ADM_s100
                                 + 1.14107 w_k1_AXP_s100
                                 + 1.18586 w_k1_DIS_s100
                                 + 1.22684 w_k1_FDX_s100
                                 + 0.824418 w_k1_IBM_s100
                                 + 1.24955 w_k1_INTC_s100
                                 + 0.932053 w_k1_KO_s100
                                 + 0.891683 w_k1_MAT_s100
                                 + 1.04305 w_k1_MSFT_s100
                                 + 1.09068 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.848192 w_k1_ADM_s100
                                 + 0.891852 w_k1_AXP_s100
                                 + 0.785925 w_k1_DIS_s100
                                 + 0.826809 w_k1_FDX_s100
                                 + 1.29905 w_k1_IBM_s100
                                 + 0.981715 w_k1_INTC_s100
                                 + 1.06793 w_k1_KO_s100 + 1.57937 w_k1_MAT_s100
                                 + 0.908903 w_k1_MSFT_s100
                                 + 0.921804 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01084 w_k1_ADM_s101
                                 + 0.708958 w_k1_AXP_s101
                                 + 0.674123 w_k1_DIS_s101
                                 + 0.837696 w_k1_FDX_s101
                                 + 0.681218 w_k1_IBM_s101
                                 + 0.730671 w_k1_INTC_s101
                                 + 0.745061 w_k1_KO_s101
                                 + 0.627187 w_k1_MAT_s101
                                 + 0.864093 w_k1_MSFT_s101
                                 + 0.831534 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.0102 w_k1_ADM_s101
                                 + 0.951731 w_k1_AXP_s101
                                 + 0.98871 w_k1_DIS_s101
                                 + 1.11836 w_k1_FDX_s101
                                 + 0.76978 w_k1_IBM_s101
                                 + 1.36435 w_k1_INTC_s101 + 0.9647 w_k1_KO_s101
                                 + 0.840591 w_k1_MAT_s101
                                 + 1.21431 w_k1_MSFT_s101
                                 + 1.05735 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.09242 w_k1_ADM_s101
                                 + 1.24779 w_k1_AXP_s101
                                 + 1.01171 w_k1_DIS_s101
                                 + 1.06373 w_k1_FDX_s101
                                 + 1.25216 w_k1_IBM_s101 + 1.016 w_k1_INTC_s101
                                 + 1.03367 w_k1_KO_s101 + 1.55911 w_k1_MAT_s101
                                 + 1.07488 w_k1_MSFT_s101
                                 + 0.985431 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.27524 w_k1_ADM_s101
                                 + 1.45145 w_k1_AXP_s101
                                 + 1.42439 w_k1_DIS_s101
                                 + 1.34031 w_k1_FDX_s101
                                 + 0.980415 w_k1_IBM_s101
                                 + 1.26322 w_k1_INTC_s101
                                 + 1.10367 w_k1_KO_s101 + 1.23265 w_k1_MAT_s101
                                 + 1.35555 w_k1_MSFT_s101
                                 + 1.17719 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.927653 w_k1_ADM_s101
                                 + 1.00999 w_k1_AXP_s101
                                 + 1.23195 w_k1_DIS_s101
                                 + 0.974008 w_k1_FDX_s101
                                 + 1.41282 w_k1_IBM_s101
                                 + 1.17116 w_k1_INTC_s101
                                 + 1.30607 w_k1_KO_s101 + 1.23611 w_k1_MAT_s101
                                 + 1.15114 w_k1_MSFT_s101
                                 + 1.10056 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.963102 w_k1_ADM_s101
                                 + 0.714053 w_k1_AXP_s101
                                 + 0.782454 w_k1_DIS_s101
                                 + 0.736945 w_k1_FDX_s101
                                 + 1.08774 w_k1_IBM_s101
                                 + 0.859602 w_k1_INTC_s101
                                 + 0.936152 w_k1_KO_s101
                                 + 0.848248 w_k1_MAT_s101
                                 + 0.922254 w_k1_MSFT_s101
                                 + 0.921546 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.40669 w_k1_ADM_s101
                                 + 1.68115 w_k1_AXP_s101
                                 + 1.39903 w_k1_DIS_s101
                                 + 1.33824 w_k1_FDX_s101
                                 + 1.70402 w_k1_IBM_s101
                                 + 1.3741 w_k1_INTC_s101 + 1.49755 w_k1_KO_s101
                                 + 1.86521 w_k1_MAT_s101
                                 + 1.23862 w_k1_MSFT_s101
                                 + 1.37659 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.971195 w_k1_ADM_s101
                                 + 1.26914 w_k1_AXP_s101
                                 + 0.992761 w_k1_DIS_s101
                                 + 0.971109 w_k1_FDX_s101
                                 + 0.864706 w_k1_IBM_s101
                                 + 0.981483 w_k1_INTC_s101
                                 + 1.01032 w_k1_KO_s101 + 1.27753 w_k1_MAT_s101
                                 + 1.06323 w_k1_MSFT_s101
                                 + 1.09508 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.89154 w_k1_ADM_s102
                                 + 0.968918 w_k1_AXP_s102
                                 + 0.790984 w_k1_DIS_s102
                                 + 0.729382 w_k1_FDX_s102
                                 + 1.14399 w_k1_IBM_s102
                                 + 0.677809 w_k1_INTC_s102
                                 + 1.0441 w_k1_KO_s102 + 1.52622 w_k1_MAT_s102
                                 + 0.762955 w_k1_MSFT_s102
                                 + 0.893614 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.02136 w_k1_ADM_s102
                                 + 0.85152 w_k1_AXP_s102
                                 + 0.750777 w_k1_DIS_s102
                                 + 0.926176 w_k1_FDX_s102
                                 + 0.46848 w_k1_IBM_s102
                                 + 0.83765 w_k1_INTC_s102
                                 + 0.644867 w_k1_KO_s102
                                 + 0.453716 w_k1_MAT_s102
                                 + 0.842568 w_k1_MSFT_s102
                                 + 0.752476 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.03442 w_k1_ADM_s102
                                 + 0.897421 w_k1_AXP_s102
                                 + 0.908772 w_k1_DIS_s102
                                 + 0.936411 w_k1_FDX_s102
                                 + 0.716417 w_k1_IBM_s102
                                 + 1.0031 w_k1_INTC_s102
                                 + 0.924225 w_k1_KO_s102
                                 + 0.683051 w_k1_MAT_s102
                                 + 1.03855 w_k1_MSFT_s102
                                 + 0.880477 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.907211 w_k1_ADM_s102
                                 + 1.35991 w_k1_AXP_s102
                                 + 1.15132 w_k1_DIS_s102
                                 + 1.14747 w_k1_FDX_s102
                                 + 1.01182 w_k1_IBM_s102
                                 + 1.08083 w_k1_INTC_s102
                                 + 0.967492 w_k1_KO_s102
                                 + 1.17769 w_k1_MAT_s102
                                 + 1.0699 w_k1_MSFT_s102 + 1.06215 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.976278 w_k1_ADM_s102
                                 + 0.8192 w_k1_AXP_s102
                                 + 0.831454 w_k1_DIS_s102
                                 + 0.866132 w_k1_FDX_s102
                                 + 0.907893 w_k1_IBM_s102
                                 + 0.801975 w_k1_INTC_s102
                                 + 0.832879 w_k1_KO_s102 + 1.1192 w_k1_MAT_s102
                                 + 0.933087 w_k1_MSFT_s102
                                 + 0.897482 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.0002 w_k1_ADM_s102
                                 + 1.08969 w_k1_AXP_s102
                                 + 1.20272 w_k1_DIS_s102
                                 + 0.981637 w_k1_FDX_s102
                                 + 1.39172 w_k1_IBM_s102
                                 + 0.984395 w_k1_INTC_s102
                                 + 1.26375 w_k1_KO_s102 + 1.52261 w_k1_MAT_s102
                                 + 1.12628 w_k1_MSFT_s102
                                 + 1.13094 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.62098 w_k1_ADM_s102
                                 + 1.46022 w_k1_AXP_s102
                                 + 1.51498 w_k1_DIS_s102
                                 + 1.28361 w_k1_FDX_s102
                                 + 0.829417 w_k1_IBM_s102
                                 + 1.11557 w_k1_INTC_s102
                                 + 1.18454 w_k1_KO_s102 + 1.23159 w_k1_MAT_s102
                                 + 1.18162 w_k1_MSFT_s102
                                 + 1.12221 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.95784 w_k1_ADM_s102
                                 + 0.900151 w_k1_AXP_s102
                                 + 0.646655 w_k1_DIS_s102
                                 + 0.807064 w_k1_FDX_s102
                                 + 0.657509 w_k1_IBM_s102
                                 + 0.963539 w_k1_INTC_s102
                                 + 0.727058 w_k1_KO_s102
                                 + 0.500848 w_k1_MAT_s102
                                 + 0.897074 w_k1_MSFT_s102
                                 + 0.849512 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.19355 w_k1_ADM_s103
                                 + 1.42347 w_k1_AXP_s103 + 1.35 w_k1_DIS_s103
                                 + 1.13777 w_k1_FDX_s103
                                 + 1.50583 w_k1_IBM_s103
                                 + 1.03476 w_k1_INTC_s103
                                 + 1.39372 w_k1_KO_s103 + 1.64374 w_k1_MAT_s103
                                 + 1.15311 w_k1_MSFT_s103 + 1.176 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.22229 w_k1_ADM_s103
                                 + 1.19649 w_k1_AXP_s103
                                 + 1.13793 w_k1_DIS_s103 + 1.249 w_k1_FDX_s103
                                 + 0.639228 w_k1_IBM_s103
                                 + 1.27279 w_k1_INTC_s103
                                 + 0.891906 w_k1_KO_s103
                                 + 0.640866 w_k1_MAT_s103
                                 + 1.1321 w_k1_MSFT_s103 + 1.12497 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.07137 w_k1_ADM_s103
                                 + 0.904633 w_k1_AXP_s103
                                 + 0.678596 w_k1_DIS_s103
                                 + 0.863851 w_k1_FDX_s103
                                 + 0.641745 w_k1_IBM_s103
                                 + 0.948101 w_k1_INTC_s103
                                 + 0.769574 w_k1_KO_s103
                                 + 0.74274 w_k1_MAT_s103
                                 + 0.945391 w_k1_MSFT_s103
                                 + 0.915542 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.775908 w_k1_ADM_s103
                                 + 0.889319 w_k1_AXP_s103
                                 + 0.613455 w_k1_DIS_s103
                                 + 0.719768 w_k1_FDX_s103
                                 + 1.05313 w_k1_IBM_s103
                                 + 0.876246 w_k1_INTC_s103
                                 + 0.928755 w_k1_KO_s103
                                 + 1.06452 w_k1_MAT_s103
                                 + 0.740903 w_k1_MSFT_s103
                                 + 0.939882 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.26847 w_k1_ADM_s103
                                 + 1.47237 w_k1_AXP_s103
                                 + 1.57185 w_k1_DIS_s103
                                 + 1.43285 w_k1_FDX_s103
                                 + 0.892379 w_k1_IBM_s103
                                 + 1.29485 w_k1_INTC_s103
                                 + 1.08756 w_k1_KO_s103
                                 + 0.967432 w_k1_MAT_s103
                                 + 1.34425 w_k1_MSFT_s103
                                 + 1.00064 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.994153 w_k1_ADM_s103
                                 + 0.900581 w_k1_AXP_s103
                                 + 0.957833 w_k1_DIS_s103
                                 + 0.991908 w_k1_FDX_s103
                                 + 0.898418 w_k1_IBM_s103
                                 + 1.01442 w_k1_INTC_s103
                                 + 1.06202 w_k1_KO_s103
                                 + 0.662354 w_k1_MAT_s103
                                 + 1.05967 w_k1_MSFT_s103 + 1.0091 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.906838 w_k1_ADM_s103
                                 + 0.741773 w_k1_AXP_s103
                                 + 0.68679 w_k1_DIS_s103
                                 + 0.806394 w_k1_FDX_s103
                                 + 0.851917 w_k1_IBM_s103
                                 + 0.860982 w_k1_INTC_s103
                                 + 0.731031 w_k1_KO_s103
                                 + 0.828348 w_k1_MAT_s103
                                 + 0.84374 w_k1_MSFT_s103
                                 + 0.886456 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.70591 w_k1_ADM_s103
                                 + 0.867453 w_k1_AXP_s103
                                 + 0.943588 w_k1_DIS_s103
                                 + 0.791933 w_k1_FDX_s103
                                 + 1.28398 w_k1_IBM_s103
                                 + 0.936775 w_k1_INTC_s103
                                 + 1.10983 w_k1_KO_s103
                                 + 0.852494 w_k1_MAT_s103
                                 + 0.830139 w_k1_MSFT_s103
                                 + 1.02678 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.963909 w_k1_ADM_s104
                                 + 1.21578 w_k1_AXP_s104
                                 + 1.14393 w_k1_DIS_s104
                                 + 1.17031 w_k1_FDX_s104
                                 + 0.946425 w_k1_IBM_s104
                                 + 1.17255 w_k1_INTC_s104
                                 + 1.03556 w_k1_KO_s104 + 1.10609 w_k1_MAT_s104
                                 + 1.0366 w_k1_MSFT_s104
                                 + 0.935959 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.910366 w_k1_ADM_s104
                                 + 1.00687 w_k1_AXP_s104
                                 + 0.802581 w_k1_DIS_s104
                                 + 0.965627 w_k1_FDX_s104
                                 + 0.827104 w_k1_IBM_s104
                                 + 1.17989 w_k1_INTC_s104
                                 + 0.807602 w_k1_KO_s104
                                 + 0.976588 w_k1_MAT_s104
                                 + 0.992575 w_k1_MSFT_s104
                                 + 0.874607 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.16779 w_k1_ADM_s104
                                 + 1.70032 w_k1_AXP_s104
                                 + 1.53502 w_k1_DIS_s104
                                 + 1.50115 w_k1_FDX_s104
                                 + 1.12494 w_k1_IBM_s104
                                 + 1.33714 w_k1_INTC_s104 + 1.2118 w_k1_KO_s104
                                 + 1.66889 w_k1_MAT_s104
                                 + 1.40514 w_k1_MSFT_s104
                                 + 1.31609 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.806333 w_k1_ADM_s104
                                 + 0.746306 w_k1_AXP_s104
                                 + 0.940661 w_k1_DIS_s104
                                 + 0.911071 w_k1_FDX_s104
                                 + 0.587385 w_k1_IBM_s104
                                 + 1.00522 w_k1_INTC_s104
                                 + 0.759775 w_k1_KO_s104
                                 + 0.392788 w_k1_MAT_s104
                                 + 0.980329 w_k1_MSFT_s104
                                 + 0.855423 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.796363 w_k1_ADM_s104
                                 + 0.707963 w_k1_AXP_s104
                                 + 0.707625 w_k1_DIS_s104
                                 + 0.748827 w_k1_FDX_s104
                                 + 0.711807 w_k1_IBM_s104
                                 + 0.894459 w_k1_INTC_s104
                                 + 0.752348 w_k1_KO_s104
                                 + 0.51432 w_k1_MAT_s104
                                 + 0.887068 w_k1_MSFT_s104
                                 + 0.827217 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.01678 w_k1_ADM_s104
                                 + 1.07211 w_k1_AXP_s104
                                 + 1.39305 w_k1_DIS_s104 + 1.1254 w_k1_FDX_s104
                                 + 1.46035 w_k1_IBM_s104
                                 + 1.07855 w_k1_INTC_s104
                                 + 1.15342 w_k1_KO_s104 + 1.38422 w_k1_MAT_s104
                                 + 1.25749 w_k1_MSFT_s104
                                 + 1.08632 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.977621 w_k1_ADM_s104
                                 + 0.847403 w_k1_AXP_s104
                                 + 1.22263 w_k1_DIS_s104
                                 + 1.10824 w_k1_FDX_s104
                                 + 0.818671 w_k1_IBM_s104
                                 + 1.14113 w_k1_INTC_s104
                                 + 0.950628 w_k1_KO_s104
                                 + 0.625414 w_k1_MAT_s104
                                 + 1.12429 w_k1_MSFT_s104
                                 + 0.992483 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.779778 w_k1_ADM_s104
                                 + 1.01902 w_k1_AXP_s104
                                 + 0.759381 w_k1_DIS_s104
                                 + 0.847864 w_k1_FDX_s104
                                 + 1.15835 w_k1_IBM_s104
                                 + 0.901694 w_k1_INTC_s104
                                 + 0.892683 w_k1_KO_s104
                                 + 1.29161 w_k1_MAT_s104
                                 + 0.851213 w_k1_MSFT_s104
                                 + 0.92555 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.952159 w_k1_ADM_s105
                                 + 0.727166 w_k1_AXP_s105
                                 + 0.884533 w_k1_DIS_s105
                                 + 0.930441 w_k1_FDX_s105
                                 + 0.981793 w_k1_IBM_s105
                                 + 0.927707 w_k1_INTC_s105
                                 + 1.03337 w_k1_KO_s105
                                 + 0.914897 w_k1_MAT_s105
                                 + 0.934019 w_k1_MSFT_s105
                                 + 0.950492 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.916243 w_k1_ADM_s105
                                 + 1.10299 w_k1_AXP_s105
                                 + 0.877501 w_k1_DIS_s105
                                 + 0.996791 w_k1_FDX_s105
                                 + 0.939537 w_k1_IBM_s105
                                 + 1.07041 w_k1_INTC_s105
                                 + 0.808697 w_k1_KO_s105
                                 + 0.861318 w_k1_MAT_s105
                                 + 0.828128 w_k1_MSFT_s105
                                 + 0.864679 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.10133 w_k1_ADM_s105
                                 + 0.78152 w_k1_AXP_s105 + 1.0494 w_k1_DIS_s105
                                 + 1.07019 w_k1_FDX_s105
                                 + 0.658803 w_k1_IBM_s105
                                 + 0.880575 w_k1_INTC_s105
                                 + 0.924596 w_k1_KO_s105
                                 + 0.755661 w_k1_MAT_s105
                                 + 1.04987 w_k1_MSFT_s105
                                 + 0.969305 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.894965 w_k1_ADM_s105
                                 + 0.69174 w_k1_AXP_s105
                                 + 0.691979 w_k1_DIS_s105
                                 + 0.761406 w_k1_FDX_s105
                                 + 0.816653 w_k1_IBM_s105
                                 + 0.672796 w_k1_INTC_s105
                                 + 0.724122 w_k1_KO_s105
                                 + 0.877674 w_k1_MAT_s105
                                 + 0.750527 w_k1_MSFT_s105
                                 + 0.802328 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.31165 w_k1_ADM_s105
                                 + 1.45081 w_k1_AXP_s105
                                 + 1.70955 w_k1_DIS_s105
                                 + 1.41263 w_k1_FDX_s105
                                 + 1.36286 w_k1_IBM_s105
                                 + 1.27256 w_k1_INTC_s105 + 1.3597 w_k1_KO_s105
                                 + 1.08065 w_k1_MAT_s105
                                 + 1.37527 w_k1_MSFT_s105
                                 + 1.27452 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.32151 w_k1_ADM_s105
                                 + 1.35803 w_k1_AXP_s105
                                 + 0.985606 w_k1_DIS_s105
                                 + 1.27256 w_k1_FDX_s105
                                 + 1.03087 w_k1_IBM_s105
                                 + 1.17137 w_k1_INTC_s105
                                 + 0.988141 w_k1_KO_s105
                                 + 1.11815 w_k1_MAT_s105
                                 + 1.04456 w_k1_MSFT_s105
                                 + 1.04325 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.02413 w_k1_ADM_s105
                                 + 1.13123 w_k1_AXP_s105
                                 + 0.941585 w_k1_DIS_s105
                                 + 0.94907 w_k1_FDX_s105
                                 + 1.34465 w_k1_IBM_s105
                                 + 1.01917 w_k1_INTC_s105
                                 + 1.11989 w_k1_KO_s105 + 1.11151 w_k1_MAT_s105
                                 + 0.837439 w_k1_MSFT_s105
                                 + 0.968333 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.24808 w_k1_ADM_s105
                                 + 1.08528 w_k1_AXP_s105 + 1.1917 w_k1_DIS_s105
                                 + 1.13713 w_k1_FDX_s105
                                 + 1.09728 w_k1_IBM_s105
                                 + 1.16476 w_k1_INTC_s105
                                 + 1.18059 w_k1_KO_s105 + 1.21213 w_k1_MAT_s105
                                 + 1.15473 w_k1_MSFT_s105 + 1.1649 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.825648 w_k1_ADM_s106
                                 + 0.737587 w_k1_AXP_s106
                                 + 0.704082 w_k1_DIS_s106
                                 + 0.777705 w_k1_FDX_s106
                                 + 0.967042 w_k1_IBM_s106
                                 + 0.726338 w_k1_INTC_s106
                                 + 0.76898 w_k1_KO_s106
                                 + 0.568215 w_k1_MAT_s106
                                 + 0.79284 w_k1_MSFT_s106
                                 + 0.848632 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.834986 w_k1_ADM_s106
                                 + 1.14371 w_k1_AXP_s106 + 1.3652 w_k1_DIS_s106
                                 + 1.15282 w_k1_FDX_s106
                                 + 1.18945 w_k1_IBM_s106
                                 + 1.20365 w_k1_INTC_s106
                                 + 1.24924 w_k1_KO_s106
                                 + 0.921888 w_k1_MAT_s106
                                 + 1.1577 w_k1_MSFT_s106 + 1.0909 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.00121 w_k1_ADM_s106
                                 + 0.813062 w_k1_AXP_s106
                                 + 0.909288 w_k1_DIS_s106
                                 + 1.0888 w_k1_FDX_s106
                                 + 0.570049 w_k1_IBM_s106
                                 + 0.978405 w_k1_INTC_s106
                                 + 0.758143 w_k1_KO_s106
                                 + 0.345509 w_k1_MAT_s106
                                 + 1.0169 w_k1_MSFT_s106 + 0.94841 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.06918 w_k1_ADM_s106
                                 + 0.900626 w_k1_AXP_s106
                                 + 1.22207 w_k1_DIS_s106
                                 + 1.01973 w_k1_FDX_s106
                                 + 1.07889 w_k1_IBM_s106
                                 + 1.07196 w_k1_INTC_s106
                                 + 1.14893 w_k1_KO_s106
                                 + 0.607604 w_k1_MAT_s106
                                 + 1.19861 w_k1_MSFT_s106
                                 + 1.01193 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.09395 w_k1_ADM_s106
                                 + 1.28855 w_k1_AXP_s106
                                 + 1.19822 w_k1_DIS_s106
                                 + 1.11812 w_k1_FDX_s106 + 1.6643 w_k1_IBM_s106
                                 + 0.957202 w_k1_INTC_s106
                                 + 1.31522 w_k1_KO_s106 + 1.53693 w_k1_MAT_s106
                                 + 1.10727 w_k1_MSFT_s106
                                 + 1.14441 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.811647 w_k1_ADM_s106
                                 + 0.77145 w_k1_AXP_s106
                                 + 0.809752 w_k1_DIS_s106
                                 + 0.8928 w_k1_FDX_s106
                                 + 0.754738 w_k1_IBM_s106
                                 + 0.898534 w_k1_INTC_s106
                                 + 0.792301 w_k1_KO_s106
                                 + 0.540944 w_k1_MAT_s106
                                 + 0.840547 w_k1_MSFT_s106
                                 + 0.80051 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.05696 w_k1_ADM_s106
                                 + 1.18982 w_k1_AXP_s106
                                 + 0.915573 w_k1_DIS_s106
                                 + 0.931742 w_k1_FDX_s106
                                 + 1.20053 w_k1_IBM_s106
                                 + 0.753985 w_k1_INTC_s106
                                 + 0.851496 w_k1_KO_s106 + 1.07 w_k1_MAT_s106
                                 + 0.854442 w_k1_MSFT_s106
                                 + 1.00396 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.33594 w_k1_ADM_s106
                                 + 1.30006 w_k1_AXP_s106
                                 + 1.14639 w_k1_DIS_s106
                                 + 1.16561 w_k1_FDX_s106
                                 + 0.897213 w_k1_IBM_s106
                                 + 1.03766 w_k1_INTC_s106
                                 + 0.990105 w_k1_KO_s106
                                 + 0.946975 w_k1_MAT_s106
                                 + 1.10623 w_k1_MSFT_s106
                                 + 1.13317 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.941532 w_k1_ADM_s107
                                 + 0.934135 w_k1_AXP_s107
                                 + 1.31433 w_k1_DIS_s107
                                 + 1.12499 w_k1_FDX_s107
                                 + 0.875497 w_k1_IBM_s107
                                 + 1.02086 w_k1_INTC_s107
                                 + 1.03109 w_k1_KO_s107
                                 + 0.789809 w_k1_MAT_s107
                                 + 1.1483 w_k1_MSFT_s107
                                 + 0.984184 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.08613 w_k1_ADM_s107
                                 + 0.892096 w_k1_AXP_s107
                                 + 0.871521 w_k1_DIS_s107
                                 + 0.916186 w_k1_FDX_s107
                                 + 1.24821 w_k1_IBM_s107
                                 + 0.964779 w_k1_INTC_s107
                                 + 1.12773 w_k1_KO_s107 + 1.28622 w_k1_MAT_s107
                                 + 0.905317 w_k1_MSFT_s107
                                 + 1.02909 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.84521 w_k1_ADM_s107
                                 + 0.807906 w_k1_AXP_s107
                                 + 0.751515 w_k1_DIS_s107
                                 + 0.861066 w_k1_FDX_s107
                                 + 0.693564 w_k1_IBM_s107
                                 + 0.808478 w_k1_INTC_s107
                                 + 0.795751 w_k1_KO_s107
                                 + 0.611844 w_k1_MAT_s107
                                 + 0.818754 w_k1_MSFT_s107
                                 + 0.897409 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.29678 w_k1_ADM_s107
                                 + 1.65672 w_k1_AXP_s107
                                 + 1.39568 w_k1_DIS_s107
                                 + 1.37721 w_k1_FDX_s107
                                 + 1.03731 w_k1_IBM_s107
                                 + 1.17829 w_k1_INTC_s107
                                 + 1.17516 w_k1_KO_s107 + 1.06212 w_k1_MAT_s107
                                 + 1.26334 w_k1_MSFT_s107
                                 + 1.25456 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.27443 w_k1_ADM_s107
                                 + 1.2064 w_k1_AXP_s107 + 1.36647 w_k1_DIS_s107
                                 + 1.07907 w_k1_FDX_s107 + 1.6024 w_k1_IBM_s107
                                 + 1.11543 w_k1_INTC_s107
                                 + 1.44307 w_k1_KO_s107 + 1.59658 w_k1_MAT_s107
                                 + 1.14058 w_k1_MSFT_s107
                                 + 1.21861 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.4987 w_k1_ADM_s107
                                 + 1.54956 w_k1_AXP_s107
                                 + 1.77354 w_k1_DIS_s107
                                 + 1.47633 w_k1_FDX_s107
                                 + 1.58028 w_k1_IBM_s107
                                 + 1.43282 w_k1_INTC_s107
                                 + 1.54111 w_k1_KO_s107 + 1.30101 w_k1_MAT_s107
                                 + 1.49191 w_k1_MSFT_s107
                                 + 1.33507 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.984574 w_k1_ADM_s107
                                 + 0.864148 w_k1_AXP_s107
                                 + 0.919273 w_k1_DIS_s107
                                 + 0.877794 w_k1_FDX_s107
                                 + 0.995086 w_k1_IBM_s107
                                 + 0.878357 w_k1_INTC_s107
                                 + 0.968304 w_k1_KO_s107
                                 + 0.68046 w_k1_MAT_s107
                                 + 0.940705 w_k1_MSFT_s107
                                 + 0.998744 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.30169 w_k1_ADM_s107
                                 + 1.12733 w_k1_AXP_s107
                                 + 1.21481 w_k1_DIS_s107
                                 + 1.15082 w_k1_FDX_s107
                                 + 1.09862 w_k1_IBM_s107
                                 + 0.898511 w_k1_INTC_s107
                                 + 1.25983 w_k1_KO_s107 + 1.22937 w_k1_MAT_s107
                                 + 1.12854 w_k1_MSFT_s107
                                 + 1.21251 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.896485 w_k1_ADM_s110
                                 + 1.02846 w_k1_AXP_s110
                                 + 1.26605 w_k1_DIS_s110
                                 + 0.99895 w_k1_FDX_s110
                                 + 1.34744 w_k1_IBM_s110
                                 + 1.08341 w_k1_INTC_s110
                                 + 1.08738 w_k1_KO_s110 + 1.15214 w_k1_MAT_s110
                                 + 1.13816 w_k1_MSFT_s110
                                 + 1.09525 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.956733 w_k1_ADM_s110
                                 + 1.64685 w_k1_AXP_s110
                                 + 1.73431 w_k1_DIS_s110
                                 + 1.44356 w_k1_FDX_s110
                                 + 1.53777 w_k1_IBM_s110
                                 + 1.3478 w_k1_INTC_s110 + 1.50775 w_k1_KO_s110
                                 + 1.88308 w_k1_MAT_s110
                                 + 1.18051 w_k1_MSFT_s110
                                 + 1.29579 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.85101 w_k1_ADM_s110
                                 + 0.77814 w_k1_AXP_s110
                                 + 0.751994 w_k1_DIS_s110
                                 + 0.855304 w_k1_FDX_s110
                                 + 0.749278 w_k1_IBM_s110
                                 + 0.98706 w_k1_INTC_s110
                                 + 0.794275 w_k1_KO_s110
                                 + 0.825799 w_k1_MAT_s110
                                 + 0.995897 w_k1_MSFT_s110
                                 + 0.975893 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.03653 w_k1_ADM_s110
                                 + 1.07584 w_k1_AXP_s110
                                 + 1.09893 w_k1_DIS_s110
                                 + 1.12088 w_k1_FDX_s110
                                 + 0.961987 w_k1_IBM_s110
                                 + 1.28223 w_k1_INTC_s110
                                 + 0.959408 w_k1_KO_s110
                                 + 0.912346 w_k1_MAT_s110
                                 + 1.07917 w_k1_MSFT_s110
                                 + 1.01748 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.627485 w_k1_ADM_s110
                                 + 0.672124 w_k1_AXP_s110
                                 + 0.610794 w_k1_DIS_s110
                                 + 0.718999 w_k1_FDX_s110
                                 + 0.775735 w_k1_IBM_s110
                                 + 0.759489 w_k1_INTC_s110
                                 + 0.666538 w_k1_KO_s110
                                 + 0.920238 w_k1_MAT_s110
                                 + 0.609519 w_k1_MSFT_s110
                                 + 0.760823 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.08289 w_k1_ADM_s110
                                 + 1.45482 w_k1_AXP_s110
                                 + 1.35022 w_k1_DIS_s110
                                 + 1.40671 w_k1_FDX_s110
                                 + 1.11417 w_k1_IBM_s110
                                 + 1.30156 w_k1_INTC_s110
                                 + 1.05801 w_k1_KO_s110 + 1.16905 w_k1_MAT_s110
                                 + 1.14112 w_k1_MSFT_s110 + 1.1872 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.963592 w_k1_ADM_s110
                                 + 0.923878 w_k1_AXP_s110
                                 + 0.906136 w_k1_DIS_s110
                                 + 1.06791 w_k1_FDX_s110
                                 + 0.716666 w_k1_IBM_s110
                                 + 0.953754 w_k1_INTC_s110
                                 + 0.878631 w_k1_KO_s110
                                 + 0.536733 w_k1_MAT_s110
                                 + 0.979302 w_k1_MSFT_s110
                                 + 0.969617 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.763392 w_k1_ADM_s110
                                 + 0.946698 w_k1_AXP_s110
                                 + 0.76775 w_k1_DIS_s110
                                 + 0.877459 w_k1_FDX_s110
                                 + 1.16162 w_k1_IBM_s110
                                 + 1.10692 w_k1_INTC_s110
                                 + 0.922061 w_k1_KO_s110
                                 + 1.12276 w_k1_MAT_s110
                                 + 0.922839 w_k1_MSFT_s110
                                 + 0.964051 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.888288 w_k1_ADM_s111
                                 + 0.748821 w_k1_AXP_s111
                                 + 0.746922 w_k1_DIS_s111
                                 + 0.823744 w_k1_FDX_s111
                                 + 0.748256 w_k1_IBM_s111
                                 + 0.801793 w_k1_INTC_s111
                                 + 0.707892 w_k1_KO_s111
                                 + 0.632005 w_k1_MAT_s111
                                 + 0.845822 w_k1_MSFT_s111
                                 + 0.849243 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.07428 w_k1_ADM_s111
                                 + 1.3268 w_k1_AXP_s111 + 1.28096 w_k1_DIS_s111
                                 + 1.25977 w_k1_FDX_s111
                                 + 0.926805 w_k1_IBM_s111
                                 + 1.15139 w_k1_INTC_s111 + 1.1116 w_k1_KO_s111
                                 + 1.0354 w_k1_MAT_s111
                                 + 1.15342 w_k1_MSFT_s111
                                 + 1.02961 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.11013 w_k1_ADM_s111
                                 + 1.26301 w_k1_AXP_s111 + 1.7438 w_k1_DIS_s111
                                 + 1.51031 w_k1_FDX_s111
                                 + 1.22596 w_k1_IBM_s111
                                 + 1.49374 w_k1_INTC_s111
                                 + 1.20279 w_k1_KO_s111
                                 + 0.907356 w_k1_MAT_s111
                                 + 1.32962 w_k1_MSFT_s111
                                 + 1.18201 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.920904 w_k1_ADM_s111
                                 + 0.988642 w_k1_AXP_s111
                                 + 0.92043 w_k1_DIS_s111
                                 + 0.897897 w_k1_FDX_s111
                                 + 1.17267 w_k1_IBM_s111
                                 + 1.09656 w_k1_INTC_s111
                                 + 1.00645 w_k1_KO_s111 + 0.81389 w_k1_MAT_s111
                                 + 1.02114 w_k1_MSFT_s111
                                 + 0.996638 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.06071 w_k1_ADM_s111
                                 + 1.78169 w_k1_AXP_s111
                                 + 1.08782 w_k1_DIS_s111
                                 + 1.09705 w_k1_FDX_s111
                                 + 1.15177 w_k1_IBM_s111
                                 + 1.1247 w_k1_INTC_s111 + 1.01686 w_k1_KO_s111
                                 + 1.56182 w_k1_MAT_s111
                                 + 1.00846 w_k1_MSFT_s111
                                 + 1.24852 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.98837 w_k1_ADM_s111
                                 + 1.06765 w_k1_AXP_s111
                                 + 1.45723 w_k1_DIS_s111
                                 + 1.15868 w_k1_FDX_s111
                                 + 0.993472 w_k1_IBM_s111
                                 + 1.16465 w_k1_INTC_s111
                                 + 1.17959 w_k1_KO_s111
                                 + 0.887469 w_k1_MAT_s111
                                 + 1.22703 w_k1_MSFT_s111
                                 + 1.07688 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.08992 w_k1_ADM_s111
                                 + 1.34162 w_k1_AXP_s111
                                 + 1.41359 w_k1_DIS_s111
                                 + 1.07077 w_k1_FDX_s111
                                 + 1.60698 w_k1_IBM_s111
                                 + 1.2156 w_k1_INTC_s111 + 1.22111 w_k1_KO_s111
                                 + 1.27272 w_k1_MAT_s111
                                 + 1.14666 w_k1_MSFT_s111
                                 + 1.05658 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.13268 w_k1_ADM_s111
                                 + 1.02659 w_k1_AXP_s111
                                 + 1.22086 w_k1_DIS_s111
                                 + 1.11315 w_k1_FDX_s111
                                 + 0.822381 w_k1_IBM_s111
                                 + 1.07844 w_k1_INTC_s111
                                 + 0.890528 w_k1_KO_s111
                                 + 0.612117 w_k1_MAT_s111
                                 + 1.14771 w_k1_MSFT_s111
                                 + 1.04982 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.50916 w_k1_ADM_s112
                                 + 1.37417 w_k1_AXP_s112
                                 + 1.52051 w_k1_DIS_s112
                                 + 1.19238 w_k1_FDX_s112
                                 + 1.47011 w_k1_IBM_s112
                                 + 1.02469 w_k1_INTC_s112
                                 + 1.33656 w_k1_KO_s112 + 1.55339 w_k1_MAT_s112
                                 + 1.0963 w_k1_MSFT_s112 + 1.28663 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.722121 w_k1_ADM_s112
                                 + 0.799926 w_k1_AXP_s112
                                 + 0.695567 w_k1_DIS_s112
                                 + 0.796096 w_k1_FDX_s112
                                 + 1.02066 w_k1_IBM_s112
                                 + 0.699212 w_k1_INTC_s112
                                 + 0.880741 w_k1_KO_s112
                                 + 0.879325 w_k1_MAT_s112
                                 + 0.624451 w_k1_MSFT_s112
                                 + 0.911186 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.874869 w_k1_ADM_s112
                                 + 0.822481 w_k1_AXP_s112
                                 + 0.635988 w_k1_DIS_s112
                                 + 0.681281 w_k1_FDX_s112
                                 + 0.891195 w_k1_IBM_s112
                                 + 0.855512 w_k1_INTC_s112
                                 + 0.823546 w_k1_KO_s112
                                 + 0.804292 w_k1_MAT_s112
                                 + 0.80094 w_k1_MSFT_s112
                                 + 0.882605 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.60403 w_k1_ADM_s112
                                 + 0.882695 w_k1_AXP_s112
                                 + 0.818509 w_k1_DIS_s112
                                 + 0.78739 w_k1_FDX_s112
                                 + 1.29638 w_k1_IBM_s112
                                 + 0.965364 w_k1_INTC_s112
                                 + 0.986494 w_k1_KO_s112
                                 + 1.26109 w_k1_MAT_s112
                                 + 0.793929 w_k1_MSFT_s112
                                 + 0.925586 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.07837 w_k1_ADM_s112
                                 + 1.17257 w_k1_AXP_s112
                                 + 1.05073 w_k1_DIS_s112
                                 + 1.05874 w_k1_FDX_s112
                                 + 1.15187 w_k1_IBM_s112
                                 + 0.846333 w_k1_INTC_s112
                                 + 1.03077 w_k1_KO_s112 + 1.27748 w_k1_MAT_s112
                                 + 1.08069 w_k1_MSFT_s112
                                 + 1.13532 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.951773 w_k1_ADM_s112
                                 + 1.04921 w_k1_AXP_s112
                                 + 1.26876 w_k1_DIS_s112
                                 + 1.02404 w_k1_FDX_s112
                                 + 1.04447 w_k1_IBM_s112
                                 + 0.950401 w_k1_INTC_s112
                                 + 1.02138 w_k1_KO_s112
                                 + 0.908568 w_k1_MAT_s112
                                 + 1.09454 w_k1_MSFT_s112
                                 + 1.09422 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.73117 w_k1_ADM_s112
                                 + 0.749223 w_k1_AXP_s112
                                 + 0.693417 w_k1_DIS_s112
                                 + 0.793782 w_k1_FDX_s112
                                 + 0.684756 w_k1_IBM_s112
                                 + 0.649481 w_k1_INTC_s112
                                 + 0.758735 w_k1_KO_s112
                                 + 0.728095 w_k1_MAT_s112
                                 + 0.798825 w_k1_MSFT_s112
                                 + 0.882997 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.06442 w_k1_ADM_s112
                                 + 0.91269 w_k1_AXP_s112
                                 + 1.13552 w_k1_DIS_s112
                                 + 1.11105 w_k1_FDX_s112
                                 + 0.847564 w_k1_IBM_s112
                                 + 0.912245 w_k1_INTC_s112
                                 + 1.1738 w_k1_KO_s112 + 0.958043 w_k1_MAT_s112
                                 + 1.12969 w_k1_MSFT_s112
                                 + 1.01227 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.881266 w_k1_ADM_s113
                                 + 0.7612 w_k1_AXP_s113 + 0.75138 w_k1_DIS_s113
                                 + 0.885026 w_k1_FDX_s113
                                 + 0.689662 w_k1_IBM_s113
                                 + 0.975655 w_k1_INTC_s113
                                 + 0.757416 w_k1_KO_s113
                                 + 0.531572 w_k1_MAT_s113
                                 + 0.865075 w_k1_MSFT_s113
                                 + 0.895819 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.887876 w_k1_ADM_s113
                                 + 1.00345 w_k1_AXP_s113
                                 + 1.12321 w_k1_DIS_s113
                                 + 0.954741 w_k1_FDX_s113
                                 + 1.29693 w_k1_IBM_s113
                                 + 1.04053 w_k1_INTC_s113
                                 + 1.16414 w_k1_KO_s113 + 1.04389 w_k1_MAT_s113
                                 + 1.08172 w_k1_MSFT_s113 + 1.0656 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.869634 w_k1_ADM_s113
                                 + 1.39252 w_k1_AXP_s113
                                 + 1.11587 w_k1_DIS_s113 + 1.0492 w_k1_FDX_s113
                                 + 1.00459 w_k1_IBM_s113
                                 + 1.28882 w_k1_INTC_s113
                                 + 1.05113 w_k1_KO_s113
                                 + 0.961309 w_k1_MAT_s113
                                 + 1.14726 w_k1_MSFT_s113
                                 + 1.05089 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.883155 w_k1_ADM_s113
                                 + 0.817617 w_k1_AXP_s113
                                 + 1.0968 w_k1_DIS_s113 + 1.04773 w_k1_FDX_s113
                                 + 0.822648 w_k1_IBM_s113
                                 + 1.32486 w_k1_INTC_s113
                                 + 1.14017 w_k1_KO_s113
                                 + 0.737222 w_k1_MAT_s113
                                 + 1.15846 w_k1_MSFT_s113
                                 + 1.08189 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.894516 w_k1_ADM_s113
                                 + 1.30462 w_k1_AXP_s113
                                 + 1.48909 w_k1_DIS_s113
                                 + 1.15546 w_k1_FDX_s113
                                 + 1.60755 w_k1_IBM_s113
                                 + 1.23562 w_k1_INTC_s113
                                 + 1.49163 w_k1_KO_s113 + 1.83463 w_k1_MAT_s113
                                 + 1.14891 w_k1_MSFT_s113
                                 + 1.18283 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.739024 w_k1_ADM_s113
                                 + 0.612771 w_k1_AXP_s113
                                 + 0.656088 w_k1_DIS_s113
                                 + 0.640401 w_k1_FDX_s113
                                 + 0.641832 w_k1_IBM_s113
                                 + 0.881416 w_k1_INTC_s113
                                 + 0.745372 w_k1_KO_s113
                                 + 0.716922 w_k1_MAT_s113
                                 + 0.862041 w_k1_MSFT_s113
                                 + 0.878163 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.10207 w_k1_ADM_s113
                                 + 1.37658 w_k1_AXP_s113
                                 + 1.72719 w_k1_DIS_s113
                                 + 1.42471 w_k1_FDX_s113 + 1.2289 w_k1_IBM_s113
                                 + 1.47519 w_k1_INTC_s113
                                 + 1.53352 w_k1_KO_s113 + 1.08713 w_k1_MAT_s113
                                 + 1.43605 w_k1_MSFT_s113
                                 + 1.51035 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.888887 w_k1_ADM_s113
                                 + 0.901717 w_k1_AXP_s113
                                 + 0.860523 w_k1_DIS_s113
                                 + 0.833635 w_k1_FDX_s113
                                 + 0.986449 w_k1_IBM_s113
                                 + 0.951936 w_k1_INTC_s113
                                 + 1.08915 w_k1_KO_s113
                                 + 0.984509 w_k1_MAT_s113
                                 + 0.954429 w_k1_MSFT_s113
                                 + 1.06353 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.19449 w_k1_ADM_s114
                                 + 0.871279 w_k1_AXP_s114
                                 + 1.03006 w_k1_DIS_s114
                                 + 0.936447 w_k1_FDX_s114
                                 + 0.950407 w_k1_IBM_s114
                                 + 0.87682 w_k1_INTC_s114
                                 + 1.02887 w_k1_KO_s114
                                 + 0.824808 w_k1_MAT_s114
                                 + 1.00191 w_k1_MSFT_s114
                                 + 0.956646 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.810981 w_k1_ADM_s114
                                 + 0.749691 w_k1_AXP_s114
                                 + 0.666814 w_k1_DIS_s114
                                 + 0.860928 w_k1_FDX_s114
                                 + 0.646912 w_k1_IBM_s114
                                 + 0.872511 w_k1_INTC_s114
                                 + 0.658111 w_k1_KO_s114
                                 + 0.582505 w_k1_MAT_s114
                                 + 0.866785 w_k1_MSFT_s114
                                 + 0.774531 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.31198 w_k1_ADM_s114
                                 + 1.14384 w_k1_AXP_s114
                                 + 1.12536 w_k1_DIS_s114 + 1.1314 w_k1_FDX_s114
                                 + 0.758727 w_k1_IBM_s114
                                 + 1.00237 w_k1_INTC_s114
                                 + 0.987054 w_k1_KO_s114
                                 + 1.01111 w_k1_MAT_s114
                                 + 1.16678 w_k1_MSFT_s114
                                 + 1.08278 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.71341 w_k1_ADM_s114
                                 + 1.20795 w_k1_AXP_s114
                                 + 1.70155 w_k1_DIS_s114
                                 + 1.22245 w_k1_FDX_s114
                                 + 1.29037 w_k1_IBM_s114
                                 + 1.0807 w_k1_INTC_s114 + 1.33765 w_k1_KO_s114
                                 + 0.858737 w_k1_MAT_s114
                                 + 1.17515 w_k1_MSFT_s114
                                 + 1.15695 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.09298 w_k1_ADM_s114
                                 + 1.1198 w_k1_AXP_s114 + 1.29393 w_k1_DIS_s114
                                 + 1.29861 w_k1_FDX_s114
                                 + 0.84473 w_k1_IBM_s114
                                 + 1.12868 w_k1_INTC_s114
                                 + 0.957845 w_k1_KO_s114
                                 + 0.80761 w_k1_MAT_s114
                                 + 1.19793 w_k1_MSFT_s114
                                 + 1.06206 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.838598 w_k1_ADM_s114
                                 + 0.842812 w_k1_AXP_s114
                                 + 0.669702 w_k1_DIS_s114
                                 + 0.790859 w_k1_FDX_s114
                                 + 1.12402 w_k1_IBM_s114
                                 + 0.714523 w_k1_INTC_s114
                                 + 0.832666 w_k1_KO_s114
                                 + 0.942421 w_k1_MAT_s114
                                 + 0.775346 w_k1_MSFT_s114
                                 + 0.873451 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.26173 w_k1_ADM_s114
                                 + 1.42204 w_k1_AXP_s114 + 1.1213 w_k1_DIS_s114
                                 + 1.11695 w_k1_FDX_s114
                                 + 1.25042 w_k1_IBM_s114
                                 + 1.03244 w_k1_INTC_s114
                                 + 1.05145 w_k1_KO_s114 + 1.23883 w_k1_MAT_s114
                                 + 0.912207 w_k1_MSFT_s114
                                 + 0.956595 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.67419 w_k1_ADM_s114
                                 + 1.78984 w_k1_AXP_s114 + 1.4978 w_k1_DIS_s114
                                 + 1.48375 w_k1_FDX_s114
                                 + 1.17101 w_k1_IBM_s114
                                 + 1.17845 w_k1_INTC_s114
                                 + 1.19641 w_k1_KO_s114 + 1.34783 w_k1_MAT_s114
                                 + 1.19329 w_k1_MSFT_s114
                                 + 1.11396 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.718719 w_k1_ADM_s115
                                 + 0.898631 w_k1_AXP_s115
                                 + 0.98887 w_k1_DIS_s115
                                 + 0.867771 w_k1_FDX_s115
                                 + 1.02999 w_k1_IBM_s115
                                 + 1.00962 w_k1_INTC_s115
                                 + 1.01385 w_k1_KO_s115 + 1.08361 w_k1_MAT_s115
                                 + 0.932543 w_k1_MSFT_s115
                                 + 0.971352 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.934898 w_k1_ADM_s115
                                 + 1.18978 w_k1_AXP_s115
                                 + 1.03001 w_k1_DIS_s115 + 1.109 w_k1_FDX_s115
                                 + 0.81571 w_k1_IBM_s115
                                 + 1.20541 w_k1_INTC_s115
                                 + 0.906327 w_k1_KO_s115
                                 + 0.94078 w_k1_MAT_s115
                                 + 1.01494 w_k1_MSFT_s115
                                 + 0.961091 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.17471 w_k1_ADM_s115
                                 + 1.20799 w_k1_AXP_s115
                                 + 1.28709 w_k1_DIS_s115
                                 + 1.07386 w_k1_FDX_s115
                                 + 1.45757 w_k1_IBM_s115
                                 + 0.961652 w_k1_INTC_s115
                                 + 1.29871 w_k1_KO_s115 + 1.63169 w_k1_MAT_s115
                                 + 1.20646 w_k1_MSFT_s115
                                 + 1.05631 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.3154 w_k1_ADM_s115
                                 + 1.80371 w_k1_AXP_s115
                                 + 1.82378 w_k1_DIS_s115
                                 + 1.59654 w_k1_FDX_s115
                                 + 1.24914 w_k1_IBM_s115
                                 + 1.25969 w_k1_INTC_s115
                                 + 1.47981 w_k1_KO_s115 + 1.5651 w_k1_MAT_s115
                                 + 1.42719 w_k1_MSFT_s115
                                 + 1.22188 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.06834 w_k1_ADM_s115
                                 + 1.2287 w_k1_AXP_s115 + 1.27015 w_k1_DIS_s115
                                 + 1.18418 w_k1_FDX_s115
                                 + 1.13975 w_k1_IBM_s115
                                 + 1.43424 w_k1_INTC_s115
                                 + 1.24482 w_k1_KO_s115 + 1.08061 w_k1_MAT_s115
                                 + 1.26678 w_k1_MSFT_s115
                                 + 1.10493 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.06019 w_k1_ADM_s115
                                 + 0.919937 w_k1_AXP_s115
                                 + 0.85524 w_k1_DIS_s115
                                 + 0.81699 w_k1_FDX_s115
                                 + 1.32652 w_k1_IBM_s115
                                 + 0.927081 w_k1_INTC_s115
                                 + 0.998736 w_k1_KO_s115
                                 + 1.21808 w_k1_MAT_s115
                                 + 0.909561 w_k1_MSFT_s115
                                 + 0.939761 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.950575 w_k1_ADM_s115
                                 + 0.909353 w_k1_AXP_s115
                                 + 1.17997 w_k1_DIS_s115
                                 + 1.09244 w_k1_FDX_s115
                                 + 0.719547 w_k1_IBM_s115
                                 + 1.05939 w_k1_INTC_s115
                                 + 1.02765 w_k1_KO_s115
                                 + 0.845532 w_k1_MAT_s115
                                 + 1.24101 w_k1_MSFT_s115
                                 + 1.00894 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.826847 w_k1_ADM_s115
                                 + 0.671802 w_k1_AXP_s115
                                 + 0.727337 w_k1_DIS_s115
                                 + 0.824998 w_k1_FDX_s115
                                 + 0.712138 w_k1_IBM_s115
                                 + 0.742579 w_k1_INTC_s115
                                 + 0.740457 w_k1_KO_s115
                                 + 0.862105 w_k1_MAT_s115
                                 + 0.864952 w_k1_MSFT_s115
                                 + 0.875857 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.02132 w_k1_ADM_s116
                                 + 0.910974 w_k1_AXP_s116
                                 + 0.727166 w_k1_DIS_s116
                                 + 0.833323 w_k1_FDX_s116
                                 + 0.90102 w_k1_IBM_s116
                                 + 0.874492 w_k1_INTC_s116
                                 + 0.771989 w_k1_KO_s116
                                 + 0.686979 w_k1_MAT_s116
                                 + 0.810515 w_k1_MSFT_s116
                                 + 0.878645 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.08699 w_k1_ADM_s116
                                 + 0.970454 w_k1_AXP_s116
                                 + 1.02974 w_k1_DIS_s116
                                 + 1.01383 w_k1_FDX_s116
                                 + 0.750918 w_k1_IBM_s116
                                 + 0.977323 w_k1_INTC_s116
                                 + 0.815941 w_k1_KO_s116
                                 + 0.704482 w_k1_MAT_s116
                                 + 0.967159 w_k1_MSFT_s116
                                 + 0.916339 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.7559 w_k1_ADM_s116
                                 + 1.63643 w_k1_AXP_s116 + 2.1781 w_k1_DIS_s116
                                 + 1.37488 w_k1_FDX_s116
                                 + 1.81777 w_k1_IBM_s116
                                 + 1.46525 w_k1_INTC_s116
                                 + 1.82847 w_k1_KO_s116 + 1.19381 w_k1_MAT_s116
                                 + 1.38831 w_k1_MSFT_s116
                                 + 1.31717 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.791066 w_k1_ADM_s116
                                 + 0.685458 w_k1_AXP_s116
                                 + 0.648206 w_k1_DIS_s116
                                 + 0.832637 w_k1_FDX_s116
                                 + 0.602309 w_k1_IBM_s116
                                 + 0.808038 w_k1_INTC_s116
                                 + 0.663196 w_k1_KO_s116
                                 + 0.42993 w_k1_MAT_s116
                                 + 0.845459 w_k1_MSFT_s116
                                 + 0.818174 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.868591 w_k1_ADM_s116
                                 + 1.04101 w_k1_AXP_s116
                                 + 1.22953 w_k1_DIS_s116
                                 + 1.05368 w_k1_FDX_s116
                                 + 1.49043 w_k1_IBM_s116 + 1.163 w_k1_INTC_s116
                                 + 1.30473 w_k1_KO_s116 + 1.67967 w_k1_MAT_s116
                                 + 1.16831 w_k1_MSFT_s116
                                 + 1.15349 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.25578 w_k1_ADM_s116
                                 + 1.23611 w_k1_AXP_s116
                                 + 1.15682 w_k1_DIS_s116
                                 + 1.19352 w_k1_FDX_s116
                                 + 1.13273 w_k1_IBM_s116
                                 + 0.873832 w_k1_INTC_s116
                                 + 0.978931 w_k1_KO_s116
                                 + 1.41045 w_k1_MAT_s116
                                 + 1.09815 w_k1_MSFT_s116
                                 + 1.08911 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.696442 w_k1_ADM_s116
                                 + 0.712236 w_k1_AXP_s116
                                 + 0.650802 w_k1_DIS_s116
                                 + 0.766461 w_k1_FDX_s116
                                 + 0.861765 w_k1_IBM_s116
                                 + 0.730532 w_k1_INTC_s116
                                 + 0.657057 w_k1_KO_s116
                                 + 0.795002 w_k1_MAT_s116
                                 + 0.744555 w_k1_MSFT_s116
                                 + 0.790894 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.994369 w_k1_ADM_s116
                                 + 0.797145 w_k1_AXP_s116
                                 + 0.785446 w_k1_DIS_s116
                                 + 0.794666 w_k1_FDX_s116
                                 + 1.16919 w_k1_IBM_s116
                                 + 0.655437 w_k1_INTC_s116
                                 + 0.938805 w_k1_KO_s116
                                 + 1.38759 w_k1_MAT_s116
                                 + 0.843227 w_k1_MSFT_s116
                                 + 0.876518 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.19069 w_k1_ADM_s117
                                 + 1.22518 w_k1_AXP_s117
                                 + 1.45768 w_k1_DIS_s117
                                 + 1.13791 w_k1_FDX_s117
                                 + 1.41537 w_k1_IBM_s117
                                 + 0.915545 w_k1_INTC_s117
                                 + 1.28509 w_k1_KO_s117 + 1.29286 w_k1_MAT_s117
                                 + 1.03197 w_k1_MSFT_s117
                                 + 1.00934 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.97195 w_k1_ADM_s117
                                 + 0.773364 w_k1_AXP_s117
                                 + 0.778001 w_k1_DIS_s117
                                 + 0.780891 w_k1_FDX_s117
                                 + 1.17361 w_k1_IBM_s117
                                 + 0.772408 w_k1_INTC_s117
                                 + 0.978329 w_k1_KO_s117
                                 + 0.95717 w_k1_MAT_s117
                                 + 0.918806 w_k1_MSFT_s117
                                 + 0.872343 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.810761 w_k1_ADM_s117
                                 + 1.15609 w_k1_AXP_s117
                                 + 0.896539 w_k1_DIS_s117
                                 + 1.05387 w_k1_FDX_s117
                                 + 0.979701 w_k1_IBM_s117
                                 + 1.00295 w_k1_INTC_s117
                                 + 0.904253 w_k1_KO_s117
                                 + 1.27338 w_k1_MAT_s117
                                 + 0.907729 w_k1_MSFT_s117
                                 + 0.945105 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.684912 w_k1_ADM_s117
                                 + 0.704326 w_k1_AXP_s117
                                 + 0.868638 w_k1_DIS_s117
                                 + 0.866763 w_k1_FDX_s117
                                 + 0.907659 w_k1_IBM_s117
                                 + 0.871962 w_k1_INTC_s117
                                 + 0.897364 w_k1_KO_s117
                                 + 0.609614 w_k1_MAT_s117
                                 + 0.870958 w_k1_MSFT_s117
                                 + 0.878817 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.689265 w_k1_ADM_s117
                                 + 0.74654 w_k1_AXP_s117
                                 + 0.753902 w_k1_DIS_s117
                                 + 0.861224 w_k1_FDX_s117
                                 + 0.739935 w_k1_IBM_s117
                                 + 0.872951 w_k1_INTC_s117
                                 + 0.733286 w_k1_KO_s117
                                 + 0.653141 w_k1_MAT_s117
                                 + 0.765229 w_k1_MSFT_s117
                                 + 0.7419 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.670315 w_k1_ADM_s117
                                 + 0.802238 w_k1_AXP_s117
                                 + 1.17769 w_k1_DIS_s117
                                 + 1.10588 w_k1_FDX_s117
                                 + 0.920641 w_k1_IBM_s117
                                 + 1.09514 w_k1_INTC_s117
                                 + 0.930465 w_k1_KO_s117
                                 + 0.95738 w_k1_MAT_s117
                                 + 1.12886 w_k1_MSFT_s117
                                 + 0.977365 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.01574 w_k1_ADM_s117
                                 + 1.41754 w_k1_AXP_s117
                                 + 1.39188 w_k1_DIS_s117
                                 + 1.31769 w_k1_FDX_s117
                                 + 0.991391 w_k1_IBM_s117
                                 + 1.34132 w_k1_INTC_s117
                                 + 1.17432 w_k1_KO_s117 + 1.10737 w_k1_MAT_s117
                                 + 1.23568 w_k1_MSFT_s117
                                 + 1.10798 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.06298 w_k1_ADM_s117
                                 + 0.963439 w_k1_AXP_s117
                                 + 0.875485 w_k1_DIS_s117
                                 + 0.901193 w_k1_FDX_s117
                                 + 1.60808 w_k1_IBM_s117
                                 + 0.988898 w_k1_INTC_s117
                                 + 1.20142 w_k1_KO_s117 + 1.43845 w_k1_MAT_s117
                                 + 0.936819 w_k1_MSFT_s117
                                 + 1.03699 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0113487278681421 w_k1_ADM_s000
                                 + 0.0234481815340402 w_k1_ADM_s001
                                 + 0.014505145868017 w_k1_ADM_s002
                                 + 0.0195812991469507 w_k1_ADM_s003
                                 + 0.0158178100825695 w_k1_ADM_s004
                                 + 0.025868497851881 w_k1_ADM_s005
                                 + 0.0155087467348535 w_k1_ADM_s006
                                 + 0.0226305584189441 w_k1_ADM_s007
                                 + 0.0516609570913672 w_k1_ADM_s010
                                 + 0.0254302597493 w_k1_ADM_s011
                                 + 0.055092470482923 w_k1_ADM_s012
                                 + 0.0237048865905668 w_k1_ADM_s013
                                 + 0.0183519041153221 w_k1_ADM_s014
                                 + 0.0199797672726443 w_k1_ADM_s015
                                 + 0.0458305711315069 w_k1_ADM_s016
                                 + 0.0492316263420339 w_k1_ADM_s017
                                 + 0.0567132667205292 w_k1_ADM_s100
                                 + 0.0366999780767278 w_k1_ADM_s101
                                 + 0.0254231449100876 w_k1_ADM_s102
                                 + 0.067595716086504 w_k1_ADM_s103
                                 + 0.0527639930074868 w_k1_ADM_s104
                                 + 0.028564807093187 w_k1_ADM_s105
                                 + 0.0829580987250954 w_k1_ADM_s106
                                 + 0.0739315435983375 w_k1_ADM_s107
                                 + 0.0293899819304866 w_k1_ADM_s110
                                 + 0.0103130098811543 w_k1_ADM_s111
                                 + 0.0174870434923353 w_k1_ADM_s112
                                 + 0.0212059875454572 w_k1_ADM_s113
                                 + 0.0233883835053428 w_k1_ADM_s114
                                 + 0.0241318301653866 w_k1_ADM_s115
                                 + 0.00925045091692164 w_k1_ADM_s116
                                 + 0.00711817750729935 w_k1_ADM_s117
                                 + 0.0102027407862075 w_k1_AXP_s000
                                 + 0.0236216774010313 w_k1_AXP_s001
                                 + 0.0143407342332683 w_k1_AXP_s002
                                 + 0.0191755538193517 w_k1_AXP_s003
                                 + 0.0134233730378701 w_k1_AXP_s004
                                 + 0.0259786884100487 w_k1_AXP_s005
                                 + 0.0145328543010153 w_k1_AXP_s006
                                 + 0.0250476838832187 w_k1_AXP_s007
                                 + 0.0462613363888805 w_k1_AXP_s010
                                 + 0.0245997362401014 w_k1_AXP_s011
                                 + 0.0596582114628858 w_k1_AXP_s012
                                 + 0.0234420288161381 w_k1_AXP_s013
                                 + 0.0172706059749177 w_k1_AXP_s014
                                 + 0.0193905785558857 w_k1_AXP_s015
                                 + 0.0479743167267487 w_k1_AXP_s016
                                 + 0.048299584872404 w_k1_AXP_s017
                                 + 0.0573441095151291 w_k1_AXP_s100
                                 + 0.0372995352723905 w_k1_AXP_s101
                                 + 0.0263077407395275 w_k1_AXP_s102
                                 + 0.0721074569244879 w_k1_AXP_s103
                                 + 0.0586677562389271 w_k1_AXP_s104
                                 + 0.0272191278946194 w_k1_AXP_s105
                                 + 0.0846087594851571 w_k1_AXP_s106
                                 + 0.0707646076774327 w_k1_AXP_s107
                                 + 0.0342030654254268 w_k1_AXP_s110
                                 + 0.0110472251271837 w_k1_AXP_s111
                                 + 0.0173319364041094 w_k1_AXP_s112
                                 + 0.0238347639372645 w_k1_AXP_s113
                                 + 0.0216703934299998 w_k1_AXP_s114
                                 + 0.0266181722668208 w_k1_AXP_s115
                                 + 0.00881407668904695 w_k1_AXP_s116
                                 + 0.00767973815303437 w_k1_AXP_s117
                                 + 0.0108034284329409 w_k1_DIS_s000
                                 + 0.0238815800535751 w_k1_DIS_s001
                                 + 0.0155801899580575 w_k1_DIS_s002
                                 + 0.0200139801453395 w_k1_DIS_s003
                                 + 0.0144467499599238 w_k1_DIS_s004
                                 + 0.0234362260985164 w_k1_DIS_s005
                                 + 0.016398810032345 w_k1_DIS_s006
                                 + 0.0259165505882589 w_k1_DIS_s007
                                 + 0.0517673338388635 w_k1_DIS_s010
                                 + 0.0261913707462253 w_k1_DIS_s011
                                 + 0.0577049848808386 w_k1_DIS_s012
                                 + 0.0253631291499473 w_k1_DIS_s013
                                 + 0.0169535159589351 w_k1_DIS_s014
                                 + 0.0185577586264668 w_k1_DIS_s015
                                 + 0.0498837063215195 w_k1_DIS_s016
                                 + 0.0461973474664537 w_k1_DIS_s017
                                 + 0.061502898289447 w_k1_DIS_s100
                                 + 0.0352103485838562 w_k1_DIS_s101
                                 + 0.0253180933117013 w_k1_DIS_s102
                                 + 0.0699305020797601 w_k1_DIS_s103
                                 + 0.0602203024152059 w_k1_DIS_s104
                                 + 0.0269375769901494 w_k1_DIS_s105
                                 + 0.0851457911453087 w_k1_DIS_s106
                                 + 0.0740551367183553 w_k1_DIS_s107
                                 + 0.0343039464002653 w_k1_DIS_s110
                                 + 0.0113111030951493 w_k1_DIS_s111
                                 + 0.0186218752515439 w_k1_DIS_s112
                                 + 0.0256744077758943 w_k1_DIS_s113
                                 + 0.0208132121802452 w_k1_DIS_s114
                                 + 0.0274503099976902 w_k1_DIS_s115
                                 + 0.00899505173216309 w_k1_DIS_s116
                                 + 0.00798879150421398 w_k1_DIS_s117
                                 + 0.0100687278287596 w_k1_FDX_s000
                                 + 0.0231332888710718 w_k1_FDX_s001
                                 + 0.0143995127049462 w_k1_FDX_s002
                                 + 0.0196346321471267 w_k1_FDX_s003
                                 + 0.0143128741506184 w_k1_FDX_s004
                                 + 0.0257891891324975 w_k1_FDX_s005
                                 + 0.0168032470144511 w_k1_FDX_s006
                                 + 0.027055634822516 w_k1_FDX_s007
                                 + 0.0499506703430082 w_k1_FDX_s010
                                 + 0.0249760556983937 w_k1_FDX_s011
                                 + 0.0594379322386765 w_k1_FDX_s012
                                 + 0.0242443161093411 w_k1_FDX_s013
                                 + 0.0175004749255765 w_k1_FDX_s014
                                 + 0.0188271991677496 w_k1_FDX_s015
                                 + 0.0474534741324119 w_k1_FDX_s016
                                 + 0.0455953858966573 w_k1_FDX_s017
                                 + 0.0599150418676142 w_k1_FDX_s100
                                 + 0.0352893701018082 w_k1_FDX_s101
                                 + 0.0235642480044542 w_k1_FDX_s102
                                 + 0.0671804620610013 w_k1_FDX_s103
                                 + 0.059260662956977 w_k1_FDX_s104
                                 + 0.0273530980520739 w_k1_FDX_s105
                                 + 0.0849073621531616 w_k1_FDX_s106
                                 + 0.0704495608576332 w_k1_FDX_s107
                                 + 0.0339216088109716 w_k1_FDX_s110
                                 + 0.010554014230256 w_k1_FDX_s111
                                 + 0.0172776247029937 w_k1_FDX_s112
                                 + 0.0235532366857504 w_k1_FDX_s113
                                 + 0.0205952218338644 w_k1_FDX_s114
                                 + 0.0260434860388948 w_k1_FDX_s115
                                 + 0.00873622379583199 w_k1_FDX_s116
                                 + 0.00795409031854411 w_k1_FDX_s117
                                 + 0.00994647463776283 w_k1_IBM_s000
                                 + 0.0238424408163807 w_k1_IBM_s001
                                 + 0.0137576311057714 w_k1_IBM_s002
                                 + 0.0194091894429104 w_k1_IBM_s003
                                 + 0.0140778520031266 w_k1_IBM_s004
                                 + 0.0245804885866659 w_k1_IBM_s005
                                 + 0.0173979486616001 w_k1_IBM_s006
                                 + 0.0246165307018153 w_k1_IBM_s007
                                 + 0.0491406006392408 w_k1_IBM_s010
                                 + 0.0236851403936405 w_k1_IBM_s011
                                 + 0.0613382424085644 w_k1_IBM_s012
                                 + 0.0263048499440501 w_k1_IBM_s013
                                 + 0.0173599815023899 w_k1_IBM_s014
                                 + 0.0189310535325162 w_k1_IBM_s015
                                 + 0.0485457753356115 w_k1_IBM_s016
                                 + 0.0468966788189922 w_k1_IBM_s017
                                 + 0.0625211858620059 w_k1_IBM_s100
                                 + 0.0348167065507873 w_k1_IBM_s101
                                 + 0.023628209765575 w_k1_IBM_s102
                                 + 0.0738588730780235 w_k1_IBM_s103
                                 + 0.0564793644307004 w_k1_IBM_s104
                                 + 0.0278638779433749 w_k1_IBM_s105
                                 + 0.0878233229805488 w_k1_IBM_s106
                                 + 0.0715709920516938 w_k1_IBM_s107
                                 + 0.0362190802703069 w_k1_IBM_s110
                                 + 0.0106591418954396 w_k1_IBM_s111
                                 + 0.0183162549421284 w_k1_IBM_s112
                                 + 0.0247535147702502 w_k1_IBM_s113
                                 + 0.0204718193417028 w_k1_IBM_s114
                                 + 0.0256940914751338 w_k1_IBM_s115
                                 + 0.00890620533539622 w_k1_IBM_s116
                                 + 0.0085378282085987 w_k1_IBM_s117
                                 + 0.00888691380665783 w_k1_INTC_s000
                                 + 0.0240180800638786 w_k1_INTC_s001
                                 + 0.0147038889650123 w_k1_INTC_s002
                                 + 0.0202018858490088 w_k1_INTC_s003
                                 + 0.0130748779061128 w_k1_INTC_s004
                                 + 0.0218781638864543 w_k1_INTC_s005
                                 + 0.0183791987403006 w_k1_INTC_s006
                                 + 0.0251198055684604 w_k1_INTC_s007
                                 + 0.0435484935992056 w_k1_INTC_s010
                                 + 0.0235159887904828 w_k1_INTC_s011
                                 + 0.0564779750448062 w_k1_INTC_s012
                                 + 0.0244120345263249 w_k1_INTC_s013
                                 + 0.0180999056748049 w_k1_INTC_s014
                                 + 0.0204639820376875 w_k1_INTC_s015
                                 + 0.0540913390957637 w_k1_INTC_s016
                                 + 0.0537513563371034 w_k1_INTC_s017
                                 + 0.0634528964554274 w_k1_INTC_s100
                                 + 0.0365246680250468 w_k1_INTC_s101
                                 + 0.0225746773321902 w_k1_INTC_s102
                                 + 0.069158666421258 w_k1_INTC_s103
                                 + 0.0620203873475006 w_k1_INTC_s104
                                 + 0.0266464901913376 w_k1_INTC_s105
                                 + 0.080052447318415 w_k1_INTC_s106
                                 + 0.0659641603577324 w_k1_INTC_s107
                                 + 0.0368517341307628 w_k1_INTC_s110
                                 + 0.0110059632476882 w_k1_INTC_s111
                                 + 0.0155972411088289 w_k1_INTC_s112
                                 + 0.0269238375195304 w_k1_INTC_s113
                                 + 0.0186712663720089 w_k1_INTC_s114
                                 + 0.0266552520183429 w_k1_INTC_s115
                                 + 0.00802742193629151 w_k1_INTC_s116
                                 + 0.00779158785689146 w_k1_INTC_s117
                                 + 0.011585646853333 w_k1_KO_s000
                                 + 0.0232669773189622 w_k1_KO_s001
                                 + 0.0147687111448884 w_k1_KO_s002
                                 + 0.020181904064385 w_k1_KO_s003
                                 + 0.0162318969517877 w_k1_KO_s004
                                 + 0.0258634476921244 w_k1_KO_s005
                                 + 0.0184919229232755 w_k1_KO_s006
                                 + 0.0260648269404315 w_k1_KO_s007
                                 + 0.0472980368771833 w_k1_KO_s010
                                 + 0.0225081504565706 w_k1_KO_s011
                                 + 0.0586556572711024 w_k1_KO_s012
                                 + 0.0262676790565615 w_k1_KO_s013
                                 + 0.0161475887739975 w_k1_KO_s014
                                 + 0.0202073452919202 w_k1_KO_s015
                                 + 0.049752528205907 w_k1_KO_s016
                                 + 0.0485395131757061 w_k1_KO_s017
                                 + 0.0645269504555664 w_k1_KO_s100
                                 + 0.0351460645821603 w_k1_KO_s101
                                 + 0.0243865653220354 w_k1_KO_s102
                                 + 0.0724810380406964 w_k1_KO_s103
                                 + 0.0542326956060376 w_k1_KO_s104
                                 + 0.026809594994262 w_k1_KO_s105
                                 + 0.0825579404929685 w_k1_KO_s106
                                 + 0.0740464936389249 w_k1_KO_s107
                                 + 0.0326659112054255 w_k1_KO_s110
                                 + 0.0100031064443829 w_k1_KO_s111
                                 + 0.0185624424761111 w_k1_KO_s112
                                 + 0.0266364288596356 w_k1_KO_s113
                                 + 0.0195720313055384 w_k1_KO_s114
                                 + 0.0263923047697529 w_k1_KO_s115
                                 + 0.00818476213423656 w_k1_KO_s116
                                 + 0.00794343273760253 w_k1_KO_s117
                                 + 0.010186164813667 w_k1_MAT_s000
                                 + 0.0230018460565464 w_k1_MAT_s001
                                 + 0.0149166444624062 w_k1_MAT_s002
                                 + 0.0202142283387693 w_k1_MAT_s003
                                 + 0.0131221415125342 w_k1_MAT_s004
                                 + 0.0287849036746339 w_k1_MAT_s005
                                 + 0.0169543322006561 w_k1_MAT_s006
                                 + 0.0267198473898057 w_k1_MAT_s007
                                 + 0.0440772879543168 w_k1_MAT_s010
                                 + 0.0265674854727913 w_k1_MAT_s011
                                 + 0.0774164353809368 w_k1_MAT_s012
                                 + 0.0268740755763231 w_k1_MAT_s013
                                 + 0.0138673095548068 w_k1_MAT_s014
                                 + 0.0172793072392571 w_k1_MAT_s015
                                 + 0.0461854864053327 w_k1_MAT_s016
                                 + 0.0555098084196023 w_k1_MAT_s017
                                 + 0.0719841528631588 w_k1_MAT_s100
                                 + 0.0376667618256799 w_k1_MAT_s101
                                 + 0.0278868984926409 w_k1_MAT_s102
                                 + 0.0679344680352285 w_k1_MAT_s103
                                 + 0.0582692285251284 w_k1_MAT_s104
                                 + 0.026806875733677 w_k1_MAT_s105
                                 + 0.0690400401363945 w_k1_MAT_s106
                                 + 0.0658723328625516 w_k1_MAT_s107
                                 + 0.0359906184979466 w_k1_MAT_s110
                                 + 0.00886789966045463 w_k1_MAT_s111
                                 + 0.0186402224304564 w_k1_MAT_s112
                                 + 0.0235257894005273 w_k1_MAT_s113
                                 + 0.0190325961073711 w_k1_MAT_s114
                                 + 0.028085048079787 w_k1_MAT_s115
                                 + 0.00914955418059759 w_k1_MAT_s116
                                 + 0.00802397361846009 w_k1_MAT_s117
                                 + 0.0103518527194511 w_k1_MSFT_s000
                                 + 0.0219481526101159 w_k1_MSFT_s001
                                 + 0.0151027088762384 w_k1_MSFT_s002
                                 + 0.0209466558720032 w_k1_MSFT_s003
                                 + 0.0146074154708615 w_k1_MSFT_s004
                                 + 0.0240492284454139 w_k1_MSFT_s005
                                 + 0.0168477165855143 w_k1_MSFT_s006
                                 + 0.0256061880608611 w_k1_MSFT_s007
                                 + 0.0458268970621247 w_k1_MSFT_s010
                                 + 0.0269711613866282 w_k1_MSFT_s011
                                 + 0.0558229318931802 w_k1_MSFT_s012
                                 + 0.0247889083836554 w_k1_MSFT_s013
                                 + 0.0174781777757913 w_k1_MSFT_s014
                                 + 0.0178685651856757 w_k1_MSFT_s015
                                 + 0.0525322848595187 w_k1_MSFT_s016
                                 + 0.0441516617561544 w_k1_MSFT_s017
                                 + 0.0605064145623276 w_k1_MSFT_s100
                                 + 0.0372374392964145 w_k1_MSFT_s101
                                 + 0.0241130824710847 w_k1_MSFT_s102
                                 + 0.0683386337051553 w_k1_MSFT_s103
                                 + 0.060386501873716 w_k1_MSFT_s104
                                 + 0.0257493909886286 w_k1_MSFT_s105
                                 + 0.083585041122228 w_k1_MSFT_s106
                                 + 0.0703054103940217 w_k1_MSFT_s107
                                 + 0.0335463091725188 w_k1_MSFT_s110
                                 + 0.0108403345344859 w_k1_MSFT_s111
                                 + 0.0172444765974452 w_k1_MSFT_s112
                                 + 0.025289019621159 w_k1_MSFT_s113
                                 + 0.0194830459897356 w_k1_MSFT_s114
                                 + 0.0268463491109954 w_k1_MSFT_s115
                                 + 0.00855234820813487 w_k1_MSFT_s116
                                 + 0.00769507533254717 w_k1_MSFT_s117
                                 + 0.0100312799847181 w_k1_PG_s000
                                 + 0.0228522303459389 w_k1_PG_s001
                                 + 0.0147169895499906 w_k1_PG_s002
                                 + 0.0216998505559934 w_k1_PG_s003
                                 + 0.0145662993289002 w_k1_PG_s004
                                 + 0.0264578340137906 w_k1_PG_s005
                                 + 0.0158810460190068 w_k1_PG_s006
                                 + 0.0257767551733022 w_k1_PG_s007
                                 + 0.0491851927824152 w_k1_PG_s010
                                 + 0.0251140015760982 w_k1_PG_s011
                                 + 0.0589519755764784 w_k1_PG_s012
                                 + 0.023634349827179 w_k1_PG_s013
                                 + 0.0161203641957585 w_k1_PG_s014
                                 + 0.0190534911844152 w_k1_PG_s015
                                 + 0.0498605610560364 w_k1_PG_s016
                                 + 0.0446767408271414 w_k1_PG_s017
                                 + 0.0577246733931921 w_k1_PG_s100
                                 + 0.0356790801702094 w_k1_PG_s101
                                 + 0.0237352667072287 w_k1_PG_s102
                                 + 0.0701885653729777 w_k1_PG_s103
                                 + 0.0554154580193048 w_k1_PG_s104
                                 + 0.0263802715546473 w_k1_PG_s105
                                 + 0.0831660961853514 w_k1_PG_s106
                                 + 0.0728368564910383 w_k1_PG_s107
                                 + 0.0342335370755715 w_k1_PG_s110
                                 + 0.0103755997974538 w_k1_PG_s111
                                 + 0.0181545983139445 w_k1_PG_s112
                                 + 0.0255911565899251 w_k1_PG_s113
                                 + 0.0191758898650151 w_k1_PG_s114
                                 + 0.0252428463961138 w_k1_PG_s115
                                 + 0.0084626598159024 w_k1_PG_s116
                                 + 0.0074457336856416 w_k1_PG_s117 >= 1.17
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
