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
 w_k1_ADM_s0_return_constraint:  1.04379 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.961223 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.04183 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.860961 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.990043 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15537 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.3701 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.20117 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.918054 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.798797 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.22844 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.24791 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.904985 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.954317 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.06991 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.11583 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.794909 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.985095 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.78034 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.93411 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.16081 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.913764 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.02444 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.785497 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.26092 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.08 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.776415 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.22758 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.03089 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.05291 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.925864 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.16924 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.779762 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.774695 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.911178 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.886294 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.914826 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.38566 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.02937 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.03469 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.18867 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.755728 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.868213 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.18181 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.96206 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.14907 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.832755 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.86822 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.14335 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.35657 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.601733 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.758453 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.35698 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.34022 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.778309 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.97461 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.899747 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.10191 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.66622 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.93566 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.00393 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.712053 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.59899 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.50734 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.648271 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.80763 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.19859 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.958185 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.911901 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.89672 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.702009 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.60398 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.885933 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.36237 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.02751 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.33059 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.03358 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.08989 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.25094 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.669201 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.782073 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.25486 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.05369 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.16518 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.748707 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.631861 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.31371 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.39918 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.617695 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.941926 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.63128 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.49402 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.727461 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.03483 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.78491 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.12879 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.50601 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.21372 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.04557 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.685576 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.85772 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.16506 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.613441 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.91582 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.10155 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.16994 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.07772 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 2.0644 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.701463 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.647057 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.2782 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.19823 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.945803 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.37316 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.04825 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.976134 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.17696 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.787173 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.930043 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.17074 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.1707 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.11615 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.848658 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.776167 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.22928 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.12549 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.798272 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.870388 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.25003 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.23627 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.833551 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.00561 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.829373 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.16452 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.20115 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.00622 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.01492 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.708602 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.42351 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.04849 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.728294 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.67247 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.03072 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.06924 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.965906 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.64842 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.765866 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.729445 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.12313 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.09432 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.00066 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.30912 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.877423 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.22396 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.17525 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.930103 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.816816 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.30324 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.579036 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.04025 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.709103 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.924442 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.849447 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.56338 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.495782 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.936969 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.969384 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.34407 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.607498 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.923857 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.10055 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.782895 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.98225 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.14745 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.814227 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.991214 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.35413 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.60289 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.585185 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.04824 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.11478 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.990064 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.21753 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.57196 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.990726 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.651677 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.646213 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.37201 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.836751 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.830294 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.02079 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.01123 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.1781 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.872756 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.902762 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.07509 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.22338 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.01409 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.965139 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.729019 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.00592 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.04332 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.877079 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.934827 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.14992 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.10618 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.905044 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.91752 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.852221 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02479 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.04592 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.09846 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.976884 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.860727 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.36633 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.10333 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.769229 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.45792 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.04335 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.17167 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.09831 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.57093 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.870059 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.864348 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.24121 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.00018 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.09099 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.16218 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.96092 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.1546 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.17749 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.853574 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.793821 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.15705 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.820396 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.22544 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.766993 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.784713 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.05152 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.29159 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.579836 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.921572 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.19498 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.1978 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.727164 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.965738 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.917032 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.831041 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.37467 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.02606 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.900979 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.80901 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.36633 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.39421 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.639659 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.30525 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.12142 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.05549 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.18181 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.85853 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.927162 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.734906 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.969107 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.07162 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.874925 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.950698 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.842707 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.28822 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.24018 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.950121 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.753505 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.25143 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.445072 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.23038 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.780515 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.01416 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.24442 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.46 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.305988 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.03727 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.20704 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.27341 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.639568 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.909666 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.06344 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.956044 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.11709 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.902023 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.709518 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.696276 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.16672 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.84961 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.479555 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.59289 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.21815 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.795933 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.0585 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 2.12758 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.821739 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.701255 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.574236 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.62999 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.990114 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.855719 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.05448 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.01832 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.16131 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.7931 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.888589 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.19595 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.09322 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.13652 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.784691 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.705907 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.16511 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.02187 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.772837 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.949947 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.26411 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.17555 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.862413 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.03964 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.838575 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.0972 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.09457 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.04604 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.987011 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.782882 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.33785 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.09267 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.795464 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.52937 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.943056 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.19169 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.0262 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.53045 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.764473 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.751972 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.12613 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.07899 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.969833 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.23495 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.967983 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.03783 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.14113 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.923036 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.916489 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.17217 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.00033 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.08523 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.8304 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.818914 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.11227 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.12526 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.787163 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.950281 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.13875 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.16276 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.823161 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.998576 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.863923 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.960526 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.09688 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.940597 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.897567 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.840029 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.17355 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.15134 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.809171 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.30769 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.00755 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.07729 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.959858 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.29648 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.85769 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.744917 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.960951 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.01879 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.964606 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.07415 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.147999750182486 z_k1_s0000
                                 - 0.381999328164287 z_k1_s0001
                                 - 0.147999750182486 z_k1_s0002
                                 - 0.230998993443031 z_k1_s0003
                                 - 0.201999652793671 z_k1_s0004
                                 - 0.123999844766339 z_k1_s0005
                                 - 0.181000024116681 z_k1_s0006
                                 - 0.122999803819889 z_k1_s0007
                                 - 0.57900302907466 z_k1_s0010
                                 - 0.629003541689617 z_k1_s0011
                                 - 0.547001731998904 z_k1_s0012
                                 - 0.763002008491156 z_k1_s0013
                                 - 0.610001408891568 z_k1_s0014
                                 - 0.950000049662367 z_k1_s0015
                                 - 0.564002875290172 z_k1_s0016
                                 - 0.203000627713454 z_k1_s0017
                                 - 0.61100168021774 z_k1_s0020
                                 - 0.194000626777172 z_k1_s0021
                                 - 0.422999961992461 z_k1_s0022
                                 - 0.492003416798338 z_k1_s0023
                                 - 0.6310019460718 z_k1_s0024
                                 - 0.820003091987619 z_k1_s0025
                                 - 0.265000984956922 z_k1_s0026
                                 - 0.4680015361677 z_k1_s0027
                                 - 0.509002112377673 z_k1_s0030
                                 - 0.455000967223419 z_k1_s0031
                                 - 0.408000264812086 z_k1_s0032
                                 - 0.656003463960032 z_k1_s0033
                                 - 0.22900069199975 z_k1_s0034
                                 - 0.402001020463725 z_k1_s0035
                                 - 0.503002868029312 z_k1_s0036
                                 - 0.811002483670372 z_k1_s0037
                                 - 0.462000794608986 z_k1_s0040
                                 - 0.34600028512223 z_k1_s0041
                                 - 1.08600193143162 z_k1_s0042
                                 - 0.583000740434711 z_k1_s0043
                                 - 1.70100175724023 z_k1_s0044
                                 - 0.572001246087753 z_k1_s0045
                                 - 0.686004184242596 z_k1_s0046
                                 - 0.684002940904559 z_k1_s0047
                                 - 0.725999623467916 z_k1_s0050
                                 - 1.02799833377611 z_k1_s0051
                                 - 1.09000245410873 z_k1_s0052
                                 - 1.75100057667297 z_k1_s0053
                                 - 1.07800238562496 z_k1_s0054
                                 - 0.764998716090198 z_k1_s0055
                                 - 0.459999612011083 z_k1_s0056
                                 - 0.635000234082658 z_k1_s0057
                                 - 0.415999848611901 z_k1_s0060
                                 - 0.344000233659866 z_k1_s0061
                                 - 0.246999948988319 z_k1_s0062
                                 - 0.175999888105032 z_k1_s0063
                                 - 0.627999233192929 z_k1_s0064
                                 - 0.329000702628219 z_k1_s0065
                                 - 0.708000878695333 z_k1_s0066
                                 - 0.262000102020009 z_k1_s0067
                                 - 0.493999816076651 z_k1_s0070
                                 - 0.260000011771944 z_k1_s0071
                                 - 0.582000200904155 z_k1_s0072
                                 - 0.137000163754095 z_k1_s0073
                                 - 0.234000099004767 z_k1_s0074
                                 - 0.424998744098111 z_k1_s0075
                                 - 0.499999909292177 z_k1_s0076
                                 - 0.31499899981388 z_k1_s0077
                                 - 0.161000356521702 z_k1_s0100
                                 - 0.294000192996233 z_k1_s0101
                                 - 0.14000004173631 z_k1_s0102
                                 - 0.267000497772032 z_k1_s0103
                                 - 0.190999516215348 z_k1_s0104
                                 - 0.247999876132928 z_k1_s0105
                                 - 0.115000039658397 z_k1_s0106
                                 - 0.0889999652573796 z_k1_s0107
                                 - 0.272999421816301 z_k1_s0110
                                 - 0.348999413262361 z_k1_s0111
                                 - 0.538997628880243 z_k1_s0112
                                 - 0.608999198420223 z_k1_s0113
                                 - 0.605998577069368 z_k1_s0114
                                 - 0.314999658341381 z_k1_s0115
                                 - 0.239999638945971 z_k1_s0116
                                 - 0.594997474114412 z_k1_s0117
                                 - 0.456999939598074 z_k1_s0120
                                 - 0.482999762148601 z_k1_s0121
                                 - 0.357999942809875 z_k1_s0122
                                 - 1.4980001450981 z_k1_s0123
                                 - 1.39399859889716 z_k1_s0124
                                 - 0.486999648079497 z_k1_s0125
                                 - 0.320999869949677 z_k1_s0126
                                 - 0.642000867898771 z_k1_s0127
                                 - 0.292999626300324 z_k1_s0130
                                 - 0.114000053378726 z_k1_s0131
                                 - 0.414000755370399 z_k1_s0132
                                 - 0.377999320902706 z_k1_s0133
                                 - 0.141000079828958 z_k1_s0134
                                 - 0.0980000182711716 z_k1_s0135
                                 - 0.153000149884646 z_k1_s0136
                                 - 0.158000018749434 z_k1_s0137
                                 - 0.835998179338816 z_k1_s0140
                                 - 1.21499808478474 z_k1_s0141
                                 - 0.69999673805102 z_k1_s0142
                                 - 0.34599988630369 z_k1_s0143
                                 - 1.02799682351328 z_k1_s0144
                                 - 0.477999386473193 z_k1_s0145
                                 - 0.670998083627672 z_k1_s0146
                                 - 0.48999944583395 z_k1_s0147
                                 - 0.463000420990846 z_k1_s0150
                                 - 0.483000159068652 z_k1_s0151
                                 - 0.483998377475365 z_k1_s0152
                                 - 0.595999268707312 z_k1_s0153
                                 - 0.836998274041418 z_k1_s0154
                                 - 0.474998691840039 z_k1_s0155
                                 - 0.254999607987315 z_k1_s0156
                                 - 0.336999320105063 z_k1_s0157
                                 - 0.129000179330875 z_k1_s0160
                                 - 0.179000630220782 z_k1_s0161
                                 - 0.182999829627053 z_k1_s0162
                                 - 0.140000113550473 z_k1_s0163
                                 - 0.299000180434871 z_k1_s0164
                                 - 0.283000436806542 z_k1_s0165
                                 - 0.0740000663323999 z_k1_s0166
                                 - 0.186000333932973 z_k1_s0167
                                 - 0.122000044113972 z_k1_s0170
                                 - 0.129000037976244 z_k1_s0171
                                 - 0.10900001774116 z_k1_s0172
                                 - 0.328000278975357 z_k1_s0173
                                 - 0.15500061952216 z_k1_s0174
                                 - 0.262000714559855 z_k1_s0175
                                 - 0.117000105155237 z_k1_s0176
                                 - 0.100000101175423 z_k1_s0177
                                 - 0.343000512950894 z_k1_s1000
                                 - 0.332000537445227 z_k1_s1001
                                 - 0.49800080616784 z_k1_s1002
                                 - 0.175000025254269 z_k1_s1003
                                 - 0.434999350542659 z_k1_s1004
                                 - 0.839000046843525 z_k1_s1005
                                 - 0.427001283449805 z_k1_s1006
                                 - 0.461998971238025 z_k1_s1007
                                 - 0.58900008227618 z_k1_s1010
                                 - 0.205000449000014 z_k1_s1011
                                 - 0.78000422263533 z_k1_s1012
                                 - 0.857004151608422 z_k1_s1013
                                 - 0.298000563401759 z_k1_s1014
                                 - 0.84200397776906 z_k1_s1015
                                 - 0.633003482411882 z_k1_s1016
                                 - 0.613003250626065 z_k1_s1017
                                 - 0.0820000691771282 z_k1_s1020
                                 - 0.171999980834757 z_k1_s1021
                                 - 0.215000312793846 z_k1_s1022
                                 - 0.146999631207195 z_k1_s1023
                                 - 0.0930000883130961 z_k1_s1024
                                 - 0.235000592495896 z_k1_s1025
                                 - 0.219000907534894 z_k1_s1026
                                 - 0.184000418056306 z_k1_s1027
                                 - 0.18200058757891 z_k1_s1030
                                 - 0.36400117515782 z_k1_s1031
                                 - 0.209000675946665 z_k1_s1032
                                 - 0.17200066804308 z_k1_s1033
                                 - 0.307000673280405 z_k1_s1034
                                 - 0.298000134456132 z_k1_s1035
                                 - 0.319000664043691 z_k1_s1036
                                 - 0.332000472119981 z_k1_s1037
                                 - 0.148000135204386 z_k1_s1040
                                 - 0.449000098709617 z_k1_s1041
                                 - 0.133000049414005 z_k1_s1042
                                 - 0.355999405769091 z_k1_s1043
                                 - 0.268999471825442 z_k1_s1044
                                 - 0.330999933785792 z_k1_s1045
                                 - 0.192000131875917 z_k1_s1046
                                 - 0.13500016821283 z_k1_s1047
                                 - 0.150000454454101 z_k1_s1050
                                 - 0.0930000868814121 z_k1_s1051
                                 - 0.1590002624812 z_k1_s1052
                                 - 0.0650000751300289 z_k1_s1053
                                 - 0.20500049928719 z_k1_s1054
                                 - 0.102000016708593 z_k1_s1055
                                 - 0.203000541947834 z_k1_s1056
                                 - 0.240999731395587 z_k1_s1057
                                 - 0.764000355552916 z_k1_s1060
                                 - 0.260000165341623 z_k1_s1061
                                 - 0.436000097331116 z_k1_s1062
                                 - 0.937000135172514 z_k1_s1063
                                 - 0.245000210093134 z_k1_s1064
                                 - 0.44100102344843 z_k1_s1065
                                 - 0.272000371520709 z_k1_s1066
                                 - 0.678000558354848 z_k1_s1067
                                 - 0.0980001147426061 z_k1_s1070
                                 - 0.210999831576137 z_k1_s1071
                                 - 0.16999987190021 z_k1_s1072
                                 - 0.267000675743885 z_k1_s1073
                                 - 0.116000054032007 z_k1_s1074
                                 - 0.232999418962946 z_k1_s1075
                                 - 0.141000138917405 z_k1_s1076
                                 - 0.367000694685251 z_k1_s1077
                                 - 0.146999419315815 z_k1_s1100
                                 - 0.23800014250993 z_k1_s1101
                                 - 0.0939998819444818 z_k1_s1102
                                 - 0.14299930945024 z_k1_s1103
                                 - 0.106999883758057 z_k1_s1104
                                 - 0.191999589554904 z_k1_s1105
                                 - 0.126999864686662 z_k1_s1106
                                 - 0.372999232229019 z_k1_s1107
                                 - 0.0929998920624771 z_k1_s1110
                                 - 0.107999960169255 z_k1_s1111
                                 - 0.195000162388731 z_k1_s1112
                                 - 0.238999494569358 z_k1_s1113
                                 - 0.159999762473123 z_k1_s1114
                                 - 0.142999637085483 z_k1_s1115
                                 - 0.141999487945156 z_k1_s1116
                                 - 0.124999362557516 z_k1_s1117
                                 - 0.387998979681349 z_k1_s1120
                                 - 0.399998854300621 z_k1_s1121
                                 - 1.24199564507065 z_k1_s1122
                                 - 0.65299972351383 z_k1_s1123
                                 - 0.390998948336167 z_k1_s1124
                                 - 0.673999025098888 z_k1_s1125
                                 - 0.383998861808703 z_k1_s1126
                                 - 0.658000469602972 z_k1_s1127
                                 - 0.443000474099109 z_k1_s1130
                                 - 0.255999971841879 z_k1_s1131
                                 - 0.157999880664189 z_k1_s1132
                                 - 0.359000829517972 z_k1_s1133
                                 - 0.672998706047937 z_k1_s1134
                                 - 0.690999280243504 z_k1_s1135
                                 - 0.244999924444405 z_k1_s1136
                                 - 0.210000021952295 z_k1_s1137
                                 - 0.105999513318065 z_k1_s1140
                                 - 0.141999683539185 z_k1_s1141
                                 - 0.0989999974082336 z_k1_s1142
                                 - 0.16200024299521 z_k1_s1143
                                 - 0.0850000241888788 z_k1_s1144
                                 - 0.0699999706967398 z_k1_s1145
                                 - 0.183999707797215 z_k1_s1146
                                 - 0.19800041321633 z_k1_s1147
                                 - 0.56799851627711 z_k1_s1150
                                 - 0.408000577829509 z_k1_s1151
                                 - 0.728000478716523 z_k1_s1152
                                 - 0.225999452152425 z_k1_s1153
                                 - 0.313999324296559 z_k1_s1154
                                 - 0.56500064237697 z_k1_s1155
                                 - 0.931996743639465 z_k1_s1156
                                 - 0.282999296173206 z_k1_s1157
                                 - 0.329000215161115 z_k1_s1160
                                 - 0.197000329333838 z_k1_s1161
                                 - 0.330001148570213 z_k1_s1162
                                 - 0.503002020118007 z_k1_s1163
                                 - 0.261000413032107 z_k1_s1164
                                 - 0.278000506964561 z_k1_s1165
                                 - 0.584000190145904 z_k1_s1166
                                 - 0.386001795407006 z_k1_s1167
                                 - 0.158999794140052 z_k1_s1170
                                 - 0.25599941370251 z_k1_s1171
                                 - 0.398998968661466 z_k1_s1172
                                 - 0.182999751712148 z_k1_s1173
                                 - 0.155999824518512 z_k1_s1174
                                 - 0.457000387342318 z_k1_s1175
                                 - 0.128999897324876 z_k1_s1176
                                 - 0.267000305313694 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.723495 w_k1_ADM_s000
                                 + 0.684569 w_k1_AXP_s000
                                 + 0.715403 w_k1_DIS_s000
                                 + 0.738892 w_k1_FDX_s000
                                 + 0.83694 w_k1_IBM_s000
                                 + 0.624236 w_k1_INTC_s000
                                 + 0.881235 w_k1_KO_s000
                                 + 0.97771 w_k1_MAT_s000
                                 + 0.829596 w_k1_MSFT_s000
                                 + 0.844144 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.938872 w_k1_ADM_s000
                                 + 1.13949 w_k1_AXP_s000
                                 + 1.29179 w_k1_DIS_s000 + 1.1595 w_k1_FDX_s000
                                 + 1.17885 w_k1_IBM_s000
                                 + 0.98986 w_k1_INTC_s000
                                 + 1.09491 w_k1_KO_s000
                                 + 0.970823 w_k1_MAT_s000
                                 + 1.00363 w_k1_MSFT_s000
                                 + 1.02694 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.942568 w_k1_ADM_s000
                                 + 1.04499 w_k1_AXP_s000
                                 + 0.780622 w_k1_DIS_s000
                                 + 0.856169 w_k1_FDX_s000
                                 + 0.852671 w_k1_IBM_s000
                                 + 0.669353 w_k1_INTC_s000
                                 + 0.768468 w_k1_KO_s000
                                 + 1.49759 w_k1_MAT_s000
                                 + 0.765554 w_k1_MSFT_s000
                                 + 0.898561 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.866555 w_k1_ADM_s000
                                 + 0.976868 w_k1_AXP_s000
                                 + 1.03216 w_k1_DIS_s000
                                 + 0.880411 w_k1_FDX_s000
                                 + 1.11206 w_k1_IBM_s000
                                 + 0.829164 w_k1_INTC_s000
                                 + 0.995521 w_k1_KO_s000
                                 + 1.32862 w_k1_MAT_s000
                                 + 0.929876 w_k1_MSFT_s000
                                 + 1.03279 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.86763 w_k1_ADM_s000
                                 + 1.15389 w_k1_AXP_s000
                                 + 0.875839 w_k1_DIS_s000
                                 + 1.05513 w_k1_FDX_s000
                                 + 0.799094 w_k1_IBM_s000
                                 + 0.830891 w_k1_INTC_s000
                                 + 0.931784 w_k1_KO_s000
                                 + 1.61231 w_k1_MAT_s000
                                 + 1.07826 w_k1_MSFT_s000
                                 + 0.879372 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.85335 w_k1_ADM_s000
                                 + 0.874411 w_k1_AXP_s000
                                 + 0.716792 w_k1_DIS_s000
                                 + 0.952675 w_k1_FDX_s000
                                 + 0.640096 w_k1_IBM_s000
                                 + 0.639432 w_k1_INTC_s000
                                 + 0.764689 w_k1_KO_s000
                                 + 0.804669 w_k1_MAT_s000
                                 + 0.831452 w_k1_MSFT_s000
                                 + 0.772836 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.983001 w_k1_ADM_s000
                                 + 1.7548 w_k1_AXP_s000 + 1.74281 w_k1_DIS_s000
                                 + 1.45655 w_k1_FDX_s000
                                 + 1.20313 w_k1_IBM_s000
                                 + 1.12101 w_k1_INTC_s000
                                 + 1.29089 w_k1_KO_s000 + 1.99879 w_k1_MAT_s000
                                 + 1.18392 w_k1_MSFT_s000
                                 + 1.36067 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.65397 w_k1_ADM_s000
                                 + 0.93653 w_k1_AXP_s000
                                 + 0.596995 w_k1_DIS_s000
                                 + 0.719249 w_k1_FDX_s000
                                 + 0.896683 w_k1_IBM_s000
                                 + 0.624142 w_k1_INTC_s000
                                 + 0.815281 w_k1_KO_s000
                                 + 1.55769 w_k1_MAT_s000
                                 + 0.750812 w_k1_MSFT_s000
                                 + 0.784486 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.11574 w_k1_ADM_s001
                                 + 1.41295 w_k1_AXP_s001
                                 + 1.16195 w_k1_DIS_s001
                                 + 1.17163 w_k1_FDX_s001
                                 + 0.937486 w_k1_IBM_s001
                                 + 1.22285 w_k1_INTC_s001
                                 + 0.848549 w_k1_KO_s001
                                 + 0.86158 w_k1_MAT_s001
                                 + 0.844449 w_k1_MSFT_s001
                                 + 1.10286 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.628718 w_k1_ADM_s001
                                 + 0.747188 w_k1_AXP_s001
                                 + 0.629321 w_k1_DIS_s001
                                 + 0.716728 w_k1_FDX_s001
                                 + 0.941364 w_k1_IBM_s001
                                 + 0.907218 w_k1_INTC_s001
                                 + 0.770733 w_k1_KO_s001
                                 + 0.962584 w_k1_MAT_s001
                                 + 0.717618 w_k1_MSFT_s001
                                 + 0.8436 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.892639 w_k1_ADM_s001
                                 + 0.885096 w_k1_AXP_s001
                                 + 0.725348 w_k1_DIS_s001
                                 + 0.905924 w_k1_FDX_s001
                                 + 0.757873 w_k1_IBM_s001
                                 + 0.923791 w_k1_INTC_s001
                                 + 0.677286 w_k1_KO_s001
                                 + 0.643501 w_k1_MAT_s001
                                 + 0.756713 w_k1_MSFT_s001
                                 + 0.900299 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.26792 w_k1_ADM_s001
                                 + 1.37822 w_k1_AXP_s001
                                 + 1.67405 w_k1_DIS_s001
                                 + 1.34901 w_k1_FDX_s001 + 1.5261 w_k1_IBM_s001
                                 + 1.44179 w_k1_INTC_s001
                                 + 1.20265 w_k1_KO_s001 + 1.44223 w_k1_MAT_s001
                                 + 1.21117 w_k1_MSFT_s001
                                 + 1.23706 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.963303 w_k1_ADM_s001
                                 + 0.953669 w_k1_AXP_s001
                                 + 1.24196 w_k1_DIS_s001
                                 + 1.04328 w_k1_FDX_s001
                                 + 0.931355 w_k1_IBM_s001
                                 + 1.09166 w_k1_INTC_s001
                                 + 0.986692 w_k1_KO_s001
                                 + 0.995677 w_k1_MAT_s001
                                 + 1.14881 w_k1_MSFT_s001
                                 + 1.03206 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.914353 w_k1_ADM_s001
                                 + 1.1577 w_k1_AXP_s001 + 1.048 w_k1_DIS_s001
                                 + 1.00813 w_k1_FDX_s001
                                 + 1.32653 w_k1_IBM_s001
                                 + 1.33625 w_k1_INTC_s001
                                 + 0.94816 w_k1_KO_s001 + 1.1301 w_k1_MAT_s001
                                 + 0.949752 w_k1_MSFT_s001
                                 + 0.957235 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.928555 w_k1_ADM_s001
                                 + 0.778891 w_k1_AXP_s001
                                 + 0.86442 w_k1_DIS_s001
                                 + 0.822293 w_k1_FDX_s001
                                 + 0.845206 w_k1_IBM_s001
                                 + 0.909278 w_k1_INTC_s001
                                 + 0.76994 w_k1_KO_s001 + 0.72488 w_k1_MAT_s001
                                 + 0.942597 w_k1_MSFT_s001
                                 + 0.903512 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.899628 w_k1_ADM_s001
                                 + 0.81723 w_k1_AXP_s001
                                 + 1.00178 w_k1_DIS_s001
                                 + 0.791064 w_k1_FDX_s001
                                 + 1.12082 w_k1_IBM_s001
                                 + 0.882472 w_k1_INTC_s001
                                 + 0.864369 w_k1_KO_s001
                                 + 0.833601 w_k1_MAT_s001
                                 + 0.887647 w_k1_MSFT_s001
                                 + 1.01824 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.17099 w_k1_ADM_s002
                                 + 1.23601 w_k1_AXP_s002
                                 + 1.22945 w_k1_DIS_s002
                                 + 1.21235 w_k1_FDX_s002
                                 + 0.948179 w_k1_IBM_s002
                                 + 1.05917 w_k1_INTC_s002
                                 + 1.07626 w_k1_KO_s002
                                 + 0.944808 w_k1_MAT_s002
                                 + 1.17948 w_k1_MSFT_s002
                                 + 1.29852 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.865619 w_k1_ADM_s002
                                 + 0.638778 w_k1_AXP_s002
                                 + 1.01101 w_k1_DIS_s002
                                 + 0.862094 w_k1_FDX_s002
                                 + 0.979572 w_k1_IBM_s002
                                 + 0.81716 w_k1_INTC_s002
                                 + 0.980139 w_k1_KO_s002
                                 + 0.912303 w_k1_MAT_s002
                                 + 1.02967 w_k1_MSFT_s002
                                 + 1.02626 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.06475 w_k1_ADM_s002
                                 + 1.09179 w_k1_AXP_s002
                                 + 0.906523 w_k1_DIS_s002
                                 + 1.07955 w_k1_FDX_s002
                                 + 0.752285 w_k1_IBM_s002
                                 + 0.869549 w_k1_INTC_s002
                                 + 0.954819 w_k1_KO_s002
                                 + 0.815033 w_k1_MAT_s002
                                 + 1.06839 w_k1_MSFT_s002
                                 + 1.04637 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.42705 w_k1_ADM_s002
                                 + 1.50198 w_k1_AXP_s002 + 1.5927 w_k1_DIS_s002
                                 + 1.33723 w_k1_FDX_s002
                                 + 1.45521 w_k1_IBM_s002
                                 + 1.33433 w_k1_INTC_s002
                                 + 1.33087 w_k1_KO_s002 + 1.387 w_k1_MAT_s002
                                 + 1.39208 w_k1_MSFT_s002
                                 + 1.29216 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.832951 w_k1_ADM_s002
                                 + 0.694468 w_k1_AXP_s002
                                 + 0.64136 w_k1_DIS_s002
                                 + 0.782736 w_k1_FDX_s002
                                 + 0.707016 w_k1_IBM_s002
                                 + 0.657501 w_k1_INTC_s002
                                 + 0.676403 w_k1_KO_s002
                                 + 0.386229 w_k1_MAT_s002
                                 + 0.787163 w_k1_MSFT_s002
                                 + 0.889493 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.99586 w_k1_ADM_s002
                                 + 0.923905 w_k1_AXP_s002
                                 + 0.873908 w_k1_DIS_s002
                                 + 0.919217 w_k1_FDX_s002
                                 + 1.02577 w_k1_IBM_s002
                                 + 0.81582 w_k1_INTC_s002
                                 + 0.90931 w_k1_KO_s002
                                 + 0.948382 w_k1_MAT_s002
                                 + 1.00804 w_k1_MSFT_s002
                                 + 0.897518 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.02806 w_k1_ADM_s002
                                 + 0.883404 w_k1_AXP_s002
                                 + 0.785482 w_k1_DIS_s002
                                 + 0.84289 w_k1_FDX_s002
                                 + 1.13218 w_k1_IBM_s002
                                 + 0.67852 w_k1_INTC_s002
                                 + 0.916043 w_k1_KO_s002
                                 + 1.12123 w_k1_MAT_s002
                                 + 0.80814 w_k1_MSFT_s002
                                 + 1.08503 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.17402 w_k1_ADM_s002
                                 + 1.05898 w_k1_AXP_s002
                                 + 1.09691 w_k1_DIS_s002
                                 + 1.03555 w_k1_FDX_s002
                                 + 1.37258 w_k1_IBM_s002
                                 + 0.865465 w_k1_INTC_s002
                                 + 1.2376 w_k1_KO_s002 + 1.12864 w_k1_MAT_s002
                                 + 0.972804 w_k1_MSFT_s002
                                 + 1.07236 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.910493 w_k1_ADM_s003
                                 + 0.95525 w_k1_AXP_s003
                                 + 1.09235 w_k1_DIS_s003
                                 + 0.817554 w_k1_FDX_s003
                                 + 1.41529 w_k1_IBM_s003
                                 + 1.17585 w_k1_INTC_s003
                                 + 1.18004 w_k1_KO_s003 + 1.08046 w_k1_MAT_s003
                                 + 1.00399 w_k1_MSFT_s003
                                 + 1.09449 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.22788 w_k1_ADM_s003
                                 + 0.921075 w_k1_AXP_s003
                                 + 1.20709 w_k1_DIS_s003
                                 + 1.09664 w_k1_FDX_s003
                                 + 0.81595 w_k1_IBM_s003
                                 + 1.13237 w_k1_INTC_s003
                                 + 1.04015 w_k1_KO_s003
                                 + 0.866098 w_k1_MAT_s003
                                 + 1.16717 w_k1_MSFT_s003
                                 + 1.20663 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.52858 w_k1_ADM_s003
                                 + 1.46066 w_k1_AXP_s003 + 1.5912 w_k1_DIS_s003
                                 + 1.3709 w_k1_FDX_s003 + 1.23877 w_k1_IBM_s003
                                 + 1.24391 w_k1_INTC_s003
                                 + 1.44602 w_k1_KO_s003 + 1.52393 w_k1_MAT_s003
                                 + 1.36737 w_k1_MSFT_s003
                                 + 1.50466 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.06518 w_k1_ADM_s003
                                 + 1.14977 w_k1_AXP_s003
                                 + 1.03246 w_k1_DIS_s003
                                 + 1.04074 w_k1_FDX_s003
                                 + 0.912327 w_k1_IBM_s003
                                 + 1.14198 w_k1_INTC_s003
                                 + 0.811482 w_k1_KO_s003
                                 + 1.01603 w_k1_MAT_s003
                                 + 1.0139 w_k1_MSFT_s003 + 1.06649 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.45148 w_k1_ADM_s003
                                 + 1.38691 w_k1_AXP_s003
                                 + 1.56117 w_k1_DIS_s003 + 1.1756 w_k1_FDX_s003
                                 + 1.60177 w_k1_IBM_s003
                                 + 1.37201 w_k1_INTC_s003
                                 + 1.42179 w_k1_KO_s003 + 1.67171 w_k1_MAT_s003
                                 + 1.24986 w_k1_MSFT_s003
                                 + 1.24987 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.05568 w_k1_ADM_s003
                                 + 0.856137 w_k1_AXP_s003
                                 + 1.04297 w_k1_DIS_s003
                                 + 0.956348 w_k1_FDX_s003
                                 + 1.0179 w_k1_IBM_s003
                                 + 1.08184 w_k1_INTC_s003
                                 + 1.07695 w_k1_KO_s003
                                 + 0.951193 w_k1_MAT_s003
                                 + 1.01236 w_k1_MSFT_s003
                                 + 1.01723 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.15215 w_k1_ADM_s003
                                 + 1.01899 w_k1_AXP_s003
                                 + 0.981987 w_k1_DIS_s003
                                 + 0.92332 w_k1_FDX_s003
                                 + 1.19107 w_k1_IBM_s003
                                 + 1.19775 w_k1_INTC_s003
                                 + 1.07997 w_k1_KO_s003 + 1.25389 w_k1_MAT_s003
                                 + 1.10412 w_k1_MSFT_s003
                                 + 0.952238 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.91386 w_k1_ADM_s003
                                 + 0.741468 w_k1_AXP_s003
                                 + 0.675917 w_k1_DIS_s003
                                 + 0.75199 w_k1_FDX_s003
                                 + 0.698949 w_k1_IBM_s003
                                 + 0.666283 w_k1_INTC_s003
                                 + 0.754104 w_k1_KO_s003
                                 + 0.737178 w_k1_MAT_s003
                                 + 0.781549 w_k1_MSFT_s003
                                 + 0.93076 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.31063 w_k1_ADM_s004
                                 + 1.48277 w_k1_AXP_s004
                                 + 1.36261 w_k1_DIS_s004
                                 + 1.17258 w_k1_FDX_s004
                                 + 1.55732 w_k1_IBM_s004
                                 + 1.24124 w_k1_INTC_s004 + 1.2782 w_k1_KO_s004
                                 + 1.05771 w_k1_MAT_s004
                                 + 1.18659 w_k1_MSFT_s004
                                 + 1.09799 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.31218 w_k1_ADM_s004
                                 + 1.05635 w_k1_AXP_s004
                                 + 1.50009 w_k1_DIS_s004
                                 + 1.20766 w_k1_FDX_s004
                                 + 1.09426 w_k1_IBM_s004
                                 + 1.33025 w_k1_INTC_s004
                                 + 1.23034 w_k1_KO_s004
                                 + 0.830888 w_k1_MAT_s004
                                 + 1.23785 w_k1_MSFT_s004
                                 + 1.09574 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.13054 w_k1_ADM_s004
                                 + 0.976427 w_k1_AXP_s004
                                 + 0.748167 w_k1_DIS_s004
                                 + 0.954527 w_k1_FDX_s004
                                 + 0.848076 w_k1_IBM_s004
                                 + 0.892203 w_k1_INTC_s004
                                 + 0.753662 w_k1_KO_s004
                                 + 0.762563 w_k1_MAT_s004
                                 + 0.880743 w_k1_MSFT_s004
                                 + 0.868509 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.03258 w_k1_ADM_s004
                                 + 1.3871 w_k1_AXP_s004 + 1.07742 w_k1_DIS_s004
                                 + 1.16366 w_k1_FDX_s004
                                 + 0.971418 w_k1_IBM_s004
                                 + 1.15523 w_k1_INTC_s004
                                 + 1.00948 w_k1_KO_s004
                                 + 0.915161 w_k1_MAT_s004
                                 + 1.03479 w_k1_MSFT_s004
                                 + 1.12317 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.912789 w_k1_ADM_s004
                                 + 0.771065 w_k1_AXP_s004
                                 + 0.63712 w_k1_DIS_s004
                                 + 0.758143 w_k1_FDX_s004
                                 + 0.87854 w_k1_IBM_s004
                                 + 0.896144 w_k1_INTC_s004
                                 + 0.80033 w_k1_KO_s004
                                 + 0.626924 w_k1_MAT_s004
                                 + 0.799389 w_k1_MSFT_s004
                                 + 0.799886 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.56137 w_k1_ADM_s004
                                 + 1.71334 w_k1_AXP_s004
                                 + 2.08173 w_k1_DIS_s004
                                 + 1.58805 w_k1_FDX_s004
                                 + 1.29867 w_k1_IBM_s004
                                 + 1.60588 w_k1_INTC_s004 + 1.3883 w_k1_KO_s004
                                 + 1.04432 w_k1_MAT_s004
                                 + 1.55249 w_k1_MSFT_s004
                                 + 1.39472 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.11776 w_k1_ADM_s004
                                 + 0.916244 w_k1_AXP_s004
                                 + 1.10512 w_k1_DIS_s004
                                 + 0.947201 w_k1_FDX_s004
                                 + 1.1891 w_k1_IBM_s004
                                 + 0.979959 w_k1_INTC_s004
                                 + 1.04175 w_k1_KO_s004
                                 + 0.689855 w_k1_MAT_s004
                                 + 0.936666 w_k1_MSFT_s004
                                 + 0.933892 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.959465 w_k1_ADM_s004
                                 + 1.04342 w_k1_AXP_s004
                                 + 1.07329 w_k1_DIS_s004
                                 + 1.09726 w_k1_FDX_s004
                                 + 0.850124 w_k1_IBM_s004
                                 + 1.13539 w_k1_INTC_s004
                                 + 0.913444 w_k1_KO_s004
                                 + 0.601556 w_k1_MAT_s004
                                 + 0.974904 w_k1_MSFT_s004
                                 + 1.00196 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.811616 w_k1_ADM_s005
                                 + 0.807272 w_k1_AXP_s005
                                 + 0.67373 w_k1_DIS_s005
                                 + 0.806273 w_k1_FDX_s005
                                 + 0.790413 w_k1_IBM_s005
                                 + 0.910161 w_k1_INTC_s005
                                 + 0.775432 w_k1_KO_s005
                                 + 1.00473 w_k1_MAT_s005
                                 + 0.88988 w_k1_MSFT_s005
                                 + 0.920907 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.990173 w_k1_ADM_s005
                                 + 1.46494 w_k1_AXP_s005
                                 + 1.90474 w_k1_DIS_s005
                                 + 1.52428 w_k1_FDX_s005
                                 + 1.32169 w_k1_IBM_s005
                                 + 1.38994 w_k1_INTC_s005
                                 + 1.55308 w_k1_KO_s005 + 1.81458 w_k1_MAT_s005
                                 + 1.62069 w_k1_MSFT_s005
                                 + 1.44143 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.16926 w_k1_ADM_s005
                                 + 1.35961 w_k1_AXP_s005
                                 + 1.31279 w_k1_DIS_s005
                                 + 1.27694 w_k1_FDX_s005
                                 + 1.04949 w_k1_IBM_s005
                                 + 1.15131 w_k1_INTC_s005
                                 + 1.23137 w_k1_KO_s005 + 1.81768 w_k1_MAT_s005
                                 + 1.23836 w_k1_MSFT_s005
                                 + 1.03449 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.00389 w_k1_ADM_s005
                                 + 0.980727 w_k1_AXP_s005
                                 + 0.926474 w_k1_DIS_s005
                                 + 1.02858 w_k1_FDX_s005
                                 + 0.919715 w_k1_IBM_s005
                                 + 0.914605 w_k1_INTC_s005
                                 + 1.0186 w_k1_KO_s005 + 1.23652 w_k1_MAT_s005
                                 + 1.03282 w_k1_MSFT_s005
                                 + 0.994081 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.895678 w_k1_ADM_s005
                                 + 0.817999 w_k1_AXP_s005
                                 + 0.939362 w_k1_DIS_s005
                                 + 0.887346 w_k1_FDX_s005
                                 + 1.20317 w_k1_IBM_s005
                                 + 1.01875 w_k1_INTC_s005
                                 + 1.13013 w_k1_KO_s005 + 1.43277 w_k1_MAT_s005
                                 + 0.917285 w_k1_MSFT_s005
                                 + 1.05254 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.653996 w_k1_ADM_s005
                                 + 0.536408 w_k1_AXP_s005
                                 + 0.502623 w_k1_DIS_s005
                                 + 0.606924 w_k1_FDX_s005
                                 + 0.773293 w_k1_IBM_s005
                                 + 0.817848 w_k1_INTC_s005
                                 + 0.711334 w_k1_KO_s005
                                 + 0.84461 w_k1_MAT_s005
                                 + 0.778078 w_k1_MSFT_s005
                                 + 0.791613 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.984535 w_k1_ADM_s005
                                 + 1.08381 w_k1_AXP_s005
                                 + 0.962976 w_k1_DIS_s005
                                 + 0.872892 w_k1_FDX_s005
                                 + 1.66782 w_k1_IBM_s005
                                 + 0.866848 w_k1_INTC_s005
                                 + 1.12929 w_k1_KO_s005 + 1.81719 w_k1_MAT_s005
                                 + 0.890808 w_k1_MSFT_s005
                                 + 1.0263 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.839641 w_k1_ADM_s005
                                 + 0.527259 w_k1_AXP_s005
                                 + 0.759781 w_k1_DIS_s005
                                 + 0.773961 w_k1_FDX_s005
                                 + 0.714225 w_k1_IBM_s005
                                 + 0.828295 w_k1_INTC_s005
                                 + 0.790709 w_k1_KO_s005
                                 + 0.687987 w_k1_MAT_s005
                                 + 0.862658 w_k1_MSFT_s005
                                 + 0.879894 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.932991 w_k1_ADM_s006
                                 + 0.943835 w_k1_AXP_s006
                                 + 1.09795 w_k1_DIS_s006
                                 + 0.972344 w_k1_FDX_s006
                                 + 1.3284 w_k1_IBM_s006
                                 + 0.814228 w_k1_INTC_s006
                                 + 1.24158 w_k1_KO_s006 + 1.51484 w_k1_MAT_s006
                                 + 0.948602 w_k1_MSFT_s006
                                 + 0.959901 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.907966 w_k1_ADM_s006
                                 + 0.935184 w_k1_AXP_s006
                                 + 0.695307 w_k1_DIS_s006
                                 + 0.803012 w_k1_FDX_s006
                                 + 0.749718 w_k1_IBM_s006
                                 + 0.750371 w_k1_INTC_s006
                                 + 0.853136 w_k1_KO_s006
                                 + 1.50919 w_k1_MAT_s006
                                 + 0.795647 w_k1_MSFT_s006
                                 + 0.814272 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.31414 w_k1_ADM_s006
                                 + 1.29675 w_k1_AXP_s006 + 1.9165 w_k1_DIS_s006
                                 + 1.45206 w_k1_FDX_s006
                                 + 1.39928 w_k1_IBM_s006
                                 + 1.08501 w_k1_INTC_s006
                                 + 1.55398 w_k1_KO_s006 + 1.98717 w_k1_MAT_s006
                                 + 1.55115 w_k1_MSFT_s006 + 1.3166 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.583736 w_k1_ADM_s006
                                 + 0.499094 w_k1_AXP_s006
                                 + 0.571681 w_k1_DIS_s006
                                 + 0.614863 w_k1_FDX_s006
                                 + 0.984805 w_k1_IBM_s006
                                 + 0.592338 w_k1_INTC_s006
                                 + 0.831654 w_k1_KO_s006
                                 + 1.39587 w_k1_MAT_s006
                                 + 0.711739 w_k1_MSFT_s006
                                 + 0.72862 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.00933 w_k1_ADM_s006
                                 + 0.984205 w_k1_AXP_s006
                                 + 0.939881 w_k1_DIS_s006
                                 + 0.949484 w_k1_FDX_s006
                                 + 0.951556 w_k1_IBM_s006
                                 + 0.811459 w_k1_INTC_s006
                                 + 0.938983 w_k1_KO_s006
                                 + 1.51872 w_k1_MAT_s006
                                 + 0.914729 w_k1_MSFT_s006
                                 + 0.90718 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.19858 w_k1_ADM_s006
                                 + 1.33541 w_k1_AXP_s006
                                 + 1.37309 w_k1_DIS_s006
                                 + 1.19784 w_k1_FDX_s006
                                 + 1.41473 w_k1_IBM_s006
                                 + 0.998235 w_k1_INTC_s006
                                 + 1.32782 w_k1_KO_s006 + 2.63889 w_k1_MAT_s006
                                 + 1.18047 w_k1_MSFT_s006
                                 + 1.08449 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.11485 w_k1_ADM_s006
                                 + 1.27358 w_k1_AXP_s006 + 1.0623 w_k1_DIS_s006
                                 + 1.16971 w_k1_FDX_s006
                                 + 0.90467 w_k1_IBM_s006
                                 + 1.00299 w_k1_INTC_s006
                                 + 0.904881 w_k1_KO_s006
                                 + 1.50707 w_k1_MAT_s006
                                 + 1.15709 w_k1_MSFT_s006
                                 + 1.05148 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.997453 w_k1_ADM_s006
                                 + 0.880606 w_k1_AXP_s006
                                 + 0.715363 w_k1_DIS_s006
                                 + 0.791707 w_k1_FDX_s006
                                 + 1.18648 w_k1_IBM_s006
                                 + 0.74753 w_k1_INTC_s006
                                 + 1.01016 w_k1_KO_s006 + 1.70838 w_k1_MAT_s006
                                 + 0.852653 w_k1_MSFT_s006
                                 + 0.958893 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.921885 w_k1_ADM_s007
                                 + 0.835124 w_k1_AXP_s007
                                 + 0.929983 w_k1_DIS_s007
                                 + 0.930731 w_k1_FDX_s007
                                 + 1.08042 w_k1_IBM_s007
                                 + 1.02742 w_k1_INTC_s007
                                 + 1.00816 w_k1_KO_s007
                                 + 0.665249 w_k1_MAT_s007
                                 + 0.870459 w_k1_MSFT_s007
                                 + 0.910168 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.857184 w_k1_ADM_s007
                                 + 0.600873 w_k1_AXP_s007
                                 + 0.618275 w_k1_DIS_s007
                                 + 0.782118 w_k1_FDX_s007
                                 + 0.824973 w_k1_IBM_s007
                                 + 0.762695 w_k1_INTC_s007
                                 + 0.774807 w_k1_KO_s007
                                 + 0.516812 w_k1_MAT_s007
                                 + 0.757839 w_k1_MSFT_s007
                                 + 0.822211 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.07571 w_k1_ADM_s007
                                 + 1.03089 w_k1_AXP_s007
                                 + 1.22843 w_k1_DIS_s007
                                 + 1.11546 w_k1_FDX_s007
                                 + 1.17485 w_k1_IBM_s007
                                 + 0.897317 w_k1_INTC_s007
                                 + 1.15766 w_k1_KO_s007
                                 + 0.998794 w_k1_MAT_s007
                                 + 1.01501 w_k1_MSFT_s007
                                 + 1.03361 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.630533 w_k1_ADM_s007
                                 + 0.714055 w_k1_AXP_s007
                                 + 0.749351 w_k1_DIS_s007
                                 + 0.829843 w_k1_FDX_s007
                                 + 0.802749 w_k1_IBM_s007
                                 + 0.929799 w_k1_INTC_s007
                                 + 0.840521 w_k1_KO_s007
                                 + 0.725176 w_k1_MAT_s007
                                 + 0.810591 w_k1_MSFT_s007
                                 + 0.814268 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.29813 w_k1_ADM_s007
                                 + 1.23294 w_k1_AXP_s007
                                 + 0.969018 w_k1_DIS_s007
                                 + 1.24084 w_k1_FDX_s007
                                 + 0.949151 w_k1_IBM_s007
                                 + 1.15344 w_k1_INTC_s007 + 1.0867 w_k1_KO_s007
                                 + 0.901083 w_k1_MAT_s007
                                 + 1.09458 w_k1_MSFT_s007
                                 + 1.14472 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.08339 w_k1_ADM_s007
                                 + 1.26748 w_k1_AXP_s007
                                 + 1.53699 w_k1_DIS_s007
                                 + 1.22839 w_k1_FDX_s007
                                 + 1.41819 w_k1_IBM_s007
                                 + 1.01281 w_k1_INTC_s007
                                 + 1.38862 w_k1_KO_s007 + 1.42569 w_k1_MAT_s007
                                 + 1.04002 w_k1_MSFT_s007
                                 + 1.08255 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.991935 w_k1_ADM_s007
                                 + 0.874387 w_k1_AXP_s007
                                 + 0.891848 w_k1_DIS_s007
                                 + 0.954451 w_k1_FDX_s007
                                 + 0.790743 w_k1_IBM_s007
                                 + 1.00373 w_k1_INTC_s007
                                 + 0.923166 w_k1_KO_s007
                                 + 0.710028 w_k1_MAT_s007
                                 + 1.04462 w_k1_MSFT_s007
                                 + 0.959004 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.01692 w_k1_ADM_s007
                                 + 1.13043 w_k1_AXP_s007
                                 + 0.815907 w_k1_DIS_s007
                                 + 0.961394 w_k1_FDX_s007
                                 + 1.1618 w_k1_IBM_s007
                                 + 0.788353 w_k1_INTC_s007
                                 + 1.18218 w_k1_KO_s007 + 1.17398 w_k1_MAT_s007
                                 + 0.883227 w_k1_MSFT_s007
                                 + 1.04983 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.17905 w_k1_ADM_s010
                                 + 1.14158 w_k1_AXP_s010
                                 + 1.60524 w_k1_DIS_s010 + 1.3569 w_k1_FDX_s010
                                 + 0.975317 w_k1_IBM_s010
                                 + 1.27549 w_k1_INTC_s010
                                 + 1.12301 w_k1_KO_s010 + 0.77585 w_k1_MAT_s010
                                 + 1.46602 w_k1_MSFT_s010
                                 + 1.13644 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.01739 w_k1_ADM_s010
                                 + 0.968668 w_k1_AXP_s010
                                 + 0.868234 w_k1_DIS_s010
                                 + 1.03746 w_k1_FDX_s010
                                 + 0.858543 w_k1_IBM_s010
                                 + 0.970623 w_k1_INTC_s010
                                 + 0.872444 w_k1_KO_s010
                                 + 0.762526 w_k1_MAT_s010
                                 + 0.95071 w_k1_MSFT_s010
                                 + 0.912981 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.14301 w_k1_ADM_s010
                                 + 1.59085 w_k1_AXP_s010
                                 + 1.00143 w_k1_DIS_s010 + 1.1875 w_k1_FDX_s010
                                 + 1.57094 w_k1_IBM_s010
                                 + 1.33173 w_k1_INTC_s010
                                 + 1.17096 w_k1_KO_s010 + 1.62837 w_k1_MAT_s010
                                 + 1.02171 w_k1_MSFT_s010
                                 + 1.05989 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.821235 w_k1_ADM_s010
                                 + 0.790589 w_k1_AXP_s010
                                 + 0.632327 w_k1_DIS_s010
                                 + 0.809227 w_k1_FDX_s010
                                 + 0.883163 w_k1_IBM_s010
                                 + 0.889838 w_k1_INTC_s010
                                 + 0.661993 w_k1_KO_s010
                                 + 0.762341 w_k1_MAT_s010
                                 + 0.813458 w_k1_MSFT_s010
                                 + 0.826775 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.929955 w_k1_ADM_s010
                                 + 0.911061 w_k1_AXP_s010
                                 + 1.19034 w_k1_DIS_s010
                                 + 1.26412 w_k1_FDX_s010
                                 + 0.886893 w_k1_IBM_s010
                                 + 1.08594 w_k1_INTC_s010
                                 + 1.06365 w_k1_KO_s010
                                 + 0.955595 w_k1_MAT_s010
                                 + 1.21373 w_k1_MSFT_s010 + 1.0179 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.872748 w_k1_ADM_s010
                                 + 1.02642 w_k1_AXP_s010
                                 + 0.880452 w_k1_DIS_s010
                                 + 0.904468 w_k1_FDX_s010
                                 + 1.18812 w_k1_IBM_s010
                                 + 1.06144 w_k1_INTC_s010
                                 + 1.01815 w_k1_KO_s010 + 1.1009 w_k1_MAT_s010
                                 + 0.930578 w_k1_MSFT_s010
                                 + 0.87717 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.925731 w_k1_ADM_s010
                                 + 1.13807 w_k1_AXP_s010
                                 + 0.682369 w_k1_DIS_s010
                                 + 0.991781 w_k1_FDX_s010
                                 + 0.732758 w_k1_IBM_s010
                                 + 0.91667 w_k1_INTC_s010
                                 + 0.732281 w_k1_KO_s010
                                 + 0.810265 w_k1_MAT_s010
                                 + 0.893835 w_k1_MSFT_s010
                                 + 0.882507 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.06753 w_k1_ADM_s010
                                 + 1.0468 w_k1_AXP_s010 + 1.24316 w_k1_DIS_s010
                                 + 1.07451 w_k1_FDX_s010
                                 + 1.49272 w_k1_IBM_s010
                                 + 1.07989 w_k1_INTC_s010
                                 + 1.21575 w_k1_KO_s010 + 1.05533 w_k1_MAT_s010
                                 + 0.979523 w_k1_MSFT_s010
                                 + 1.11624 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.28873 w_k1_ADM_s011
                                 + 1.36267 w_k1_AXP_s011
                                 + 1.17314 w_k1_DIS_s011
                                 + 1.09465 w_k1_FDX_s011
                                 + 1.04559 w_k1_IBM_s011
                                 + 1.04324 w_k1_INTC_s011 + 1.1893 w_k1_KO_s011
                                 + 1.02128 w_k1_MAT_s011
                                 + 1.02534 w_k1_MSFT_s011
                                 + 1.05521 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.19363 w_k1_ADM_s011
                                 + 1.65256 w_k1_AXP_s011 + 1.7328 w_k1_DIS_s011
                                 + 1.46781 w_k1_FDX_s011 + 1.2779 w_k1_IBM_s011
                                 + 1.35488 w_k1_INTC_s011
                                 + 1.33942 w_k1_KO_s011 + 1.49335 w_k1_MAT_s011
                                 + 1.47845 w_k1_MSFT_s011
                                 + 1.10884 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.00117 w_k1_ADM_s011
                                 + 0.976112 w_k1_AXP_s011
                                 + 1.05756 w_k1_DIS_s011
                                 + 0.934329 w_k1_FDX_s011
                                 + 1.1518 w_k1_IBM_s011
                                 + 0.928069 w_k1_INTC_s011
                                 + 1.17324 w_k1_KO_s011 + 1.30189 w_k1_MAT_s011
                                 + 1.08678 w_k1_MSFT_s011
                                 + 1.00424 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.38852 w_k1_ADM_s011
                                 + 1.19976 w_k1_AXP_s011
                                 + 1.18884 w_k1_DIS_s011
                                 + 1.16758 w_k1_FDX_s011
                                 + 0.788784 w_k1_IBM_s011
                                 + 1.33433 w_k1_INTC_s011
                                 + 1.04773 w_k1_KO_s011 + 1.01354 w_k1_MAT_s011
                                 + 1.17235 w_k1_MSFT_s011
                                 + 0.992925 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.851674 w_k1_ADM_s011
                                 + 0.757942 w_k1_AXP_s011
                                 + 0.696149 w_k1_DIS_s011
                                 + 0.830775 w_k1_FDX_s011
                                 + 0.712472 w_k1_IBM_s011
                                 + 0.876377 w_k1_INTC_s011
                                 + 0.83154 w_k1_KO_s011
                                 + 0.575229 w_k1_MAT_s011
                                 + 0.849733 w_k1_MSFT_s011
                                 + 0.921771 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.33253 w_k1_ADM_s011
                                 + 1.14379 w_k1_AXP_s011 + 1.499 w_k1_DIS_s011
                                 + 1.06404 w_k1_FDX_s011
                                 + 1.39068 w_k1_IBM_s011
                                 + 0.890334 w_k1_INTC_s011
                                 + 1.44109 w_k1_KO_s011 + 1.69959 w_k1_MAT_s011
                                 + 1.21699 w_k1_MSFT_s011
                                 + 1.21631 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.814144 w_k1_ADM_s011
                                 + 1.29291 w_k1_AXP_s011
                                 + 1.15504 w_k1_DIS_s011
                                 + 0.940219 w_k1_FDX_s011
                                 + 1.50658 w_k1_IBM_s011
                                 + 1.05256 w_k1_INTC_s011
                                 + 1.26939 w_k1_KO_s011 + 2.1289 w_k1_MAT_s011
                                 + 0.950776 w_k1_MSFT_s011
                                 + 0.982512 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.911638 w_k1_ADM_s011
                                 + 0.908195 w_k1_AXP_s011
                                 + 0.813214 w_k1_DIS_s011
                                 + 0.83536 w_k1_FDX_s011
                                 + 0.958949 w_k1_IBM_s011
                                 + 0.951283 w_k1_INTC_s011
                                 + 0.923784 w_k1_KO_s011
                                 + 0.952408 w_k1_MAT_s011
                                 + 0.866876 w_k1_MSFT_s011
                                 + 0.867413 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.58746 w_k1_ADM_s012
                                 + 1.46667 w_k1_AXP_s012
                                 + 1.79598 w_k1_DIS_s012
                                 + 1.55639 w_k1_FDX_s012
                                 + 1.08092 w_k1_IBM_s012
                                 + 1.25362 w_k1_INTC_s012
                                 + 1.38956 w_k1_KO_s012
                                 + 0.988907 w_k1_MAT_s012
                                 + 1.41564 w_k1_MSFT_s012
                                 + 1.33544 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.09387 w_k1_ADM_s012
                                 + 1.30123 w_k1_AXP_s012
                                 + 1.18293 w_k1_DIS_s012
                                 + 1.09516 w_k1_FDX_s012
                                 + 1.49628 w_k1_IBM_s012
                                 + 0.91467 w_k1_INTC_s012
                                 + 1.02217 w_k1_KO_s012 + 1.4754 w_k1_MAT_s012
                                 + 1.0228 w_k1_MSFT_s012 + 1.09017 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.15948 w_k1_ADM_s012
                                 + 1.27976 w_k1_AXP_s012
                                 + 1.03299 w_k1_DIS_s012
                                 + 1.23356 w_k1_FDX_s012
                                 + 0.906529 w_k1_IBM_s012
                                 + 1.07302 w_k1_INTC_s012
                                 + 0.773053 w_k1_KO_s012
                                 + 0.686421 w_k1_MAT_s012
                                 + 1.00091 w_k1_MSFT_s012
                                 + 0.993636 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.983037 w_k1_ADM_s012
                                 + 0.871691 w_k1_AXP_s012
                                 + 0.994455 w_k1_DIS_s012
                                 + 0.872812 w_k1_FDX_s012
                                 + 1.21619 w_k1_IBM_s012
                                 + 0.78565 w_k1_INTC_s012
                                 + 1.04266 w_k1_KO_s012 + 1.04153 w_k1_MAT_s012
                                 + 0.988968 w_k1_MSFT_s012
                                 + 0.964245 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.922948 w_k1_ADM_s012
                                 + 0.720203 w_k1_AXP_s012
                                 + 0.597724 w_k1_DIS_s012
                                 + 0.829217 w_k1_FDX_s012
                                 + 0.749324 w_k1_IBM_s012
                                 + 0.782774 w_k1_INTC_s012
                                 + 0.694749 w_k1_KO_s012
                                 + 0.567321 w_k1_MAT_s012
                                 + 0.802282 w_k1_MSFT_s012
                                 + 0.813211 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.20085 w_k1_ADM_s012
                                 + 1.37481 w_k1_AXP_s012
                                 + 1.55063 w_k1_DIS_s012
                                 + 1.21767 w_k1_FDX_s012
                                 + 1.33585 w_k1_IBM_s012
                                 + 1.1586 w_k1_INTC_s012 + 1.36083 w_k1_KO_s012
                                 + 1.56439 w_k1_MAT_s012
                                 + 1.33436 w_k1_MSFT_s012
                                 + 1.25233 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.98563 w_k1_ADM_s012
                                 + 1.04582 w_k1_AXP_s012
                                 + 0.962669 w_k1_DIS_s012
                                 + 1.03465 w_k1_FDX_s012
                                 + 0.766175 w_k1_IBM_s012
                                 + 0.869042 w_k1_INTC_s012
                                 + 0.902777 w_k1_KO_s012
                                 + 0.731231 w_k1_MAT_s012
                                 + 0.919554 w_k1_MSFT_s012
                                 + 1.02978 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.04315 w_k1_ADM_s012
                                 + 0.930257 w_k1_AXP_s012
                                 + 1.33397 w_k1_DIS_s012
                                 + 1.20914 w_k1_FDX_s012
                                 + 0.781328 w_k1_IBM_s012
                                 + 0.938786 w_k1_INTC_s012
                                 + 1.0171 w_k1_KO_s012 + 0.612981 w_k1_MAT_s012
                                 + 1.17514 w_k1_MSFT_s012
                                 + 1.09634 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.09257 w_k1_ADM_s013
                                 + 1.12407 w_k1_AXP_s013
                                 + 1.13389 w_k1_DIS_s013
                                 + 1.01453 w_k1_FDX_s013 + 1.439 w_k1_IBM_s013
                                 + 0.926417 w_k1_INTC_s013
                                 + 1.00942 w_k1_KO_s013 + 1.17771 w_k1_MAT_s013
                                 + 0.848609 w_k1_MSFT_s013
                                 + 1.08025 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.662489 w_k1_ADM_s013
                                 + 0.661901 w_k1_AXP_s013
                                 + 0.695326 w_k1_DIS_s013
                                 + 0.810855 w_k1_FDX_s013
                                 + 0.592232 w_k1_IBM_s013
                                 + 0.940873 w_k1_INTC_s013
                                 + 0.685444 w_k1_KO_s013
                                 + 0.636172 w_k1_MAT_s013
                                 + 0.806239 w_k1_MSFT_s013
                                 + 0.747855 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.05244 w_k1_ADM_s013
                                 + 0.897308 w_k1_AXP_s013
                                 + 0.860417 w_k1_DIS_s013
                                 + 0.956254 w_k1_FDX_s013
                                 + 0.982202 w_k1_IBM_s013
                                 + 1.00847 w_k1_INTC_s013
                                 + 0.948032 w_k1_KO_s013
                                 + 0.868021 w_k1_MAT_s013
                                 + 1.02766 w_k1_MSFT_s013
                                 + 0.988623 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.780431 w_k1_ADM_s013
                                 + 0.653644 w_k1_AXP_s013
                                 + 0.785541 w_k1_DIS_s013
                                 + 0.760172 w_k1_FDX_s013
                                 + 0.895301 w_k1_IBM_s013
                                 + 0.933279 w_k1_INTC_s013
                                 + 0.848968 w_k1_KO_s013
                                 + 0.657959 w_k1_MAT_s013
                                 + 0.851761 w_k1_MSFT_s013
                                 + 0.851143 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15907 w_k1_ADM_s013
                                 + 0.988838 w_k1_AXP_s013
                                 + 1.11812 w_k1_DIS_s013
                                 + 1.04823 w_k1_FDX_s013
                                 + 0.829282 w_k1_IBM_s013
                                 + 1.09456 w_k1_INTC_s013
                                 + 0.959387 w_k1_KO_s013
                                 + 1.06218 w_k1_MAT_s013
                                 + 1.01405 w_k1_MSFT_s013
                                 + 0.974162 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.33531 w_k1_ADM_s013
                                 + 1.64419 w_k1_AXP_s013 + 1.6795 w_k1_DIS_s013
                                 + 1.39552 w_k1_FDX_s013
                                 + 1.57974 w_k1_IBM_s013
                                 + 1.18127 w_k1_INTC_s013
                                 + 1.48752 w_k1_KO_s013 + 1.93076 w_k1_MAT_s013
                                 + 1.26502 w_k1_MSFT_s013
                                 + 1.42033 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.3804 w_k1_ADM_s013
                                 + 1.41442 w_k1_AXP_s013
                                 + 1.48444 w_k1_DIS_s013
                                 + 1.28262 w_k1_FDX_s013
                                 + 1.11937 w_k1_IBM_s013
                                 + 1.1165 w_k1_INTC_s013 + 1.22437 w_k1_KO_s013
                                 + 1.61241 w_k1_MAT_s013
                                 + 1.20627 w_k1_MSFT_s013
                                 + 1.20856 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.16875 w_k1_ADM_s013
                                 + 1.08112 w_k1_AXP_s013
                                 + 1.20582 w_k1_DIS_s013
                                 + 1.12204 w_k1_FDX_s013
                                 + 1.00447 w_k1_IBM_s013
                                 + 1.11974 w_k1_INTC_s013
                                 + 1.13138 w_k1_KO_s013 + 1.28889 w_k1_MAT_s013
                                 + 1.19342 w_k1_MSFT_s013
                                 + 1.08474 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.0755 w_k1_ADM_s014
                                 + 0.866328 w_k1_AXP_s014
                                 + 1.14668 w_k1_DIS_s014
                                 + 1.07823 w_k1_FDX_s014
                                 + 0.70742 w_k1_IBM_s014
                                 + 1.11826 w_k1_INTC_s014
                                 + 0.958246 w_k1_KO_s014
                                 + 0.67196 w_k1_MAT_s014
                                 + 1.11553 w_k1_MSFT_s014
                                 + 0.980239 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.755989 w_k1_ADM_s014
                                 + 0.709643 w_k1_AXP_s014
                                 + 0.725081 w_k1_DIS_s014
                                 + 0.739739 w_k1_FDX_s014
                                 + 0.745167 w_k1_IBM_s014
                                 + 0.934539 w_k1_INTC_s014
                                 + 0.773249 w_k1_KO_s014
                                 + 0.791685 w_k1_MAT_s014
                                 + 0.872831 w_k1_MSFT_s014
                                 + 0.829192 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.07858 w_k1_ADM_s014
                                 + 0.964297 w_k1_AXP_s014
                                 + 1.15465 w_k1_DIS_s014
                                 + 1.02272 w_k1_FDX_s014
                                 + 1.03002 w_k1_IBM_s014
                                 + 1.05488 w_k1_INTC_s014
                                 + 1.00174 w_k1_KO_s014 + 1.06406 w_k1_MAT_s014
                                 + 1.04319 w_k1_MSFT_s014
                                 + 1.04539 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.00137 w_k1_ADM_s014
                                 + 0.75873 w_k1_AXP_s014
                                 + 0.730628 w_k1_DIS_s014
                                 + 0.749443 w_k1_FDX_s014
                                 + 0.930914 w_k1_IBM_s014
                                 + 0.873833 w_k1_INTC_s014
                                 + 0.82821 w_k1_KO_s014
                                 + 0.834133 w_k1_MAT_s014
                                 + 0.836206 w_k1_MSFT_s014
                                 + 0.937612 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.24068 w_k1_ADM_s014
                                 + 1.24261 w_k1_AXP_s014
                                 + 1.27479 w_k1_DIS_s014
                                 + 1.07072 w_k1_FDX_s014
                                 + 1.53742 w_k1_IBM_s014
                                 + 1.09232 w_k1_INTC_s014
                                 + 1.36339 w_k1_KO_s014 + 1.34966 w_k1_MAT_s014
                                 + 1.24023 w_k1_MSFT_s014
                                 + 1.25533 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.06426 w_k1_ADM_s014
                                 + 0.91764 w_k1_AXP_s014
                                 + 0.907954 w_k1_DIS_s014
                                 + 0.838894 w_k1_FDX_s014
                                 + 1.20068 w_k1_IBM_s014
                                 + 0.918339 w_k1_INTC_s014
                                 + 1.14405 w_k1_KO_s014 + 1.04456 w_k1_MAT_s014
                                 + 0.893416 w_k1_MSFT_s014
                                 + 1.01612 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.23981 w_k1_ADM_s014
                                 + 1.34301 w_k1_AXP_s014
                                 + 1.24999 w_k1_DIS_s014
                                 + 1.18232 w_k1_FDX_s014
                                 + 0.974452 w_k1_IBM_s014
                                 + 0.954757 w_k1_INTC_s014
                                 + 1.09956 w_k1_KO_s014 + 1.029 w_k1_MAT_s014
                                 + 1.03625 w_k1_MSFT_s014
                                 + 1.17747 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.45737 w_k1_ADM_s014
                                 + 1.52868 w_k1_AXP_s014
                                 + 1.79208 w_k1_DIS_s014
                                 + 1.43558 w_k1_FDX_s014
                                 + 1.18664 w_k1_IBM_s014
                                 + 1.19144 w_k1_INTC_s014
                                 + 1.35841 w_k1_KO_s014 + 1.216 w_k1_MAT_s014
                                 + 1.25195 w_k1_MSFT_s014
                                 + 1.18805 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.20421 w_k1_ADM_s015
                                 + 1.42085 w_k1_AXP_s015
                                 + 1.61507 w_k1_DIS_s015
                                 + 1.31036 w_k1_FDX_s015
                                 + 1.45268 w_k1_IBM_s015
                                 + 1.12712 w_k1_INTC_s015
                                 + 1.72565 w_k1_KO_s015 + 1.35732 w_k1_MAT_s015
                                 + 1.10963 w_k1_MSFT_s015
                                 + 1.25584 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.03622 w_k1_ADM_s015
                                 + 0.844578 w_k1_AXP_s015
                                 + 1.09923 w_k1_DIS_s015
                                 + 0.907888 w_k1_FDX_s015
                                 + 1.09563 w_k1_IBM_s015
                                 + 1.18201 w_k1_INTC_s015
                                 + 1.17692 w_k1_KO_s015
                                 + 0.641591 w_k1_MAT_s015
                                 + 0.930555 w_k1_MSFT_s015
                                 + 1.02805 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.05219 w_k1_ADM_s015
                                 + 0.795293 w_k1_AXP_s015
                                 + 0.71136 w_k1_DIS_s015
                                 + 0.800936 w_k1_FDX_s015
                                 + 0.782583 w_k1_IBM_s015
                                 + 0.914096 w_k1_INTC_s015
                                 + 0.743913 w_k1_KO_s015
                                 + 0.572843 w_k1_MAT_s015
                                 + 0.773753 w_k1_MSFT_s015
                                 + 0.883515 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.14996 w_k1_ADM_s015
                                 + 1.18367 w_k1_AXP_s015
                                 + 0.992358 w_k1_DIS_s015
                                 + 0.921195 w_k1_FDX_s015
                                 + 1.31509 w_k1_IBM_s015
                                 + 1.06873 w_k1_INTC_s015
                                 + 1.34799 w_k1_KO_s015 + 1.27913 w_k1_MAT_s015
                                 + 1.04004 w_k1_MSFT_s015
                                 + 1.03828 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.18694 w_k1_ADM_s015
                                 + 1.12582 w_k1_AXP_s015
                                 + 1.36617 w_k1_DIS_s015
                                 + 1.20936 w_k1_FDX_s015
                                 + 0.953156 w_k1_IBM_s015
                                 + 1.2727 w_k1_INTC_s015 + 1.23899 w_k1_KO_s015
                                 + 0.786819 w_k1_MAT_s015
                                 + 1.10989 w_k1_MSFT_s015
                                 + 1.04925 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.17709 w_k1_ADM_s015
                                 + 0.815356 w_k1_AXP_s015
                                 + 0.974783 w_k1_DIS_s015
                                 + 0.921613 w_k1_FDX_s015
                                 + 0.719251 w_k1_IBM_s015
                                 + 1.02605 w_k1_INTC_s015
                                 + 0.990219 w_k1_KO_s015
                                 + 0.479742 w_k1_MAT_s015
                                 + 0.983512 w_k1_MSFT_s015
                                 + 0.94355 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.775867 w_k1_ADM_s015
                                 + 0.653698 w_k1_AXP_s015
                                 + 0.691313 w_k1_DIS_s015
                                 + 0.715681 w_k1_FDX_s015
                                 + 1.00589 w_k1_IBM_s015
                                 + 0.708244 w_k1_INTC_s015
                                 + 0.873988 w_k1_KO_s015
                                 + 0.790074 w_k1_MAT_s015
                                 + 0.736006 w_k1_MSFT_s015
                                 + 0.801211 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.23723 w_k1_ADM_s015
                                 + 1.17775 w_k1_AXP_s015
                                 + 0.930081 w_k1_DIS_s015
                                 + 1.03659 w_k1_FDX_s015
                                 + 0.852201 w_k1_IBM_s015
                                 + 0.875593 w_k1_INTC_s015
                                 + 0.903928 w_k1_KO_s015
                                 + 0.937961 w_k1_MAT_s015
                                 + 0.880894 w_k1_MSFT_s015
                                 + 0.970604 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.327 w_k1_ADM_s016
                                 + 1.17917 w_k1_AXP_s016
                                 + 1.96389 w_k1_DIS_s016
                                 + 1.36998 w_k1_FDX_s016
                                 + 0.969265 w_k1_IBM_s016
                                 + 1.01687 w_k1_INTC_s016
                                 + 1.53511 w_k1_KO_s016 + 1.14218 w_k1_MAT_s016
                                 + 1.38199 w_k1_MSFT_s016
                                 + 1.27314 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.804948 w_k1_ADM_s016
                                 + 0.6919 w_k1_AXP_s016 + 0.65733 w_k1_DIS_s016
                                 + 0.830575 w_k1_FDX_s016
                                 + 0.738626 w_k1_IBM_s016
                                 + 0.949195 w_k1_INTC_s016
                                 + 0.787208 w_k1_KO_s016
                                 + 0.712081 w_k1_MAT_s016
                                 + 0.918155 w_k1_MSFT_s016
                                 + 0.950356 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.05162 w_k1_ADM_s016
                                 + 0.922426 w_k1_AXP_s016
                                 + 1.10926 w_k1_DIS_s016
                                 + 1.04793 w_k1_FDX_s016
                                 + 0.887094 w_k1_IBM_s016
                                 + 0.981234 w_k1_INTC_s016
                                 + 1.06864 w_k1_KO_s016 + 0.9249 w_k1_MAT_s016
                                 + 1.28336 w_k1_MSFT_s016
                                 + 1.16857 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.20007 w_k1_ADM_s016
                                 + 1.20515 w_k1_AXP_s016
                                 + 1.76641 w_k1_DIS_s016
                                 + 1.11661 w_k1_FDX_s016
                                 + 1.68352 w_k1_IBM_s016
                                 + 1.02521 w_k1_INTC_s016
                                 + 1.61659 w_k1_KO_s016 + 1.9853 w_k1_MAT_s016
                                 + 1.21004 w_k1_MSFT_s016
                                 + 1.24733 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.759414 w_k1_ADM_s016
                                 + 0.780805 w_k1_AXP_s016
                                 + 0.994806 w_k1_DIS_s016
                                 + 0.892601 w_k1_FDX_s016
                                 + 1.18197 w_k1_IBM_s016
                                 + 1.01218 w_k1_INTC_s016
                                 + 1.12344 w_k1_KO_s016 + 1.00008 w_k1_MAT_s016
                                 + 0.957651 w_k1_MSFT_s016
                                 + 1.1788 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.694938 w_k1_ADM_s016
                                 + 0.568028 w_k1_AXP_s016
                                 + 0.775497 w_k1_DIS_s016
                                 + 0.744718 w_k1_FDX_s016
                                 + 0.863222 w_k1_IBM_s016
                                 + 0.808434 w_k1_INTC_s016
                                 + 0.999651 w_k1_KO_s016
                                 + 0.866358 w_k1_MAT_s016
                                 + 0.832868 w_k1_MSFT_s016
                                 + 1.04079 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.895234 w_k1_ADM_s016
                                 + 1.07236 w_k1_AXP_s016
                                 + 0.829665 w_k1_DIS_s016
                                 + 1.02667 w_k1_FDX_s016
                                 + 0.85438 w_k1_IBM_s016
                                 + 1.06133 w_k1_INTC_s016
                                 + 0.891572 w_k1_KO_s016
                                 + 1.03409 w_k1_MAT_s016
                                 + 1.02623 w_k1_MSFT_s016
                                 + 1.02226 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.31441 w_k1_ADM_s016
                                 + 1.39681 w_k1_AXP_s016
                                 + 1.43078 w_k1_DIS_s016
                                 + 1.27709 w_k1_FDX_s016
                                 + 1.14311 w_k1_IBM_s016
                                 + 1.46974 w_k1_INTC_s016 + 1.2827 w_k1_KO_s016
                                 + 1.20501 w_k1_MAT_s016
                                 + 1.31096 w_k1_MSFT_s016 + 1.2371 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.03075 w_k1_ADM_s017
                                 + 0.96346 w_k1_AXP_s017
                                 + 0.825983 w_k1_DIS_s017
                                 + 0.877079 w_k1_FDX_s017
                                 + 1.00423 w_k1_IBM_s017
                                 + 0.81777 w_k1_INTC_s017
                                 + 1.01421 w_k1_KO_s017
                                 + 0.976587 w_k1_MAT_s017
                                 + 0.968499 w_k1_MSFT_s017
                                 + 1.02418 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.04368 w_k1_ADM_s017
                                 + 0.891691 w_k1_AXP_s017
                                 + 0.946811 w_k1_DIS_s017
                                 + 0.98951 w_k1_FDX_s017
                                 + 0.622083 w_k1_IBM_s017
                                 + 0.974017 w_k1_INTC_s017
                                 + 0.903087 w_k1_KO_s017
                                 + 0.661164 w_k1_MAT_s017
                                 + 0.944899 w_k1_MSFT_s017
                                 + 0.993906 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.690597 w_k1_ADM_s017
                                 + 0.830753 w_k1_AXP_s017
                                 + 0.759172 w_k1_DIS_s017
                                 + 0.771375 w_k1_FDX_s017
                                 + 0.715076 w_k1_IBM_s017
                                 + 0.840654 w_k1_INTC_s017
                                 + 0.802331 w_k1_KO_s017
                                 + 0.892228 w_k1_MAT_s017
                                 + 0.82645 w_k1_MSFT_s017
                                 + 0.924594 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.15995 w_k1_ADM_s017
                                 + 1.3341 w_k1_AXP_s017 + 1.13815 w_k1_DIS_s017
                                 + 1.13884 w_k1_FDX_s017 + 1.0414 w_k1_IBM_s017
                                 + 1.11133 w_k1_INTC_s017
                                 + 1.21945 w_k1_KO_s017 + 0.87387 w_k1_MAT_s017
                                 + 1.15391 w_k1_MSFT_s017
                                 + 1.07474 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.02369 w_k1_ADM_s017
                                 + 0.941426 w_k1_AXP_s017
                                 + 0.855272 w_k1_DIS_s017
                                 + 0.999301 w_k1_FDX_s017
                                 + 0.902781 w_k1_IBM_s017
                                 + 1.03124 w_k1_INTC_s017
                                 + 0.996993 w_k1_KO_s017
                                 + 1.06198 w_k1_MAT_s017
                                 + 0.894585 w_k1_MSFT_s017 + 1.05 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.897422 w_k1_ADM_s017
                                 + 0.954413 w_k1_AXP_s017
                                 + 1.13533 w_k1_DIS_s017
                                 + 0.991243 w_k1_FDX_s017
                                 + 1.32544 w_k1_IBM_s017
                                 + 1.05895 w_k1_INTC_s017
                                 + 1.44419 w_k1_KO_s017 + 1.14466 w_k1_MAT_s017
                                 + 1.1283 w_k1_MSFT_s017 + 1.15729 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.921623 w_k1_ADM_s017
                                 + 0.7933 w_k1_AXP_s017
                                 + 0.734764 w_k1_DIS_s017
                                 + 0.856507 w_k1_FDX_s017
                                 + 0.874223 w_k1_IBM_s017
                                 + 0.851541 w_k1_INTC_s017
                                 + 0.885841 w_k1_KO_s017
                                 + 0.70671 w_k1_MAT_s017
                                 + 0.966044 w_k1_MSFT_s017
                                 + 0.880346 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.48041 w_k1_ADM_s017
                                 + 1.39755 w_k1_AXP_s017
                                 + 1.77106 w_k1_DIS_s017
                                 + 1.32436 w_k1_FDX_s017
                                 + 1.25425 w_k1_IBM_s017
                                 + 1.01042 w_k1_INTC_s017 + 1.2902 w_k1_KO_s017
                                 + 1.72998 w_k1_MAT_s017
                                 + 1.28364 w_k1_MSFT_s017
                                 + 1.36137 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.667832 w_k1_ADM_s100
                                 + 0.67147 w_k1_AXP_s100
                                 + 0.566565 w_k1_DIS_s100
                                 + 0.735717 w_k1_FDX_s100
                                 + 0.732565 w_k1_IBM_s100
                                 + 0.93054 w_k1_INTC_s100
                                 + 0.665816 w_k1_KO_s100
                                 + 0.659103 w_k1_MAT_s100
                                 + 0.78727 w_k1_MSFT_s100
                                 + 0.836539 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.21112 w_k1_ADM_s100
                                 + 1.53436 w_k1_AXP_s100
                                 + 1.59588 w_k1_DIS_s100
                                 + 1.39933 w_k1_FDX_s100
                                 + 1.32491 w_k1_IBM_s100
                                 + 1.3267 w_k1_INTC_s100 + 1.3853 w_k1_KO_s100
                                 + 1.60429 w_k1_MAT_s100
                                 + 1.38815 w_k1_MSFT_s100
                                 + 1.29355 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.04043 w_k1_ADM_s100
                                 + 0.891346 w_k1_AXP_s100
                                 + 1.00688 w_k1_DIS_s100
                                 + 1.05194 w_k1_FDX_s100
                                 + 1.10514 w_k1_IBM_s100
                                 + 0.985234 w_k1_INTC_s100
                                 + 1.01717 w_k1_KO_s100 + 1.11553 w_k1_MAT_s100
                                 + 0.972109 w_k1_MSFT_s100
                                 + 1.02643 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.773117 w_k1_ADM_s100
                                 + 0.961035 w_k1_AXP_s100
                                 + 0.838895 w_k1_DIS_s100
                                 + 0.849064 w_k1_FDX_s100
                                 + 1.58816 w_k1_IBM_s100
                                 + 1.03178 w_k1_INTC_s100
                                 + 1.32493 w_k1_KO_s100 + 1.512 w_k1_MAT_s100
                                 + 0.979378 w_k1_MSFT_s100
                                 + 0.927942 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.983684 w_k1_ADM_s100
                                 + 1.19839 w_k1_AXP_s100
                                 + 1.09103 w_k1_DIS_s100
                                 + 1.00319 w_k1_FDX_s100
                                 + 1.31141 w_k1_IBM_s100
                                 + 1.07289 w_k1_INTC_s100
                                 + 1.23127 w_k1_KO_s100 + 1.75077 w_k1_MAT_s100
                                 + 0.997387 w_k1_MSFT_s100
                                 + 1.27972 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.867554 w_k1_ADM_s100
                                 + 0.883085 w_k1_AXP_s100
                                 + 0.86964 w_k1_DIS_s100
                                 + 0.96404 w_k1_FDX_s100
                                 + 0.719828 w_k1_IBM_s100
                                 + 1.11285 w_k1_INTC_s100
                                 + 0.905654 w_k1_KO_s100
                                 + 1.09327 w_k1_MAT_s100
                                 + 1.06957 w_k1_MSFT_s100 + 1.0654 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.14034 w_k1_ADM_s100
                                 + 1.03728 w_k1_AXP_s100
                                 + 1.20405 w_k1_DIS_s100
                                 + 1.30801 w_k1_FDX_s100
                                 + 0.799266 w_k1_IBM_s100
                                 + 1.26852 w_k1_INTC_s100
                                 + 0.98704 w_k1_KO_s100
                                 + 0.896975 w_k1_MAT_s100
                                 + 1.42072 w_k1_MSFT_s100
                                 + 1.10218 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.871112 w_k1_ADM_s100
                                 + 0.728353 w_k1_AXP_s100
                                 + 0.781695 w_k1_DIS_s100
                                 + 0.791058 w_k1_FDX_s100
                                 + 1.09995 w_k1_IBM_s100
                                 + 0.943324 w_k1_INTC_s100
                                 + 1.10149 w_k1_KO_s100
                                 + 0.850368 w_k1_MAT_s100
                                 + 0.905615 w_k1_MSFT_s100
                                 + 0.955746 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.983559 w_k1_ADM_s101
                                 + 1.34511 w_k1_AXP_s101
                                 + 1.22304 w_k1_DIS_s101 + 1.1464 w_k1_FDX_s101
                                 + 1.23382 w_k1_IBM_s101
                                 + 1.01025 w_k1_INTC_s101
                                 + 1.20324 w_k1_KO_s101 + 2.0251 w_k1_MAT_s101
                                 + 1.01116 w_k1_MSFT_s101
                                 + 1.14435 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.1672 w_k1_ADM_s101
                                 + 1.54887 w_k1_AXP_s101
                                 + 1.93021 w_k1_DIS_s101
                                 + 1.52899 w_k1_FDX_s101
                                 + 1.34145 w_k1_IBM_s101
                                 + 1.48669 w_k1_INTC_s101
                                 + 1.58967 w_k1_KO_s101 + 1.87403 w_k1_MAT_s101
                                 + 1.46896 w_k1_MSFT_s101
                                 + 1.21882 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.618981 w_k1_ADM_s101
                                 + 0.66058 w_k1_AXP_s101
                                 + 0.636829 w_k1_DIS_s101
                                 + 0.745147 w_k1_FDX_s101
                                 + 0.934422 w_k1_IBM_s101
                                 + 0.820267 w_k1_INTC_s101
                                 + 0.868448 w_k1_KO_s101
                                 + 1.24247 w_k1_MAT_s101
                                 + 0.813266 w_k1_MSFT_s101
                                 + 0.823777 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.804745 w_k1_ADM_s101
                                 + 0.897755 w_k1_AXP_s101
                                 + 0.663142 w_k1_DIS_s101
                                 + 0.887153 w_k1_FDX_s101
                                 + 0.672337 w_k1_IBM_s101
                                 + 0.917694 w_k1_INTC_s101
                                 + 0.763917 w_k1_KO_s101
                                 + 0.910198 w_k1_MAT_s101
                                 + 0.844422 w_k1_MSFT_s101
                                 + 0.81222 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.17615 w_k1_ADM_s101
                                 + 1.15338 w_k1_AXP_s101
                                 + 1.32026 w_k1_DIS_s101 + 1.3933 w_k1_FDX_s101
                                 + 0.760829 w_k1_IBM_s101
                                 + 0.997462 w_k1_INTC_s101
                                 + 1.07518 w_k1_KO_s101 + 1.14655 w_k1_MAT_s101
                                 + 1.33227 w_k1_MSFT_s101
                                 + 1.05924 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.02873 w_k1_ADM_s101
                                 + 1.0611 w_k1_AXP_s101
                                 + 0.930173 w_k1_DIS_s101
                                 + 1.08255 w_k1_FDX_s101
                                 + 0.838184 w_k1_IBM_s101
                                 + 0.966999 w_k1_INTC_s101
                                 + 0.964931 w_k1_KO_s101
                                 + 1.05974 w_k1_MAT_s101
                                 + 1.04918 w_k1_MSFT_s101
                                 + 0.961919 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.934546 w_k1_ADM_s101
                                 + 1.37288 w_k1_AXP_s101
                                 + 1.18843 w_k1_DIS_s101
                                 + 1.14417 w_k1_FDX_s101
                                 + 1.03405 w_k1_IBM_s101
                                 + 1.12561 w_k1_INTC_s101
                                 + 1.10535 w_k1_KO_s101 + 1.78759 w_k1_MAT_s101
                                 + 1.05914 w_k1_MSFT_s101
                                 + 1.06025 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.04464 w_k1_ADM_s101
                                 + 1.22584 w_k1_AXP_s101 + 1.0935 w_k1_DIS_s101
                                 + 0.998741 w_k1_FDX_s101
                                 + 1.52245 w_k1_IBM_s101
                                 + 0.99209 w_k1_INTC_s101
                                 + 1.31152 w_k1_KO_s101 + 1.49714 w_k1_MAT_s101
                                 + 0.981963 w_k1_MSFT_s101
                                 + 1.06235 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.871841 w_k1_ADM_s102
                                 + 1.12255 w_k1_AXP_s102
                                 + 1.01842 w_k1_DIS_s102
                                 + 1.06257 w_k1_FDX_s102
                                 + 1.12327 w_k1_IBM_s102
                                 + 0.929159 w_k1_INTC_s102
                                 + 1.2716 w_k1_KO_s102 + 1.5683 w_k1_MAT_s102
                                 + 1.02725 w_k1_MSFT_s102
                                 + 1.07702 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.0419 w_k1_ADM_s102
                                 + 0.743315 w_k1_AXP_s102
                                 + 0.767357 w_k1_DIS_s102
                                 + 0.835208 w_k1_FDX_s102
                                 + 0.840187 w_k1_IBM_s102
                                 + 0.833821 w_k1_INTC_s102
                                 + 0.82798 w_k1_KO_s102 + 1.03803 w_k1_MAT_s102
                                 + 0.859443 w_k1_MSFT_s102
                                 + 0.937763 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.36881 w_k1_ADM_s102
                                 + 1.33383 w_k1_AXP_s102
                                 + 1.53413 w_k1_DIS_s102
                                 + 1.24824 w_k1_FDX_s102
                                 + 1.41314 w_k1_IBM_s102
                                 + 1.02185 w_k1_INTC_s102
                                 + 1.55242 w_k1_KO_s102 + 2.4042 w_k1_MAT_s102
                                 + 1.28575 w_k1_MSFT_s102
                                 + 1.28359 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.02113 w_k1_ADM_s102
                                 + 0.931063 w_k1_AXP_s102
                                 + 0.807518 w_k1_DIS_s102
                                 + 0.904371 w_k1_FDX_s102
                                 + 1.1891 w_k1_IBM_s102
                                 + 0.899568 w_k1_INTC_s102
                                 + 1.07869 w_k1_KO_s102 + 1.42595 w_k1_MAT_s102
                                 + 0.901983 w_k1_MSFT_s102
                                 + 0.996259 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.12608 w_k1_ADM_s102
                                 + 1.14245 w_k1_AXP_s102 + 1.0198 w_k1_DIS_s102
                                 + 0.954166 w_k1_FDX_s102
                                 + 1.48825 w_k1_IBM_s102
                                 + 1.20565 w_k1_INTC_s102
                                 + 1.24885 w_k1_KO_s102 + 1.69385 w_k1_MAT_s102
                                 + 1.03558 w_k1_MSFT_s102
                                 + 1.17761 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05066 w_k1_ADM_s102
                                 + 1.1673 w_k1_AXP_s102 + 1.27152 w_k1_DIS_s102
                                 + 1.22362 w_k1_FDX_s102
                                 + 0.982941 w_k1_IBM_s102
                                 + 1.24953 w_k1_INTC_s102
                                 + 1.15712 w_k1_KO_s102 + 1.11778 w_k1_MAT_s102
                                 + 1.26069 w_k1_MSFT_s102
                                 + 1.15352 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.15914 w_k1_ADM_s102
                                 + 0.806936 w_k1_AXP_s102
                                 + 0.893813 w_k1_DIS_s102
                                 + 1.00168 w_k1_FDX_s102
                                 + 0.778486 w_k1_IBM_s102
                                 + 1.08034 w_k1_INTC_s102
                                 + 1.03071 w_k1_KO_s102 + 0.72697 w_k1_MAT_s102
                                 + 0.973776 w_k1_MSFT_s102
                                 + 1.03074 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.754121 w_k1_ADM_s102
                                 + 0.657148 w_k1_AXP_s102
                                 + 0.63092 w_k1_DIS_s102
                                 + 0.731738 w_k1_FDX_s102
                                 + 0.793327 w_k1_IBM_s102
                                 + 0.659288 w_k1_INTC_s102
                                 + 0.837657 w_k1_KO_s102
                                 + 1.07963 w_k1_MAT_s102
                                 + 0.780607 w_k1_MSFT_s102
                                 + 0.878969 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.07002 w_k1_ADM_s103
                                 + 1.3357 w_k1_AXP_s103 + 1.61145 w_k1_DIS_s103
                                 + 1.53417 w_k1_FDX_s103 + 1.0286 w_k1_IBM_s103
                                 + 1.56899 w_k1_INTC_s103
                                 + 1.16328 w_k1_KO_s103
                                 + 0.976843 w_k1_MAT_s103
                                 + 1.40545 w_k1_MSFT_s103
                                 + 1.14445 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.843666 w_k1_ADM_s103
                                 + 0.718353 w_k1_AXP_s103
                                 + 0.945826 w_k1_DIS_s103
                                 + 0.851885 w_k1_FDX_s103
                                 + 0.968158 w_k1_IBM_s103
                                 + 0.826879 w_k1_INTC_s103
                                 + 0.820443 w_k1_KO_s103
                                 + 0.857819 w_k1_MAT_s103
                                 + 0.842797 w_k1_MSFT_s103
                                 + 0.975631 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.759629 w_k1_ADM_s103
                                 + 0.645772 w_k1_AXP_s103
                                 + 0.77142 w_k1_DIS_s103
                                 + 0.855708 w_k1_FDX_s103
                                 + 0.637564 w_k1_IBM_s103
                                 + 0.930779 w_k1_INTC_s103
                                 + 0.688206 w_k1_KO_s103
                                 + 0.426599 w_k1_MAT_s103
                                 + 0.966885 w_k1_MSFT_s103
                                 + 0.822582 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.812222 w_k1_ADM_s103
                                 + 0.763708 w_k1_AXP_s103
                                 + 0.647313 w_k1_DIS_s103
                                 + 0.798277 w_k1_FDX_s103
                                 + 0.765698 w_k1_IBM_s103
                                 + 0.867524 w_k1_INTC_s103
                                 + 0.724988 w_k1_KO_s103
                                 + 0.561867 w_k1_MAT_s103
                                 + 0.754072 w_k1_MSFT_s103
                                 + 0.789833 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.03899 w_k1_ADM_s103
                                 + 1.01253 w_k1_AXP_s103
                                 + 1.41652 w_k1_DIS_s103 + 1.1916 w_k1_FDX_s103
                                 + 1.51308 w_k1_IBM_s103
                                 + 1.07667 w_k1_INTC_s103
                                 + 1.22248 w_k1_KO_s103 + 1.29669 w_k1_MAT_s103
                                 + 1.12787 w_k1_MSFT_s103
                                 + 1.12629 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.930902 w_k1_ADM_s103
                                 + 1.12751 w_k1_AXP_s103 + 1.2349 w_k1_DIS_s103
                                 + 1.17018 w_k1_FDX_s103
                                 + 1.06425 w_k1_IBM_s103
                                 + 1.01036 w_k1_INTC_s103
                                 + 0.965433 w_k1_KO_s103
                                 + 1.21873 w_k1_MAT_s103
                                 + 1.01821 w_k1_MSFT_s103 + 0.9796 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.858986 w_k1_ADM_s103
                                 + 0.711148 w_k1_AXP_s103
                                 + 0.840288 w_k1_DIS_s103
                                 + 0.815126 w_k1_FDX_s103
                                 + 1.28495 w_k1_IBM_s103
                                 + 0.949982 w_k1_INTC_s103
                                 + 0.854897 w_k1_KO_s103
                                 + 0.945499 w_k1_MAT_s103
                                 + 0.925812 w_k1_MSFT_s103
                                 + 0.791133 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.03101 w_k1_ADM_s103
                                 + 0.917466 w_k1_AXP_s103
                                 + 0.997983 w_k1_DIS_s103
                                 + 1.07808 w_k1_FDX_s103
                                 + 0.837712 w_k1_IBM_s103
                                 + 1.12259 w_k1_INTC_s103
                                 + 0.988322 w_k1_KO_s103
                                 + 0.863117 w_k1_MAT_s103
                                 + 1.02293 w_k1_MSFT_s103 + 1.0575 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.03636 w_k1_ADM_s104
                                 + 0.871625 w_k1_AXP_s104
                                 + 0.778463 w_k1_DIS_s104
                                 + 0.987202 w_k1_FDX_s104
                                 + 0.781017 w_k1_IBM_s104
                                 + 0.911483 w_k1_INTC_s104
                                 + 0.885037 w_k1_KO_s104
                                 + 0.995285 w_k1_MAT_s104
                                 + 0.829806 w_k1_MSFT_s104
                                 + 0.900663 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.90708 w_k1_ADM_s104
                                 + 0.948678 w_k1_AXP_s104
                                 + 0.919345 w_k1_DIS_s104
                                 + 1.00184 w_k1_FDX_s104
                                 + 1.18225 w_k1_IBM_s104
                                 + 1.00454 w_k1_INTC_s104
                                 + 1.09668 w_k1_KO_s104 + 1.3245 w_k1_MAT_s104
                                 + 0.996492 w_k1_MSFT_s104
                                 + 0.953816 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.50469 w_k1_ADM_s104
                                 + 1.83185 w_k1_AXP_s104
                                 + 1.69924 w_k1_DIS_s104
                                 + 1.45312 w_k1_FDX_s104
                                 + 1.63509 w_k1_IBM_s104
                                 + 1.52046 w_k1_INTC_s104
                                 + 1.63932 w_k1_KO_s104 + 1.84129 w_k1_MAT_s104
                                 + 1.34723 w_k1_MSFT_s104 + 1.2468 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.829343 w_k1_ADM_s104
                                 + 0.665426 w_k1_AXP_s104
                                 + 0.646334 w_k1_DIS_s104
                                 + 0.728196 w_k1_FDX_s104
                                 + 0.759139 w_k1_IBM_s104
                                 + 0.789179 w_k1_INTC_s104
                                 + 0.794162 w_k1_KO_s104
                                 + 0.931971 w_k1_MAT_s104
                                 + 0.759056 w_k1_MSFT_s104
                                 + 0.825287 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.995158 w_k1_ADM_s104
                                 + 1.25856 w_k1_AXP_s104
                                 + 1.34104 w_k1_DIS_s104
                                 + 1.23345 w_k1_FDX_s104
                                 + 1.37831 w_k1_IBM_s104
                                 + 1.40294 w_k1_INTC_s104
                                 + 1.52685 w_k1_KO_s104 + 1.874 w_k1_MAT_s104
                                 + 1.24635 w_k1_MSFT_s104
                                 + 1.12636 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.814481 w_k1_ADM_s104
                                 + 1.1865 w_k1_AXP_s104 + 1.11586 w_k1_DIS_s104
                                 + 1.13453 w_k1_FDX_s104
                                 + 0.842778 w_k1_IBM_s104
                                 + 1.19962 w_k1_INTC_s104
                                 + 1.11326 w_k1_KO_s104 + 1.03521 w_k1_MAT_s104
                                 + 1.0684 w_k1_MSFT_s104 + 1.02723 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.07393 w_k1_ADM_s104
                                 + 0.716795 w_k1_AXP_s104
                                 + 1.18422 w_k1_DIS_s104
                                 + 1.03251 w_k1_FDX_s104
                                 + 0.960862 w_k1_IBM_s104
                                 + 0.987294 w_k1_INTC_s104
                                 + 1.25662 w_k1_KO_s104
                                 + 0.889267 w_k1_MAT_s104
                                 + 1.07408 w_k1_MSFT_s104 + 1.0203 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.43747 w_k1_ADM_s104
                                 + 1.55618 w_k1_AXP_s104
                                 + 1.53806 w_k1_DIS_s104 + 1.5057 w_k1_FDX_s104
                                 + 0.880293 w_k1_IBM_s104
                                 + 1.43493 w_k1_INTC_s104
                                 + 1.29539 w_k1_KO_s104 + 1.05735 w_k1_MAT_s104
                                 + 1.41116 w_k1_MSFT_s104
                                 + 1.13757 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.36679 w_k1_ADM_s105
                                 + 1.16701 w_k1_AXP_s105
                                 + 1.20139 w_k1_DIS_s105 + 1.1595 w_k1_FDX_s105
                                 + 1.05442 w_k1_IBM_s105
                                 + 1.23053 w_k1_INTC_s105
                                 + 1.27374 w_k1_KO_s105
                                 + 0.606424 w_k1_MAT_s105
                                 + 1.08785 w_k1_MSFT_s105
                                 + 1.04311 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.87192 w_k1_ADM_s105
                                 + 1.83967 w_k1_AXP_s105
                                 + 1.78949 w_k1_DIS_s105
                                 + 1.33269 w_k1_FDX_s105
                                 + 1.34046 w_k1_IBM_s105
                                 + 1.27931 w_k1_INTC_s105
                                 + 1.43771 w_k1_KO_s105 + 1.09919 w_k1_MAT_s105
                                 + 1.23544 w_k1_MSFT_s105
                                 + 1.36213 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.1194 w_k1_ADM_s105
                                 + 1.29212 w_k1_AXP_s105
                                 + 1.04255 w_k1_DIS_s105
                                 + 1.12687 w_k1_FDX_s105
                                 + 0.892296 w_k1_IBM_s105
                                 + 0.903996 w_k1_INTC_s105
                                 + 0.978357 w_k1_KO_s105
                                 + 0.764701 w_k1_MAT_s105
                                 + 0.845595 w_k1_MSFT_s105
                                 + 0.942868 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.23046 w_k1_ADM_s105
                                 + 1.4565 w_k1_AXP_s105 + 1.42998 w_k1_DIS_s105
                                 + 1.12986 w_k1_FDX_s105
                                 + 1.71302 w_k1_IBM_s105
                                 + 1.04139 w_k1_INTC_s105
                                 + 1.31831 w_k1_KO_s105 + 1.49552 w_k1_MAT_s105
                                 + 1.00288 w_k1_MSFT_s105
                                 + 1.20968 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.967626 w_k1_ADM_s105
                                 + 0.996745 w_k1_AXP_s105
                                 + 0.978894 w_k1_DIS_s105
                                 + 0.86594 w_k1_FDX_s105
                                 + 1.26382 w_k1_IBM_s105
                                 + 0.963764 w_k1_INTC_s105
                                 + 1.17007 w_k1_KO_s105 + 1.24281 w_k1_MAT_s105
                                 + 0.876123 w_k1_MSFT_s105
                                 + 1.0344 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.23408 w_k1_ADM_s105
                                 + 0.95426 w_k1_AXP_s105
                                 + 1.42438 w_k1_DIS_s105
                                 + 1.28078 w_k1_FDX_s105
                                 + 0.827252 w_k1_IBM_s105
                                 + 1.12826 w_k1_INTC_s105
                                 + 1.14042 w_k1_KO_s105
                                 + 0.711807 w_k1_MAT_s105
                                 + 1.35061 w_k1_MSFT_s105
                                 + 1.19403 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.07081 w_k1_ADM_s105
                                 + 0.629938 w_k1_AXP_s105
                                 + 0.72216 w_k1_DIS_s105
                                 + 0.843456 w_k1_FDX_s105
                                 + 0.65632 w_k1_IBM_s105
                                 + 0.677787 w_k1_INTC_s105
                                 + 0.755469 w_k1_KO_s105
                                 + 0.532035 w_k1_MAT_s105
                                 + 0.743113 w_k1_MSFT_s105
                                 + 0.780119 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.08013 w_k1_ADM_s105
                                 + 0.843125 w_k1_AXP_s105
                                 + 0.765399 w_k1_DIS_s105
                                 + 0.808585 w_k1_FDX_s105
                                 + 0.922905 w_k1_IBM_s105
                                 + 0.870807 w_k1_INTC_s105
                                 + 1.01888 w_k1_KO_s105
                                 + 0.773386 w_k1_MAT_s105
                                 + 0.885378 w_k1_MSFT_s105
                                 + 0.896324 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.814965 w_k1_ADM_s106
                                 + 0.847286 w_k1_AXP_s106
                                 + 0.768115 w_k1_DIS_s106
                                 + 0.882368 w_k1_FDX_s106
                                 + 0.878888 w_k1_IBM_s106
                                 + 0.953939 w_k1_INTC_s106
                                 + 0.78358 w_k1_KO_s106
                                 + 0.975379 w_k1_MAT_s106
                                 + 0.947145 w_k1_MSFT_s106
                                 + 0.8455 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.548109 w_k1_ADM_s106
                                 + 0.620404 w_k1_AXP_s106
                                 + 0.549959 w_k1_DIS_s106
                                 + 0.682315 w_k1_FDX_s106
                                 + 0.787552 w_k1_IBM_s106
                                 + 0.989707 w_k1_INTC_s106
                                 + 0.676015 w_k1_KO_s106
                                 + 0.644168 w_k1_MAT_s106
                                 + 0.869398 w_k1_MSFT_s106
                                 + 0.741862 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.20905 w_k1_ADM_s106
                                 + 1.3614 w_k1_AXP_s106 + 1.57427 w_k1_DIS_s106
                                 + 1.34532 w_k1_FDX_s106
                                 + 1.44485 w_k1_IBM_s106
                                 + 1.24482 w_k1_INTC_s106
                                 + 1.30973 w_k1_KO_s106 + 1.4349 w_k1_MAT_s106
                                 + 1.32163 w_k1_MSFT_s106
                                 + 1.19032 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.20525 w_k1_ADM_s106
                                 + 1.152 w_k1_AXP_s106 + 1.09687 w_k1_DIS_s106
                                 + 1.05271 w_k1_FDX_s106
                                 + 1.21991 w_k1_IBM_s106
                                 + 1.05758 w_k1_INTC_s106
                                 + 0.972705 w_k1_KO_s106
                                 + 1.30935 w_k1_MAT_s106
                                 + 1.11974 w_k1_MSFT_s106
                                 + 1.00099 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.12577 w_k1_ADM_s106
                                 + 1.42008 w_k1_AXP_s106
                                 + 1.11649 w_k1_DIS_s106
                                 + 1.27865 w_k1_FDX_s106
                                 + 0.946367 w_k1_IBM_s106
                                 + 1.38352 w_k1_INTC_s106
                                 + 0.95147 w_k1_KO_s106 + 1.07176 w_k1_MAT_s106
                                 + 1.23084 w_k1_MSFT_s106
                                 + 1.17678 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.840022 w_k1_ADM_s106
                                 + 1.0896 w_k1_AXP_s106
                                 + 0.802713 w_k1_DIS_s106
                                 + 0.831326 w_k1_FDX_s106
                                 + 1.50174 w_k1_IBM_s106
                                 + 0.98063 w_k1_INTC_s106
                                 + 1.01214 w_k1_KO_s106 + 1.27099 w_k1_MAT_s106
                                 + 0.904825 w_k1_MSFT_s106
                                 + 0.987983 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.34601 w_k1_ADM_s106
                                 + 1.34909 w_k1_AXP_s106
                                 + 1.50642 w_k1_DIS_s106
                                 + 1.40549 w_k1_FDX_s106
                                 + 0.801353 w_k1_IBM_s106
                                 + 1.3248 w_k1_INTC_s106 + 1.04458 w_k1_KO_s106
                                 + 1.38431 w_k1_MAT_s106
                                 + 1.46113 w_k1_MSFT_s106
                                 + 1.15728 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.961777 w_k1_ADM_s106
                                 + 0.989668 w_k1_AXP_s106
                                 + 0.938171 w_k1_DIS_s106
                                 + 1.09304 w_k1_FDX_s106
                                 + 0.80681 w_k1_IBM_s106
                                 + 1.25319 w_k1_INTC_s106
                                 + 0.80408 w_k1_KO_s106
                                 + 0.867275 w_k1_MAT_s106
                                 + 1.02987 w_k1_MSFT_s106
                                 + 0.985233 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.33768 w_k1_ADM_s107
                                 + 1.41136 w_k1_AXP_s107
                                 + 1.68976 w_k1_DIS_s107
                                 + 1.52734 w_k1_FDX_s107
                                 + 0.788512 w_k1_IBM_s107
                                 + 1.09158 w_k1_INTC_s107
                                 + 1.16492 w_k1_KO_s107
                                 + 0.914696 w_k1_MAT_s107
                                 + 1.32598 w_k1_MSFT_s107
                                 + 1.17631 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.881784 w_k1_ADM_s107
                                 + 0.925087 w_k1_AXP_s107
                                 + 0.939583 w_k1_DIS_s107
                                 + 0.82569 w_k1_FDX_s107
                                 + 1.37629 w_k1_IBM_s107
                                 + 0.976844 w_k1_INTC_s107
                                 + 1.16266 w_k1_KO_s107 + 1.40191 w_k1_MAT_s107
                                 + 0.891869 w_k1_MSFT_s107
                                 + 0.998951 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.17769 w_k1_ADM_s107
                                 + 0.870691 w_k1_AXP_s107
                                 + 1.06418 w_k1_DIS_s107
                                 + 0.996645 w_k1_FDX_s107
                                 + 1.05365 w_k1_IBM_s107
                                 + 1.01691 w_k1_INTC_s107
                                 + 1.15635 w_k1_KO_s107
                                 + 0.938752 w_k1_MAT_s107
                                 + 1.0686 w_k1_MSFT_s107
                                 + 0.946235 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.901599 w_k1_ADM_s107
                                 + 1.00862 w_k1_AXP_s107 + 1.0502 w_k1_DIS_s107
                                 + 1.03274 w_k1_FDX_s107
                                 + 0.816143 w_k1_IBM_s107
                                 + 1.0035 w_k1_INTC_s107
                                 + 0.966158 w_k1_KO_s107
                                 + 1.12989 w_k1_MAT_s107
                                 + 1.05761 w_k1_MSFT_s107
                                 + 0.981854 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.986927 w_k1_ADM_s107
                                 + 1.05772 w_k1_AXP_s107 + 1.186 w_k1_DIS_s107
                                 + 1.22733 w_k1_FDX_s107
                                 + 0.704446 w_k1_IBM_s107
                                 + 1.08222 w_k1_INTC_s107
                                 + 0.983264 w_k1_KO_s107
                                 + 0.941658 w_k1_MAT_s107
                                 + 1.05696 w_k1_MSFT_s107
                                 + 0.935876 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.696278 w_k1_ADM_s107
                                 + 0.75087 w_k1_AXP_s107
                                 + 0.783414 w_k1_DIS_s107
                                 + 0.821847 w_k1_FDX_s107
                                 + 0.919204 w_k1_IBM_s107
                                 + 0.818161 w_k1_INTC_s107
                                 + 0.97735 w_k1_KO_s107 + 1.06065 w_k1_MAT_s107
                                 + 0.900565 w_k1_MSFT_s107
                                 + 0.891779 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.794917 w_k1_ADM_s107
                                 + 0.550321 w_k1_AXP_s107
                                 + 0.689253 w_k1_DIS_s107
                                 + 0.752837 w_k1_FDX_s107
                                 + 0.737548 w_k1_IBM_s107
                                 + 0.689821 w_k1_INTC_s107
                                 + 0.777249 w_k1_KO_s107
                                 + 0.787438 w_k1_MAT_s107
                                 + 0.735906 w_k1_MSFT_s107
                                 + 0.766603 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.24232 w_k1_ADM_s107
                                 + 1.31181 w_k1_AXP_s107
                                 + 1.51187 w_k1_DIS_s107
                                 + 1.24676 w_k1_FDX_s107 + 1.1673 w_k1_IBM_s107
                                 + 1.13691 w_k1_INTC_s107
                                 + 1.47222 w_k1_KO_s107 + 1.22857 w_k1_MAT_s107
                                 + 1.114 w_k1_MSFT_s107 + 1.18383 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.09242 w_k1_ADM_s110
                                 + 0.897202 w_k1_AXP_s110
                                 + 1.19135 w_k1_DIS_s110
                                 + 1.04776 w_k1_FDX_s110
                                 + 1.03483 w_k1_IBM_s110
                                 + 1.37612 w_k1_INTC_s110
                                 + 1.19596 w_k1_KO_s110
                                 + 0.958792 w_k1_MAT_s110
                                 + 1.18376 w_k1_MSFT_s110
                                 + 0.992723 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.01037 w_k1_ADM_s110
                                 + 1.09426 w_k1_AXP_s110
                                 + 1.11824 w_k1_DIS_s110
                                 + 1.19133 w_k1_FDX_s110
                                 + 0.737394 w_k1_IBM_s110
                                 + 1.0633 w_k1_INTC_s110
                                 + 0.861931 w_k1_KO_s110
                                 + 0.877301 w_k1_MAT_s110
                                 + 1.12133 w_k1_MSFT_s110
                                 + 1.02789 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.686463 w_k1_ADM_s110
                                 + 0.759551 w_k1_AXP_s110
                                 + 0.869199 w_k1_DIS_s110
                                 + 0.750384 w_k1_FDX_s110
                                 + 1.07945 w_k1_IBM_s110
                                 + 0.795502 w_k1_INTC_s110
                                 + 0.879556 w_k1_KO_s110 + 1.1033 w_k1_MAT_s110
                                 + 0.804064 w_k1_MSFT_s110
                                 + 0.919397 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.39097 w_k1_ADM_s110
                                 + 1.00718 w_k1_AXP_s110
                                 + 1.18533 w_k1_DIS_s110
                                 + 0.948892 w_k1_FDX_s110
                                 + 1.69532 w_k1_IBM_s110
                                 + 0.753964 w_k1_INTC_s110
                                 + 1.39015 w_k1_KO_s110 + 1.67125 w_k1_MAT_s110
                                 + 1.02454 w_k1_MSFT_s110
                                 + 1.23355 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.942634 w_k1_ADM_s110
                                 + 1.44046 w_k1_AXP_s110 + 1.4116 w_k1_DIS_s110
                                 + 1.1778 w_k1_FDX_s110 + 1.29851 w_k1_IBM_s110
                                 + 1.01074 w_k1_INTC_s110 + 1.1846 w_k1_KO_s110
                                 + 1.41874 w_k1_MAT_s110
                                 + 1.01468 w_k1_MSFT_s110
                                 + 1.11939 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.69042 w_k1_ADM_s110
                                 + 1.43501 w_k1_AXP_s110
                                 + 1.79742 w_k1_DIS_s110
                                 + 1.39222 w_k1_FDX_s110
                                 + 1.07313 w_k1_IBM_s110
                                 + 1.23417 w_k1_INTC_s110
                                 + 1.20381 w_k1_KO_s110
                                 + 0.887919 w_k1_MAT_s110
                                 + 1.28704 w_k1_MSFT_s110
                                 + 1.26178 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.13661 w_k1_ADM_s110
                                 + 0.771439 w_k1_AXP_s110
                                 + 0.918179 w_k1_DIS_s110
                                 + 0.791216 w_k1_FDX_s110
                                 + 1.29525 w_k1_IBM_s110
                                 + 0.88313 w_k1_INTC_s110
                                 + 0.972423 w_k1_KO_s110
                                 + 1.06328 w_k1_MAT_s110
                                 + 0.947137 w_k1_MSFT_s110
                                 + 0.992099 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.22163 w_k1_ADM_s110
                                 + 0.778123 w_k1_AXP_s110
                                 + 0.765383 w_k1_DIS_s110
                                 + 0.843837 w_k1_FDX_s110
                                 + 0.787972 w_k1_IBM_s110
                                 + 0.890196 w_k1_INTC_s110
                                 + 0.842746 w_k1_KO_s110
                                 + 0.74567 w_k1_MAT_s110
                                 + 0.791465 w_k1_MSFT_s110
                                 + 0.954788 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.55662 w_k1_ADM_s111
                                 + 1.35509 w_k1_AXP_s111
                                 + 1.88651 w_k1_DIS_s111 + 1.4181 w_k1_FDX_s111
                                 + 0.86781 w_k1_IBM_s111
                                 + 1.30309 w_k1_INTC_s111
                                 + 1.25321 w_k1_KO_s111
                                 + 0.820267 w_k1_MAT_s111
                                 + 1.47416 w_k1_MSFT_s111
                                 + 1.29557 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.45892 w_k1_ADM_s111
                                 + 1.2421 w_k1_AXP_s111
                                 + 0.808671 w_k1_DIS_s111
                                 + 0.844587 w_k1_FDX_s111
                                 + 0.88999 w_k1_IBM_s111
                                 + 0.758249 w_k1_INTC_s111
                                 + 0.851266 w_k1_KO_s111
                                 + 1.15779 w_k1_MAT_s111
                                 + 0.736067 w_k1_MSFT_s111
                                 + 1.06061 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.75662 w_k1_ADM_s111
                                 + 1.03862 w_k1_AXP_s111
                                 + 1.07485 w_k1_DIS_s111
                                 + 1.01846 w_k1_FDX_s111
                                 + 0.643953 w_k1_IBM_s111
                                 + 0.841668 w_k1_INTC_s111
                                 + 0.717474 w_k1_KO_s111
                                 + 0.556449 w_k1_MAT_s111
                                 + 0.860232 w_k1_MSFT_s111
                                 + 0.872956 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.63018 w_k1_ADM_s111
                                 + 1.37529 w_k1_AXP_s111
                                 + 1.17777 w_k1_DIS_s111
                                 + 1.04571 w_k1_FDX_s111
                                 + 1.12667 w_k1_IBM_s111
                                 + 1.03109 w_k1_INTC_s111
                                 + 0.965218 w_k1_KO_s111
                                 + 1.08997 w_k1_MAT_s111
                                 + 0.928164 w_k1_MSFT_s111
                                 + 1.15099 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.02836 w_k1_ADM_s111
                                 + 0.703122 w_k1_AXP_s111
                                 + 0.53679 w_k1_DIS_s111
                                 + 0.695586 w_k1_FDX_s111
                                 + 0.724539 w_k1_IBM_s111
                                 + 0.596469 w_k1_INTC_s111
                                 + 0.603756 w_k1_KO_s111
                                 + 0.592802 w_k1_MAT_s111
                                 + 0.667076 w_k1_MSFT_s111
                                 + 0.798182 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.02257 w_k1_ADM_s111
                                 + 0.910691 w_k1_AXP_s111
                                 + 0.883626 w_k1_DIS_s111
                                 + 0.831936 w_k1_FDX_s111
                                 + 0.854582 w_k1_IBM_s111
                                 + 0.828357 w_k1_INTC_s111
                                 + 0.867861 w_k1_KO_s111
                                 + 0.762559 w_k1_MAT_s111
                                 + 0.933293 w_k1_MSFT_s111
                                 + 1.12059 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.17412 w_k1_ADM_s111
                                 + 1.15194 w_k1_AXP_s111
                                 + 0.818616 w_k1_DIS_s111
                                 + 0.76979 w_k1_FDX_s111
                                 + 1.39656 w_k1_IBM_s111
                                 + 1.01198 w_k1_INTC_s111
                                 + 0.89361 w_k1_KO_s111
                                 + 0.998225 w_k1_MAT_s111
                                 + 0.934805 w_k1_MSFT_s111
                                 + 0.980815 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.66775 w_k1_ADM_s111
                                 + 1.3785 w_k1_AXP_s111 + 1.38092 w_k1_DIS_s111
                                 + 1.02866 w_k1_FDX_s111
                                 + 1.69933 w_k1_IBM_s111
                                 + 1.00132 w_k1_INTC_s111
                                 + 1.25746 w_k1_KO_s111 + 1.81935 w_k1_MAT_s111
                                 + 0.96044 w_k1_MSFT_s111
                                 + 1.26654 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.04505 w_k1_ADM_s112
                                 + 1.10792 w_k1_AXP_s112
                                 + 0.793052 w_k1_DIS_s112
                                 + 0.842409 w_k1_FDX_s112
                                 + 1.00995 w_k1_IBM_s112
                                 + 0.905615 w_k1_INTC_s112
                                 + 0.790239 w_k1_KO_s112
                                 + 1.34357 w_k1_MAT_s112
                                 + 0.81754 w_k1_MSFT_s112
                                 + 0.936213 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17289 w_k1_ADM_s112
                                 + 1.42766 w_k1_AXP_s112
                                 + 1.60562 w_k1_DIS_s112
                                 + 1.19254 w_k1_FDX_s112
                                 + 1.56114 w_k1_IBM_s112
                                 + 0.992097 w_k1_INTC_s112
                                 + 1.2998 w_k1_KO_s112 + 2.22055 w_k1_MAT_s112
                                 + 1.1673 w_k1_MSFT_s112 + 1.19195 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.897364 w_k1_ADM_s112
                                 + 1.04103 w_k1_AXP_s112
                                 + 0.816043 w_k1_DIS_s112
                                 + 0.870437 w_k1_FDX_s112
                                 + 0.746142 w_k1_IBM_s112
                                 + 0.973484 w_k1_INTC_s112
                                 + 0.770906 w_k1_KO_s112 + 0.783 w_k1_MAT_s112
                                 + 0.883408 w_k1_MSFT_s112
                                 + 0.902794 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.26734 w_k1_ADM_s112
                                 + 1.39202 w_k1_AXP_s112
                                 + 1.10743 w_k1_DIS_s112
                                 + 1.18885 w_k1_FDX_s112
                                 + 0.850413 w_k1_IBM_s112
                                 + 1.23904 w_k1_INTC_s112
                                 + 1.02988 w_k1_KO_s112
                                 + 0.953767 w_k1_MAT_s112
                                 + 1.04309 w_k1_MSFT_s112
                                 + 1.06681 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.937811 w_k1_ADM_s112
                                 + 0.840246 w_k1_AXP_s112
                                 + 0.669327 w_k1_DIS_s112
                                 + 0.719127 w_k1_FDX_s112
                                 + 1.19113 w_k1_IBM_s112
                                 + 0.779734 w_k1_INTC_s112
                                 + 0.789882 w_k1_KO_s112
                                 + 0.911386 w_k1_MAT_s112
                                 + 0.609139 w_k1_MSFT_s112
                                 + 0.873818 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.32829 w_k1_ADM_s112
                                 + 1.51221 w_k1_AXP_s112
                                 + 1.39771 w_k1_DIS_s112
                                 + 1.26857 w_k1_FDX_s112
                                 + 1.08453 w_k1_IBM_s112
                                 + 1.3484 w_k1_INTC_s112 + 1.0906 w_k1_KO_s112
                                 + 1.42038 w_k1_MAT_s112
                                 + 1.32748 w_k1_MSFT_s112
                                 + 1.07499 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.02514 w_k1_ADM_s112
                                 + 1.12817 w_k1_AXP_s112
                                 + 1.36936 w_k1_DIS_s112
                                 + 1.09337 w_k1_FDX_s112
                                 + 0.865489 w_k1_IBM_s112
                                 + 1.10094 w_k1_INTC_s112
                                 + 1.01047 w_k1_KO_s112
                                 + 0.731527 w_k1_MAT_s112
                                 + 1.2494 w_k1_MSFT_s112
                                 + 0.987701 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.925549 w_k1_ADM_s112
                                 + 1.20964 w_k1_AXP_s112
                                 + 0.968985 w_k1_DIS_s112
                                 + 0.964659 w_k1_FDX_s112
                                 + 1.32705 w_k1_IBM_s112
                                 + 1.11223 w_k1_INTC_s112
                                 + 1.02149 w_k1_KO_s112 + 1.72726 w_k1_MAT_s112
                                 + 1.03247 w_k1_MSFT_s112
                                 + 1.08095 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.881384 w_k1_ADM_s113
                                 + 0.792283 w_k1_AXP_s113
                                 + 0.606737 w_k1_DIS_s113
                                 + 0.811029 w_k1_FDX_s113
                                 + 0.66275 w_k1_IBM_s113
                                 + 0.83128 w_k1_INTC_s113
                                 + 0.763453 w_k1_KO_s113
                                 + 0.579912 w_k1_MAT_s113
                                 + 0.715977 w_k1_MSFT_s113
                                 + 0.831674 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.796032 w_k1_ADM_s113
                                 + 0.805394 w_k1_AXP_s113
                                 + 0.638498 w_k1_DIS_s113
                                 + 0.709368 w_k1_FDX_s113
                                 + 1.03899 w_k1_IBM_s113
                                 + 0.869394 w_k1_INTC_s113
                                 + 0.861683 w_k1_KO_s113
                                 + 0.818885 w_k1_MAT_s113
                                 + 0.778177 w_k1_MSFT_s113
                                 + 0.805092 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.35431 w_k1_ADM_s113
                                 + 1.53108 w_k1_AXP_s113 + 1.2736 w_k1_DIS_s113
                                 + 1.18732 w_k1_FDX_s113
                                 + 1.76695 w_k1_IBM_s113
                                 + 1.09234 w_k1_INTC_s113
                                 + 1.26856 w_k1_KO_s113 + 1.53195 w_k1_MAT_s113
                                 + 1.11352 w_k1_MSFT_s113
                                 + 1.26056 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.28189 w_k1_ADM_s113
                                 + 1.25689 w_k1_AXP_s113
                                 + 1.29647 w_k1_DIS_s113 + 1.286 w_k1_FDX_s113
                                 + 0.79983 w_k1_IBM_s113
                                 + 0.973519 w_k1_INTC_s113
                                 + 0.990812 w_k1_KO_s113
                                 + 0.753709 w_k1_MAT_s113
                                 + 1.08262 w_k1_MSFT_s113
                                 + 1.06424 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.10994 w_k1_ADM_s113
                                 + 1.07278 w_k1_AXP_s113
                                 + 0.890701 w_k1_DIS_s113
                                 + 0.917368 w_k1_FDX_s113
                                 + 0.916451 w_k1_IBM_s113
                                 + 1.14309 w_k1_INTC_s113
                                 + 0.858065 w_k1_KO_s113
                                 + 0.756465 w_k1_MAT_s113
                                 + 0.957304 w_k1_MSFT_s113
                                 + 0.982051 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.00513 w_k1_ADM_s113
                                 + 1.09495 w_k1_AXP_s113 + 1.034 w_k1_DIS_s113
                                 + 1.01764 w_k1_FDX_s113
                                 + 1.20038 w_k1_IBM_s113
                                 + 0.996878 w_k1_INTC_s113
                                 + 1.24377 w_k1_KO_s113 + 1.01236 w_k1_MAT_s113
                                 + 0.995256 w_k1_MSFT_s113
                                 + 0.988654 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.24219 w_k1_ADM_s113
                                 + 1.52723 w_k1_AXP_s113
                                 + 1.69323 w_k1_DIS_s113
                                 + 1.41459 w_k1_FDX_s113
                                 + 1.13406 w_k1_IBM_s113
                                 + 1.20427 w_k1_INTC_s113
                                 + 1.13462 w_k1_KO_s113
                                 + 0.610114 w_k1_MAT_s113
                                 + 1.20861 w_k1_MSFT_s113 + 1.0642 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.583877 w_k1_ADM_s113
                                 + 1.01706 w_k1_AXP_s113
                                 + 0.635883 w_k1_DIS_s113
                                 + 0.784435 w_k1_FDX_s113
                                 + 1.45542 w_k1_IBM_s113
                                 + 0.747068 w_k1_INTC_s113
                                 + 0.929556 w_k1_KO_s113
                                 + 1.13033 w_k1_MAT_s113
                                 + 0.671379 w_k1_MSFT_s113
                                 + 0.912689 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.38236 w_k1_ADM_s114
                                 + 1.2735 w_k1_AXP_s114 + 1.19521 w_k1_DIS_s114
                                 + 1.15831 w_k1_FDX_s114
                                 + 0.950309 w_k1_IBM_s114
                                 + 1.09173 w_k1_INTC_s114
                                 + 1.07313 w_k1_KO_s114
                                 + 0.969667 w_k1_MAT_s114
                                 + 1.18099 w_k1_MSFT_s114
                                 + 1.16802 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.734055 w_k1_ADM_s114
                                 + 0.824116 w_k1_AXP_s114
                                 + 0.886963 w_k1_DIS_s114
                                 + 0.956248 w_k1_FDX_s114
                                 + 0.952575 w_k1_IBM_s114
                                 + 0.968252 w_k1_INTC_s114
                                 + 0.947363 w_k1_KO_s114
                                 + 0.866628 w_k1_MAT_s114
                                 + 0.989045 w_k1_MSFT_s114
                                 + 0.934308 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.987888 w_k1_ADM_s114
                                 + 0.627734 w_k1_AXP_s114
                                 + 0.634305 w_k1_DIS_s114
                                 + 0.793666 w_k1_FDX_s114
                                 + 0.779752 w_k1_IBM_s114
                                 + 0.764826 w_k1_INTC_s114
                                 + 0.697054 w_k1_KO_s114
                                 + 0.660923 w_k1_MAT_s114
                                 + 0.838422 w_k1_MSFT_s114
                                 + 0.78903 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.03213 w_k1_ADM_s114
                                 + 0.913443 w_k1_AXP_s114
                                 + 1.04771 w_k1_DIS_s114
                                 + 0.920659 w_k1_FDX_s114
                                 + 1.48376 w_k1_IBM_s114
                                 + 0.999893 w_k1_INTC_s114
                                 + 1.25124 w_k1_KO_s114 + 1.14577 w_k1_MAT_s114
                                 + 1.01836 w_k1_MSFT_s114
                                 + 0.95088 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.26256 w_k1_ADM_s114
                                 + 0.871985 w_k1_AXP_s114
                                 + 0.80321 w_k1_DIS_s114
                                 + 0.98958 w_k1_FDX_s114
                                 + 0.715994 w_k1_IBM_s114
                                 + 0.804906 w_k1_INTC_s114
                                 + 0.700458 w_k1_KO_s114
                                 + 0.592135 w_k1_MAT_s114
                                 + 0.893953 w_k1_MSFT_s114
                                 + 0.879794 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.947398 w_k1_ADM_s114
                                 + 0.671906 w_k1_AXP_s114
                                 + 0.630938 w_k1_DIS_s114
                                 + 0.693624 w_k1_FDX_s114
                                 + 1.10039 w_k1_IBM_s114
                                 + 0.806354 w_k1_INTC_s114
                                 + 0.959585 w_k1_KO_s114
                                 + 1.18404 w_k1_MAT_s114
                                 + 0.808433 w_k1_MSFT_s114
                                 + 0.997117 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.10496 w_k1_ADM_s114
                                 + 1.0751 w_k1_AXP_s114 + 1.06158 w_k1_DIS_s114
                                 + 1.0302 w_k1_FDX_s114 + 1.1125 w_k1_IBM_s114
                                 + 0.76695 w_k1_INTC_s114
                                 + 1.02168 w_k1_KO_s114 + 1.43525 w_k1_MAT_s114
                                 + 0.980846 w_k1_MSFT_s114
                                 + 0.951217 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.3616 w_k1_ADM_s114
                                 + 1.50616 w_k1_AXP_s114
                                 + 1.79937 w_k1_DIS_s114
                                 + 1.30416 w_k1_FDX_s114
                                 + 1.41564 w_k1_IBM_s114 + 1.185 w_k1_INTC_s114
                                 + 1.47287 w_k1_KO_s114 + 1.40571 w_k1_MAT_s114
                                 + 1.30204 w_k1_MSFT_s114
                                 + 1.25376 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.31523 w_k1_ADM_s115
                                 + 0.87067 w_k1_AXP_s115
                                 + 0.862086 w_k1_DIS_s115
                                 + 0.934446 w_k1_FDX_s115
                                 + 0.832206 w_k1_IBM_s115
                                 + 1.02615 w_k1_INTC_s115
                                 + 1.07653 w_k1_KO_s115
                                 + 0.755761 w_k1_MAT_s115
                                 + 0.933971 w_k1_MSFT_s115
                                 + 0.914422 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.971998 w_k1_ADM_s115
                                 + 0.676963 w_k1_AXP_s115
                                 + 0.606694 w_k1_DIS_s115
                                 + 0.774785 w_k1_FDX_s115
                                 + 0.656607 w_k1_IBM_s115
                                 + 0.797256 w_k1_INTC_s115
                                 + 0.735318 w_k1_KO_s115
                                 + 0.575872 w_k1_MAT_s115
                                 + 0.668566 w_k1_MSFT_s115
                                 + 0.810986 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.76926 w_k1_ADM_s115
                                 + 1.33767 w_k1_AXP_s115
                                 + 1.34954 w_k1_DIS_s115
                                 + 1.12519 w_k1_FDX_s115
                                 + 1.45163 w_k1_IBM_s115
                                 + 1.24569 w_k1_INTC_s115
                                 + 1.52756 w_k1_KO_s115 + 1.47586 w_k1_MAT_s115
                                 + 1.05096 w_k1_MSFT_s115
                                 + 1.19814 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.874003 w_k1_ADM_s115
                                 + 0.698214 w_k1_AXP_s115
                                 + 0.593806 w_k1_DIS_s115
                                 + 0.628605 w_k1_FDX_s115
                                 + 1.04375 w_k1_IBM_s115
                                 + 0.912787 w_k1_INTC_s115
                                 + 0.959057 w_k1_KO_s115
                                 + 1.09016 w_k1_MAT_s115
                                 + 0.70271 w_k1_MSFT_s115
                                 + 0.943494 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.275 w_k1_ADM_s115
                                 + 0.90589 w_k1_AXP_s115
                                 + 0.836757 w_k1_DIS_s115
                                 + 0.792067 w_k1_FDX_s115
                                 + 1.5283 w_k1_IBM_s115
                                 + 1.07943 w_k1_INTC_s115
                                 + 1.38147 w_k1_KO_s115 + 1.49432 w_k1_MAT_s115
                                 + 0.845166 w_k1_MSFT_s115
                                 + 1.12642 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.15327 w_k1_ADM_s115
                                 + 0.867579 w_k1_AXP_s115
                                 + 0.920648 w_k1_DIS_s115
                                 + 0.898441 w_k1_FDX_s115
                                 + 1.05882 w_k1_IBM_s115
                                 + 1.20708 w_k1_INTC_s115
                                 + 1.27919 w_k1_KO_s115
                                 + 0.800783 w_k1_MAT_s115
                                 + 0.98345 w_k1_MSFT_s115
                                 + 0.982032 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.06867 w_k1_ADM_s115
                                 + 1.08849 w_k1_AXP_s115
                                 + 1.03624 w_k1_DIS_s115
                                 + 1.12768 w_k1_FDX_s115
                                 + 0.785147 w_k1_IBM_s115
                                 + 1.33249 w_k1_INTC_s115
                                 + 1.15596 w_k1_KO_s115
                                 + 0.876177 w_k1_MAT_s115
                                 + 1.03887 w_k1_MSFT_s115 + 1.0958 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.47347 w_k1_ADM_s115
                                 + 1.23848 w_k1_AXP_s115
                                 + 1.71646 w_k1_DIS_s115
                                 + 1.45741 w_k1_FDX_s115
                                 + 0.940014 w_k1_IBM_s115
                                 + 1.40517 w_k1_INTC_s115
                                 + 1.35145 w_k1_KO_s115
                                 + 0.842078 w_k1_MAT_s115
                                 + 1.12968 w_k1_MSFT_s115
                                 + 1.17707 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.6553 w_k1_ADM_s116
                                 + 1.23426 w_k1_AXP_s116
                                 + 1.54095 w_k1_DIS_s116
                                 + 1.30569 w_k1_FDX_s116
                                 + 1.13165 w_k1_IBM_s116
                                 + 1.13886 w_k1_INTC_s116
                                 + 1.26212 w_k1_KO_s116 + 0.93871 w_k1_MAT_s116
                                 + 1.22248 w_k1_MSFT_s116
                                 + 1.19649 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.904677 w_k1_ADM_s116
                                 + 0.749091 w_k1_AXP_s116
                                 + 0.682194 w_k1_DIS_s116
                                 + 0.791265 w_k1_FDX_s116
                                 + 0.689247 w_k1_IBM_s116
                                 + 0.755188 w_k1_INTC_s116
                                 + 0.784042 w_k1_KO_s116
                                 + 0.591149 w_k1_MAT_s116
                                 + 0.770725 w_k1_MSFT_s116
                                 + 0.784012 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.17505 w_k1_ADM_s116
                                 + 1.57526 w_k1_AXP_s116
                                 + 1.35274 w_k1_DIS_s116
                                 + 1.20512 w_k1_FDX_s116
                                 + 1.52652 w_k1_IBM_s116
                                 + 1.09544 w_k1_INTC_s116
                                 + 1.33755 w_k1_KO_s116 + 1.59262 w_k1_MAT_s116
                                 + 1.11833 w_k1_MSFT_s116
                                 + 1.19268 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.0609 w_k1_ADM_s116
                                 + 0.823341 w_k1_AXP_s116
                                 + 0.815397 w_k1_DIS_s116
                                 + 0.949027 w_k1_FDX_s116
                                 + 0.79029 w_k1_IBM_s116
                                 + 0.951003 w_k1_INTC_s116
                                 + 0.963713 w_k1_KO_s116
                                 + 0.741927 w_k1_MAT_s116
                                 + 0.892371 w_k1_MSFT_s116
                                 + 0.910472 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.931576 w_k1_ADM_s116
                                 + 0.850115 w_k1_AXP_s116
                                 + 0.657467 w_k1_DIS_s116
                                 + 0.738806 w_k1_FDX_s116
                                 + 0.937609 w_k1_IBM_s116
                                 + 0.733992 w_k1_INTC_s116
                                 + 0.747217 w_k1_KO_s116
                                 + 0.76408 w_k1_MAT_s116
                                 + 0.702155 w_k1_MSFT_s116
                                 + 0.718272 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.28794 w_k1_ADM_s116
                                 + 0.881969 w_k1_AXP_s116
                                 + 1.25431 w_k1_DIS_s116
                                 + 1.03621 w_k1_FDX_s116
                                 + 1.04402 w_k1_IBM_s116
                                 + 0.914751 w_k1_INTC_s116
                                 + 1.2519 w_k1_KO_s116 + 0.948452 w_k1_MAT_s116
                                 + 1.12676 w_k1_MSFT_s116
                                 + 1.05452 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.11741 w_k1_ADM_s116
                                 + 0.917213 w_k1_AXP_s116
                                 + 0.936618 w_k1_DIS_s116
                                 + 0.888071 w_k1_FDX_s116
                                 + 1.23398 w_k1_IBM_s116
                                 + 1.07875 w_k1_INTC_s116
                                 + 1.23949 w_k1_KO_s116 + 1.09334 w_k1_MAT_s116
                                 + 0.954983 w_k1_MSFT_s116
                                 + 0.993036 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.12585 w_k1_ADM_s116
                                 + 1.2176 w_k1_AXP_s116
                                 + 0.998671 w_k1_DIS_s116
                                 + 1.07301 w_k1_FDX_s116
                                 + 0.840426 w_k1_IBM_s116
                                 + 0.798499 w_k1_INTC_s116
                                 + 0.894606 w_k1_KO_s116
                                 + 0.853591 w_k1_MAT_s116
                                 + 0.863582 w_k1_MSFT_s116
                                 + 0.950237 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.09018 w_k1_ADM_s117
                                 + 1.61695 w_k1_AXP_s117
                                 + 1.29374 w_k1_DIS_s117
                                 + 1.42535 w_k1_FDX_s117
                                 + 1.18472 w_k1_IBM_s117
                                 + 1.29195 w_k1_INTC_s117
                                 + 1.20277 w_k1_KO_s117 + 1.69909 w_k1_MAT_s117
                                 + 1.16929 w_k1_MSFT_s117
                                 + 1.23528 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.973851 w_k1_ADM_s117
                                 + 1.07124 w_k1_AXP_s117
                                 + 1.38895 w_k1_DIS_s117
                                 + 1.21429 w_k1_FDX_s117
                                 + 0.959666 w_k1_IBM_s117
                                 + 1.07127 w_k1_INTC_s117
                                 + 1.25327 w_k1_KO_s117 + 1.0268 w_k1_MAT_s117
                                 + 1.10769 w_k1_MSFT_s117
                                 + 0.927079 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.02514 w_k1_ADM_s117
                                 + 0.874275 w_k1_AXP_s117
                                 + 0.743526 w_k1_DIS_s117
                                 + 0.955906 w_k1_FDX_s117
                                 + 0.779876 w_k1_IBM_s117
                                 + 0.839222 w_k1_INTC_s117
                                 + 0.891087 w_k1_KO_s117
                                 + 0.606857 w_k1_MAT_s117
                                 + 0.751148 w_k1_MSFT_s117
                                 + 0.897445 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.879518 w_k1_ADM_s117
                                 + 0.910523 w_k1_AXP_s117
                                 + 0.54205 w_k1_DIS_s117
                                 + 0.777347 w_k1_FDX_s117
                                 + 0.65765 w_k1_IBM_s117
                                 + 0.860154 w_k1_INTC_s117
                                 + 0.766522 w_k1_KO_s117
                                 + 0.657255 w_k1_MAT_s117
                                 + 0.684232 w_k1_MSFT_s117
                                 + 0.834017 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.2426 w_k1_ADM_s117
                                 + 1.43739 w_k1_AXP_s117
                                 + 1.13953 w_k1_DIS_s117
                                 + 1.13394 w_k1_FDX_s117
                                 + 1.49974 w_k1_IBM_s117
                                 + 0.931821 w_k1_INTC_s117
                                 + 1.52679 w_k1_KO_s117 + 1.93084 w_k1_MAT_s117
                                 + 0.966791 w_k1_MSFT_s117
                                 + 1.12439 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.800251 w_k1_ADM_s117
                                 + 0.949647 w_k1_AXP_s117
                                 + 0.87156 w_k1_DIS_s117
                                 + 0.824877 w_k1_FDX_s117
                                 + 0.984587 w_k1_IBM_s117
                                 + 0.959986 w_k1_INTC_s117
                                 + 1.12279 w_k1_KO_s117
                                 + 0.889187 w_k1_MAT_s117
                                 + 0.823834 w_k1_MSFT_s117
                                 + 0.929858 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.873022 w_k1_ADM_s117
                                 + 1.43406 w_k1_AXP_s117
                                 + 1.87749 w_k1_DIS_s117
                                 + 1.42963 w_k1_FDX_s117
                                 + 1.35978 w_k1_IBM_s117
                                 + 1.30905 w_k1_INTC_s117
                                 + 1.66107 w_k1_KO_s117 + 1.8403 w_k1_MAT_s117
                                 + 1.16952 w_k1_MSFT_s117
                                 + 1.08356 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.946636 w_k1_ADM_s117
                                 + 1.05097 w_k1_AXP_s117
                                 + 0.950409 w_k1_DIS_s117
                                 + 0.945144 w_k1_FDX_s117
                                 + 1.31525 w_k1_IBM_s117
                                 + 0.965768 w_k1_INTC_s117
                                 + 1.30962 w_k1_KO_s117 + 1.12346 w_k1_MAT_s117
                                 + 0.944527 w_k1_MSFT_s117
                                 + 1.06782 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0134483632647911 w_k1_ADM_s000
                                 + 0.0465976759521173 w_k1_ADM_s001
                                 + 0.041999922287976 w_k1_ADM_s002
                                 + 0.0442200410830071 w_k1_ADM_s003
                                 + 0.0675811648095643 w_k1_ADM_s004
                                 + 0.0709134441359467 w_k1_ADM_s005
                                 + 0.0320663431012628 w_k1_ADM_s006
                                 + 0.0297122356170549 w_k1_ADM_s007
                                 + 0.0146376351382371 w_k1_ADM_s010
                                 + 0.0382731083783952 w_k1_ADM_s011
                                 + 0.0599898346968016 w_k1_ADM_s012
                                 + 0.018165130218024 w_k1_ADM_s013
                                 + 0.0624926896935409 w_k1_ADM_s014
                                 + 0.0442005212486257 w_k1_ADM_s015
                                 + 0.0141018640925404 w_k1_ADM_s016
                                 + 0.0136579424110995 w_k1_ADM_s017
                                 + 0.0332974892194214 w_k1_ADM_s100
                                 + 0.0443969715861691 w_k1_ADM_s101
                                 + 0.0143934066709752 w_k1_ADM_s102
                                 + 0.0201299784709161 w_k1_ADM_s103
                                 + 0.019935734160685 w_k1_ADM_s104
                                 + 0.0143899877398281 w_k1_ADM_s105
                                 + 0.0408607232230246 w_k1_ADM_s106
                                 + 0.0160281542557511 w_k1_ADM_s107
                                 + 0.0168992654293323 w_k1_ADM_s110
                                 + 0.0172043879550762 w_k1_ADM_s111
                                 + 0.0508134138919967 w_k1_ADM_s112
                                 + 0.0313690102048675 w_k1_ADM_s113
                                 + 0.0116231642771402 w_k1_ADM_s114
                                 + 0.0509410612263557 w_k1_ADM_s115
                                 + 0.0333255588340822 w_k1_ADM_s116
                                 + 0.0191755730727026 w_k1_ADM_s117
                                 + 0.0169123829531796 w_k1_AXP_s000
                                 + 0.0511056935900125 w_k1_AXP_s001
                                 + 0.0400545675267708 w_k1_AXP_s002
                                 + 0.0403117563274525 w_k1_AXP_s003
                                 + 0.0655348535567444 w_k1_AXP_s004
                                 + 0.0741678489896295 w_k1_AXP_s005
                                 + 0.0331232183816371 w_k1_AXP_s006
                                 + 0.0288704515879643 w_k1_AXP_s007
                                 + 0.0155499759074511 w_k1_AXP_s010
                                 + 0.0387580242123384 w_k1_AXP_s011
                                 + 0.056691369776594 w_k1_AXP_s012
                                 + 0.0171115235642828 w_k1_AXP_s013
                                 + 0.0589022874385806 w_k1_AXP_s014
                                 + 0.040493685457801 w_k1_AXP_s015
                                 + 0.0134686346077619 w_k1_AXP_s016
                                 + 0.0138925742532151 w_k1_AXP_s017
                                 + 0.0339342053096889 w_k1_AXP_s100
                                 + 0.0525206393393698 w_k1_AXP_s101
                                 + 0.0132173849381465 w_k1_AXP_s102
                                 + 0.0194920594440919 w_k1_AXP_s103
                                 + 0.0211447722758842 w_k1_AXP_s104
                                 + 0.012789999413025 w_k1_AXP_s105
                                 + 0.043480096873582 w_k1_AXP_s106
                                 + 0.0160750459278153 w_k1_AXP_s107
                                 + 0.0142560827153388 w_k1_AXP_s110
                                 + 0.0137000900472337 w_k1_AXP_s111
                                 + 0.0577980064747873 w_k1_AXP_s112
                                 + 0.0331664066655522 w_k1_AXP_s113
                                 + 0.0107932877837147 w_k1_AXP_s114
                                 + 0.0404195058919457 w_k1_AXP_s115
                                 + 0.0296033932255215 w_k1_AXP_s116
                                 + 0.0217061637356052 w_k1_AXP_s117
                                 + 0.0160798235001341 w_k1_DIS_s000
                                 + 0.0518678256823176 w_k1_DIS_s001
                                 + 0.0395722136485067 w_k1_DIS_s002
                                 + 0.0425063959051546 w_k1_DIS_s003
                                 + 0.0635595078936851 w_k1_DIS_s004
                                 + 0.0782296822846803 w_k1_DIS_s005
                                 + 0.0325145063300044 w_k1_DIS_s006
                                 + 0.0302067359041101 w_k1_DIS_s007
                                 + 0.0145755705136327 w_k1_DIS_s010
                                 + 0.0387415691462371 w_k1_DIS_s011
                                 + 0.0600543095975873 w_k1_DIS_s012
                                 + 0.0180452896490405 w_k1_DIS_s013
                                 + 0.0636198348415518 w_k1_DIS_s014
                                 + 0.0431067187339369 w_k1_DIS_s015
                                 + 0.0166573062409608 w_k1_DIS_s016
                                 + 0.0137207050195134 w_k1_DIS_s017
                                 + 0.0345189732453728 w_k1_DIS_s100
                                 + 0.0478035630650647 w_k1_DIS_s101
                                 + 0.0136952402316303 w_k1_DIS_s102
                                 + 0.023123884091313 w_k1_DIS_s103
                                 + 0.0214918954159695 w_k1_DIS_s104
                                 + 0.0128236666986636 w_k1_DIS_s105
                                 + 0.0411734357612957 w_k1_DIS_s106
                                 + 0.0179731628189669 w_k1_DIS_s107
                                 + 0.0159071583070904 w_k1_DIS_s110
                                 + 0.0125496566381118 w_k1_DIS_s111
                                 + 0.050538094601863 w_k1_DIS_s112
                                 + 0.0296121324440713 w_k1_DIS_s113
                                 + 0.0114921053584104 w_k1_DIS_s114
                                 + 0.040882977897428 w_k1_DIS_s115
                                 + 0.0295068927969781 w_k1_DIS_s116
                                 + 0.0202916114994842 w_k1_DIS_s117
                                 + 0.0156574142213825 w_k1_FDX_s000
                                 + 0.0487252903074391 w_k1_FDX_s001
                                 + 0.0397824039382386 w_k1_FDX_s002
                                 + 0.0388513003302882 w_k1_FDX_s003
                                 + 0.0627289847363595 w_k1_FDX_s004
                                 + 0.07659075684662 w_k1_FDX_s005
                                 + 0.0317355428571209 w_k1_FDX_s006
                                 + 0.0301850265497727 w_k1_FDX_s007
                                 + 0.015812293527681 w_k1_FDX_s010
                                 + 0.035870700603901 w_k1_FDX_s011
                                 + 0.0584664271011485 w_k1_FDX_s012
                                 + 0.0173101369470765 w_k1_FDX_s013
                                 + 0.0577384307675286 w_k1_FDX_s014
                                 + 0.0396371625261371 w_k1_FDX_s015
                                 + 0.01464653846296 w_k1_FDX_s016
                                 + 0.0133951642211314 w_k1_FDX_s017
                                 + 0.0355858934451732 w_k1_FDX_s100
                                 + 0.0499339560656016 w_k1_FDX_s101
                                 + 0.0136239864163727 w_k1_FDX_s102
                                 + 0.0223793931412285 w_k1_FDX_s103
                                 + 0.0215727322629554 w_k1_FDX_s104
                                 + 0.0122472793006656 w_k1_FDX_s105
                                 + 0.0422774120971024 w_k1_FDX_s106
                                 + 0.0166664387386229 w_k1_FDX_s107
                                 + 0.0145234873923931 w_k1_FDX_s110
                                 + 0.0113977430711751 w_k1_FDX_s111
                                 + 0.0485205790532218 w_k1_FDX_s112
                                 + 0.0302203769946072 w_k1_FDX_s113
                                 + 0.0106673582479944 w_k1_FDX_s114
                                 + 0.0402784825054059 w_k1_FDX_s115
                                 + 0.0287421823466061 w_k1_FDX_s116
                                 + 0.020517871100761 w_k1_FDX_s117
                                 + 0.015261135618495 w_k1_IBM_s000
                                 + 0.0524646093835088 w_k1_IBM_s001
                                 + 0.0403322276992078 w_k1_IBM_s002
                                 + 0.0413751607727554 w_k1_IBM_s003
                                 + 0.0621990077040976 w_k1_IBM_s004
                                 + 0.0779622070665597 w_k1_IBM_s005
                                 + 0.0334385366284407 w_k1_IBM_s006
                                 + 0.0312812761460292 w_k1_IBM_s007
                                 + 0.0154634379256116 w_k1_IBM_s010
                                 + 0.0363458974024756 w_k1_IBM_s011
                                 + 0.0580574380091974 w_k1_IBM_s012
                                 + 0.0183590991546263 w_k1_IBM_s013
                                 + 0.0592958063490645 w_k1_IBM_s014
                                 + 0.0404746842063755 w_k1_IBM_s015
                                 + 0.0152882538506909 w_k1_IBM_s016
                                 + 0.0133719477336362 w_k1_IBM_s017
                                 + 0.0354329125411806 w_k1_IBM_s100
                                 + 0.0482707242248681 w_k1_IBM_s101
                                 + 0.0140110390807121 w_k1_IBM_s102
                                 + 0.0227425223318619 w_k1_IBM_s103
                                 + 0.0208719052998784 w_k1_IBM_s104
                                 + 0.0123516472418292 w_k1_IBM_s105
                                 + 0.0430835990521765 w_k1_IBM_s106
                                 + 0.0159298637554254 w_k1_IBM_s107
                                 + 0.014749075123898 w_k1_IBM_s110
                                 + 0.0122052657642079 w_k1_IBM_s111
                                 + 0.0490058740352566 w_k1_IBM_s112
                                 + 0.031556123970064 w_k1_IBM_s113
                                 + 0.011764463215388 w_k1_IBM_s114
                                 + 0.0410915896690244 w_k1_IBM_s115
                                 + 0.0298937072436063 w_k1_IBM_s116
                                 + 0.0207606312500502 w_k1_IBM_s117
                                 + 0.0128791409605352 w_k1_INTC_s000
                                 + 0.0551141061278554 w_k1_INTC_s001
                                 + 0.0349870506576241 w_k1_INTC_s002
                                 + 0.0426231890068806 w_k1_INTC_s003
                                 + 0.0656792882545329 w_k1_INTC_s004
                                 + 0.0759462201808629 w_k1_INTC_s005
                                 + 0.0271308174418057 w_k1_INTC_s006
                                 + 0.0280601065720516 w_k1_INTC_s007
                                 + 0.015869254005813 w_k1_INTC_s010
                                 + 0.0370065225564485 w_k1_INTC_s011
                                 + 0.0511282201662363 w_k1_INTC_s012
                                 + 0.0176682671430349 w_k1_INTC_s013
                                 + 0.0589739712277668 w_k1_INTC_s014
                                 + 0.0420044763101333 w_k1_INTC_s015
                                 + 0.01507517885753 w_k1_INTC_s016
                                 + 0.0131952504134531 w_k1_INTC_s017
                                 + 0.0380871170363014 w_k1_INTC_s100
                                 + 0.047582194294597 w_k1_INTC_s101
                                 + 0.0133521308705219 w_k1_INTC_s102
                                 + 0.0223766130689576 w_k1_INTC_s103
                                 + 0.0222684730571656 w_k1_INTC_s104
                                 + 0.0117509205135853 w_k1_INTC_s105
                                 + 0.0450125909913866 w_k1_INTC_s106
                                 + 0.0158457886528534 w_k1_INTC_s107
                                 + 0.014272568288613 w_k1_INTC_s110
                                 + 0.0109638812890562 w_k1_INTC_s111
                                 + 0.0513467405395922 w_k1_INTC_s112
                                 + 0.0302297641862228 w_k1_INTC_s113
                                 + 0.00991525858645272 w_k1_INTC_s114
                                 + 0.0468176459238173 w_k1_INTC_s115
                                 + 0.0274739606773459 w_k1_INTC_s116
                                 + 0.0198272650964975 w_k1_INTC_s117
                                 + 0.0150934675119035 w_k1_KO_s000
                                 + 0.0437655858919271 w_k1_KO_s001
                                 + 0.0390082485603209 w_k1_KO_s002
                                 + 0.0410955777181049 w_k1_KO_s003
                                 + 0.0591814868916919 w_k1_KO_s004
                                 + 0.0806931829627782 w_k1_KO_s005
                                 + 0.0327203295017042 w_k1_KO_s006
                                 + 0.0316680864839388 w_k1_KO_s007
                                 + 0.0142606238525894 w_k1_KO_s010
                                 + 0.0387473251747066 w_k1_KO_s011
                                 + 0.0554136771003496 w_k1_KO_s012
                                 + 0.0173443311244424 w_k1_KO_s013
                                 + 0.060802018509846 w_k1_KO_s014
                                 + 0.0456575790290081 w_k1_KO_s015
                                 + 0.0168419336274246 w_k1_KO_s016
                                 + 0.0149324292077427 w_k1_KO_s017
                                 + 0.0365250982993562 w_k1_KO_s100
                                 + 0.0500320107895936 w_k1_KO_s101
                                 + 0.0150694435934325 w_k1_KO_s102
                                 + 0.0204273122735831 w_k1_KO_s103
                                 + 0.0231950576846569 w_k1_KO_s104
                                 + 0.0132111621861653 w_k1_KO_s105
                                 + 0.0376564449715351 w_k1_KO_s106
                                 + 0.0180570618754753 w_k1_KO_s107
                                 + 0.0145813806513864 w_k1_KO_s110
                                 + 0.0108385773498061 w_k1_KO_s111
                                 + 0.0456057028420471 w_k1_KO_s112
                                 + 0.0302504492737323 w_k1_KO_s113
                                 + 0.0112631359106299 w_k1_KO_s114
                                 + 0.0485662255231632 w_k1_KO_s115
                                 + 0.0310807160401416 w_k1_KO_s116
                                 + 0.0232313393355337 w_k1_KO_s117
                                 + 0.0202294889196186 w_k1_MAT_s000
                                 + 0.0481576208264041 w_k1_MAT_s001
                                 + 0.0362815402757409 w_k1_MAT_s002
                                 + 0.0422608165167353 w_k1_MAT_s003
                                 + 0.0468629543343567 w_k1_MAT_s004
                                 + 0.102046753566874 w_k1_MAT_s005
                                 + 0.0522239305388634 w_k1_MAT_s006
                                 + 0.0268523798122262 w_k1_MAT_s007
                                 + 0.0142325829900762 w_k1_MAT_s010
                                 + 0.0408052160442553 w_k1_MAT_s011
                                 + 0.0515146050670234 w_k1_MAT_s012
                                 + 0.0181498723261883 w_k1_MAT_s013
                                 + 0.0573014313373837 w_k1_MAT_s014
                                 + 0.0338194766057577 w_k1_MAT_s015
                                 + 0.0156686192431659 w_k1_MAT_s016
                                 + 0.0130850892500096 w_k1_MAT_s017
                                 + 0.0403354869835266 w_k1_MAT_s100
                                 + 0.0660941659247663 w_k1_MAT_s101
                                 + 0.0181172634770123 w_k1_MAT_s102
                                 + 0.020252705700037 w_k1_MAT_s103
                                 + 0.0247891698927533 w_k1_MAT_s104
                                 + 0.0103375542118188 w_k1_MAT_s105
                                 + 0.0455278644046436 w_k1_MAT_s106
                                 + 0.0176499032356285 w_k1_MAT_s107
                                 + 0.0142789314819797 w_k1_MAT_s110
                                 + 0.0114339480937652 w_k1_MAT_s111
                                 + 0.0573594255174168 w_k1_MAT_s112
                                 + 0.0257465310750985 w_k1_MAT_s113
                                 + 0.0115252252163558 w_k1_MAT_s114
                                 + 0.0396158955655513 w_k1_MAT_s115
                                 + 0.0275514330753376 w_k1_MAT_s116
                                 + 0.0214036015579949 w_k1_MAT_s117
                                 + 0.0146181554906575 w_k1_MSFT_s000
                                 + 0.0459323173887312 w_k1_MSFT_s001
                                 + 0.0404998689710271 w_k1_MSFT_s002
                                 + 0.0414750800866348 w_k1_MSFT_s003
                                 + 0.0609345621769762 w_k1_MSFT_s004
                                 + 0.0801202647479436 w_k1_MSFT_s005
                                 + 0.0318216147726506 w_k1_MSFT_s006
                                 + 0.0282749640753979 w_k1_MSFT_s007
                                 + 0.0152834544467539 w_k1_MSFT_s010
                                 + 0.0373789081326795 w_k1_MSFT_s011
                                 + 0.0579858993011946 w_k1_MSFT_s012
                                 + 0.0172804349198771 w_k1_MSFT_s013
                                 + 0.0602340719109469 w_k1_MSFT_s014
                                 + 0.03838260494505 w_k1_MSFT_s015
                                 + 0.0158870759566552 w_k1_MSFT_s016
                                 + 0.0138428280909321 w_k1_MSFT_s017
                                 + 0.0374267723940458 w_k1_MSFT_s100
                                 + 0.0480755161994793 w_k1_MSFT_s101
                                 + 0.0139054834160292 w_k1_MSFT_s102
                                 + 0.021789837825502 w_k1_MSFT_s103
                                 + 0.0210527970806113 w_k1_MSFT_s104
                                 + 0.0115930722092969 w_k1_MSFT_s105
                                 + 0.0437135522800736 w_k1_MSFT_s106
                                 + 0.0162721500967799 w_k1_MSFT_s107
                                 + 0.0143416306795815 w_k1_MSFT_s110
                                 + 0.0109915681508344 w_k1_MSFT_s111
                                 + 0.0485447657591801 w_k1_MSFT_s112
                                 + 0.028500734420756 w_k1_MSFT_s113
                                 + 0.0108446293387538 w_k1_MSFT_s114
                                 + 0.038361344513267 w_k1_MSFT_s115
                                 + 0.0275950091433846 w_k1_MSFT_s116
                                 + 0.0182477506402805 w_k1_MSFT_s117
                                 + 0.0150502105476544 w_k1_PG_s000
                                 + 0.048607491219769 w_k1_PG_s001
                                 + 0.0415749408547727 w_k1_PG_s002
                                 + 0.043486140614915 w_k1_PG_s003
                                 + 0.0596879647382714 w_k1_PG_s004
                                 + 0.0778965287389535 w_k1_PG_s005
                                 + 0.0305505074437229 w_k1_PG_s006
                                 + 0.0291465806801573 w_k1_PG_s007
                                 + 0.0143330793626962 w_k1_PG_s010
                                 + 0.0351466995530982 w_k1_PG_s011
                                 + 0.0571491837259842 w_k1_PG_s012
                                 + 0.017656385881984 w_k1_PG_s013
                                 + 0.0603153189082003 w_k1_PG_s014
                                 + 0.0398224354314626 w_k1_PG_s015
                                 + 0.0167557782202756 w_k1_PG_s016
                                 + 0.0141155886891623 w_k1_PG_s017
                                 + 0.0375267940436616 w_k1_PG_s100
                                 + 0.0471046602853132 w_k1_PG_s101
                                 + 0.014400922659639 w_k1_PG_s102
                                 + 0.0211234857227509 w_k1_PG_s103
                                 + 0.0201366105009769 w_k1_PG_s104
                                 + 0.0121991275219826 w_k1_PG_s105
                                 + 0.0400253591988833 w_k1_PG_s106
                                 + 0.0160797573528938 w_k1_PG_s107
                                 + 0.0149755515034421 w_k1_PG_s110
                                 + 0.0126589111208157 w_k1_PG_s111
                                 + 0.0481466697303866 w_k1_PG_s112
                                 + 0.0295223876231458 w_k1_PG_s113
                                 + 0.010564870710765 w_k1_PG_s114
                                 + 0.041986845594769 w_k1_PG_s115
                                 + 0.0282700389893497 w_k1_PG_s116
                                 + 0.0196968298314447 w_k1_PG_s117 >= 1.2
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
