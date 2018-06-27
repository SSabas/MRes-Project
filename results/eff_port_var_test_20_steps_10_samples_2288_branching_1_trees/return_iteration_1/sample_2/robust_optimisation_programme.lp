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
 w_k1_ADM_s0_return_constraint:  1.17423 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.93875 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.11938 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.961898 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.16239 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.855449 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.59195 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.34134 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.18135 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.989159 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.18194 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.1032 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.758582 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.05711 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.772772 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.881429 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.34293 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.27812 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.845448 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.18235 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.916527 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.00531 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.23432 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.679253 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.10748 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.861338 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.25716 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.809814 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.05969 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.45965 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.2746 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.912296 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10726 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.19281 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.00377 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.28131 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.843065 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.910402 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23381 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.909781 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.42036 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.92137 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.1538 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.851155 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.74228 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.35571 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.05543 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.918728 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.14171 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.984209 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.694263 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.915833 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.761861 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.05473 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.49118 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.50924 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.977396 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.28779 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.04069 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.1106 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.57584 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.902861 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.35821 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.00357 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.40337 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.740843 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.04316 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.54639 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.257 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.651384 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.716232 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.1496 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.02562 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.39417 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.877024 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.656581 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33751 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.888109 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.54334 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.907705 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.27395 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.751607 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.86446 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.49491 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.830877 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.31749 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.24616 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.833919 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.616985 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.972853 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.692115 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.10092 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.87657 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.17875 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.819192 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.39282 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.04561 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.04395 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.58013 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.756495 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.21967 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.986803 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.35179 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.67541 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.20407 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.87202 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.55601 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.654249 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.946209 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.32437 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.94487 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.21706 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.770304 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.611 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14153 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.94534 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.34462 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.966294 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.11597 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.857365 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.588 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.25217 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.942003 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.05435 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.2633 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.974494 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.687937 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.83369 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.825341 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.989265 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.3701 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.20884 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.81802 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.39018 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.0399 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.09661 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.2327 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.78174 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.19134 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.945111 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.32305 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.797321 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.14012 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.62901 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.4096 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.685784 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.04889 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.08191 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.01332 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.23722 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.753936 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.774359 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.29668 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.910641 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.34011 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.945825 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.12733 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.841314 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.921981 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.31367 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.999997 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.02663 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.755 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.702451 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.03187 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.44644 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.724476 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.68797 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.29154 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.22829 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.18498 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.801755 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.977736 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.771503 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 2.04008 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.3187 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.41536 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.02532 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.942055 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.701134 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.85709 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.999494 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.864721 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.887606 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.638365 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.57909 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.986481 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.947175 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.28894 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.610197 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10593 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.997665 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.34509 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.94719 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.01144 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.815561 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.25997 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.29389 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.800189 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.944273 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.891303 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.801043 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.808759 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.794376 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.822031 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.11944 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.29058 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.20174 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.07225 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.3102 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.01359 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.01714 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.17502 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.88421 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.03958 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.03521 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.26142 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.940781 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.10603 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.64897 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.23503 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.827626 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.917636 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.08479 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.987091 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.13129 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.740536 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.886257 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26637 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.963505 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.20252 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.935968 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.1489 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.756855 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.35403 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.5327 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.09975 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.31587 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.986884 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.893186 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.858377 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.17464 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.73077 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.25082 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.47143 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.18833 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.980806 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.984313 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.958 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.870163 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.4925 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.919307 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.14167 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.922114 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.12633 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.671721 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.00662 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.30274 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.08214 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.918171 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.885221 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.36778 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.0061 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.1101 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.01693 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.676265 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28117 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.923235 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.45297 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.990977 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.24536 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.81376 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.05907 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.74489 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.32373 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.11607 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.927524 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.765994 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.12032 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.45754 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.674682 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.89173 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.52443 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.51937 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.26052 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.775453 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.09145 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.615272 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 2.26569 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.35245 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.26864 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.970066 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.05511 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.651452 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.650987 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.920935 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.75524 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.69313 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.665773 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.44806 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.930625 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.25414 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.1945 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.619582 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16759 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.932909 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.2702 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.909855 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.1301 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.818597 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.46291 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.27309 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.925717 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.13387 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.08755 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.82523 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.866399 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.84761 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.848639 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.19483 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.48729 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.27637 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.850128 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.32005 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.05372 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.11368 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.14847 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.812179 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.11467 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.03226 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.21962 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.834626 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.22101 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.55582 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.46456 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.831947 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.06536 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.10174 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.03985 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.2828 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.798753 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.883805 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14717 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.972125 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12241 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.990431 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06737 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.840034 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.31185 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.26986 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.03563 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.17575 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.06702 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.998505 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.918451 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.03492 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.777752 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.07101 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.20248 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.10076 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.918027 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.01783 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.985895 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.956843 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.27015 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.9422 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.12925 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.01696 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.13052 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.801912 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.04103 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.33316 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.08319 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.839645 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.97676 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.12202 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.02477 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.18563 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.952255 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.873667 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.02599993664867 z_k1_s0000
                                 - 0.0509998037994654 z_k1_s0001
                                 - 0.0439998245132774 z_k1_s0002
                                 - 0.0239999561525877 z_k1_s0003
                                 - 0.0159999707683918 z_k1_s0004
                                 - 0.061999997677285 z_k1_s0005
                                 - 0.061999997677285 z_k1_s0006
                                 - 0.0319998147370501 z_k1_s0007
                                 - 0.0620000705812871 z_k1_s0010
                                 - 0.122000515918628 z_k1_s0011
                                 - 0.127000066029283 z_k1_s0012
                                 - 0.258000356548039 z_k1_s0013
                                 - 0.0670001076927322 z_k1_s0014
                                 - 0.181000661633206 z_k1_s0015
                                 - 0.0910001884275106 z_k1_s0016
                                 - 0.0660001002704432 z_k1_s0017
                                 - 0.0850000607892301 z_k1_s0020
                                 - 0.0469999569894293 z_k1_s0021
                                 - 0.0760001073997197 z_k1_s0022
                                 - 0.104999957210364 z_k1_s0023
                                 - 0.03099993965264 z_k1_s0024
                                 - 0.0389999483210347 z_k1_s0025
                                 - 0.0900000348945137 z_k1_s0026
                                 - 0.0279999551894699 z_k1_s0027
                                 - 0.0450001451292098 z_k1_s0030
                                 - 0.0319999642364532 z_k1_s0031
                                 - 0.0519998897717838 z_k1_s0032
                                 - 0.0369999550952773 z_k1_s0033
                                 - 0.0239999637023485 z_k1_s0034
                                 - 0.0709999005152739 z_k1_s0035
                                 - 0.0919998924423073 z_k1_s0036
                                 - 0.0259999827858575 z_k1_s0037
                                 - 0.0769998877396417 z_k1_s0040
                                 - 0.0529999328935029 z_k1_s0041
                                 - 0.0719996839387124 z_k1_s0042
                                 - 0.0589998764551116 z_k1_s0043
                                 - 0.112000110348121 z_k1_s0044
                                 - 0.0879996137028707 z_k1_s0045
                                 - 0.0919998971434167 z_k1_s0046
                                 - 0.0489998902525621 z_k1_s0047
                                 - 0.242000166878784 z_k1_s0050
                                 - 0.190999251256163 z_k1_s0051
                                 - 0.132999920350852 z_k1_s0052
                                 - 0.0720000224680862 z_k1_s0053
                                 - 0.166000008107006 z_k1_s0054
                                 - 0.275000254634938 z_k1_s0055
                                 - 0.183999405974437 z_k1_s0056
                                 - 0.166999577433242 z_k1_s0057
                                 - 0.0910000556567083 z_k1_s0060
                                 - 0.094000081546501 z_k1_s0061
                                 - 0.0640000216486658 z_k1_s0062
                                 - 0.292999775735148 z_k1_s0063
                                 - 0.188000362093094 z_k1_s0064
                                 - 0.0609999957588731 z_k1_s0065
                                 - 0.120000040591248 z_k1_s0066
                                 - 0.0840000284138739 z_k1_s0067
                                 - 0.153999062478118 z_k1_s0070
                                 - 0.0649997593467082 z_k1_s0071
                                 - 0.172999543171357 z_k1_s0072
                                 - 0.161999446137787 z_k1_s0073
                                 - 0.107999669025056 z_k1_s0074
                                 - 0.290998601174863 z_k1_s0075
                                 - 0.106999649767495 z_k1_s0076
                                 - 0.0879997430722224 z_k1_s0077
                                 - 0.364999530767547 z_k1_s0100
                                 - 0.505999395906491 z_k1_s0101
                                 - 0.424999574284061 z_k1_s0102
                                 - 1.25500028917912 z_k1_s0103
                                 - 1.11299636960886 z_k1_s0104
                                 - 0.899997265249639 z_k1_s0105
                                 - 1.19399822373014 z_k1_s0106
                                 - 1.01700144897952 z_k1_s0107
                                 - 0.4320007633071 z_k1_s0110
                                 - 0.296000084317677 z_k1_s0111
                                 - 0.247000204095878 z_k1_s0112
                                 - 0.323999793230057 z_k1_s0113
                                 - 0.405001300038107 z_k1_s0114
                                 - 0.205000173177147 z_k1_s0115
                                 - 0.898999214629006 z_k1_s0116
                                 - 0.30900016068064 z_k1_s0117
                                 - 0.323001172130256 z_k1_s0120
                                 - 0.106999957801508 z_k1_s0121
                                 - 0.274999798723468 z_k1_s0122
                                 - 0.241000067394109 z_k1_s0123
                                 - 0.488999714326483 z_k1_s0124
                                 - 0.310000179415693 z_k1_s0125
                                 - 0.215000068365702 z_k1_s0126
                                 - 0.522999445655842 z_k1_s0127
                                 - 0.139999986057192 z_k1_s0130
                                 - 0.2009999343221 z_k1_s0131
                                 - 0.0769999771114519 z_k1_s0132
                                 - 0.0839999611943079 z_k1_s0133
                                 - 0.384000692317045 z_k1_s0134
                                 - 0.166000013907821 z_k1_s0135
                                 - 0.256000461544697 z_k1_s0136
                                 - 0.213999339447268 z_k1_s0137
                                 - 0.864001934446115 z_k1_s0140
                                 - 0.232000821835584 z_k1_s0141
                                 - 0.617002326430437 z_k1_s0142
                                 - 0.577003748877644 z_k1_s0143
                                 - 0.410001295967552 z_k1_s0144
                                 - 0.426001180590139 z_k1_s0145
                                 - 0.910001659436235 z_k1_s0146
                                 - 0.500000363468683 z_k1_s0147
                                 - 0.205000072260005 z_k1_s0150
                                 - 0.46799924155737 z_k1_s0151
                                 - 0.188000104314546 z_k1_s0152
                                 - 0.213999968489944 z_k1_s0153
                                 - 0.286000824729651 z_k1_s0154
                                 - 0.158000098881375 z_k1_s0155
                                 - 0.289000509072932 z_k1_s0156
                                 - 0.300000054331709 z_k1_s0157
                                 - 0.34099979057012 z_k1_s0160
                                 - 0.206999629568128 z_k1_s0161
                                 - 0.675998633577294 z_k1_s0162
                                 - 0.288999583779998 z_k1_s0163
                                 - 0.224999665160913 z_k1_s0164
                                 - 0.521000146497973 z_k1_s0165
                                 - 0.521998225094721 z_k1_s0166
                                 - 0.337999316788647 z_k1_s0167
                                 - 0.368000498566265 z_k1_s0170
                                 - 0.732000963626363 z_k1_s0171
                                 - 0.170000132011553 z_k1_s0172
                                 - 0.162000064999221 z_k1_s0173
                                 - 0.165000090128846 z_k1_s0174
                                 - 0.425000976029125 z_k1_s0175
                                 - 0.169000123635011 z_k1_s0176
                                 - 0.3930007079798 z_k1_s0177
                                 - 0.20300043205159 z_k1_s1000
                                 - 0.153999940601569 z_k1_s1001
                                 - 0.460000341575629 z_k1_s1002
                                 - 0.248999705009865 z_k1_s1003
                                 - 0.191999199345493 z_k1_s1004
                                 - 0.17099980429472 z_k1_s1005
                                 - 0.265000114038211 z_k1_s1006
                                 - 0.20900058543722 z_k1_s1007
                                 - 0.807999239110824 z_k1_s1010
                                 - 0.496001046674181 z_k1_s1011
                                 - 1.16500233566599 z_k1_s1012
                                 - 1.60400411586676 z_k1_s1013
                                 - 1.0830036520204 z_k1_s1014
                                 - 0.376000597043047 z_k1_s1015
                                 - 0.421000765654722 z_k1_s1016
                                 - 0.812004124905576 z_k1_s1017
                                 - 0.44900121537938 z_k1_s1020
                                 - 0.4630023153987 z_k1_s1021
                                 - 0.492003138291701 z_k1_s1022
                                 - 0.388001255423303 z_k1_s1023
                                 - 0.509000106874106 z_k1_s1024
                                 - 1.05000098121595 z_k1_s1025
                                 - 0.27700086055107 z_k1_s1026
                                 - 0.448002592525018 z_k1_s1027
                                 - 1.08300407469965 z_k1_s1030
                                 - 0.894000859554908 z_k1_s1031
                                 - 0.569000871016911 z_k1_s1032
                                 - 1.74200013944393 z_k1_s1033
                                 - 1.37400424213851 z_k1_s1034
                                 - 0.550000946373248 z_k1_s1035
                                 - 0.881002232695169 z_k1_s1036
                                 - 1.27700418632556 z_k1_s1037
                                 - 0.492997753634395 z_k1_s1040
                                 - 0.338999783363901 z_k1_s1041
                                 - 0.358999683176864 z_k1_s1042
                                 - 0.341999788630835 z_k1_s1043
                                 - 0.366999967821872 z_k1_s1044
                                 - 0.920000262193606 z_k1_s1045
                                 - 0.73599777435621 z_k1_s1046
                                 - 0.502999124190103 z_k1_s1047
                                 - 1.22700265197458 z_k1_s1050
                                 - 1.0839987824668 z_k1_s1051
                                 - 0.743000001685946 z_k1_s1052
                                 - 0.469000333589355 z_k1_s1053
                                 - 0.607999603890952 z_k1_s1054
                                 - 0.552999563026315 z_k1_s1055
                                 - 1.46600236843912 z_k1_s1056
                                 - 2.03099854898524 z_k1_s1057
                                 - 0.404000240832069 z_k1_s1060
                                 - 0.384000329917664 z_k1_s1061
                                 - 1.07900124204538 z_k1_s1062
                                 - 0.646998502611735 z_k1_s1063
                                 - 0.294000002102452 z_k1_s1064
                                 - 0.467000275982613 z_k1_s1065
                                 - 0.906002911366256 z_k1_s1066
                                 - 0.677001526685034 z_k1_s1067
                                 - 0.145999994509339 z_k1_s1070
                                 - 0.108999873790567 z_k1_s1071
                                 - 0.217999967681074 z_k1_s1072
                                 - 0.163999987802273 z_k1_s1073
                                 - 0.522000025595022 z_k1_s1074
                                 - 0.215000005482242 z_k1_s1075
                                 - 0.345999308597656 z_k1_s1076
                                 - 0.480999808544513 z_k1_s1077
                                 - 0.798999332113682 z_k1_s1100
                                 - 0.458997940344373 z_k1_s1101
                                 - 0.607998693687182 z_k1_s1102
                                 - 0.185999812598726 z_k1_s1103
                                 - 0.406997904002925 z_k1_s1104
                                 - 0.431000517618226 z_k1_s1105
                                 - 0.473000402701917 z_k1_s1106
                                 - 0.387999117010972 z_k1_s1107
                                 - 0.370001015734845 z_k1_s1110
                                 - 0.367001134580456 z_k1_s1111
                                 - 0.110000187602055 z_k1_s1112
                                 - 0.262999768885563 z_k1_s1113
                                 - 0.275001644507156 z_k1_s1114
                                 - 0.53100012336312 z_k1_s1115
                                 - 0.247001186396837 z_k1_s1116
                                 - 0.188000389021811 z_k1_s1117
                                 - 0.929997077577986 z_k1_s1120
                                 - 0.186999438583899 z_k1_s1121
                                 - 0.218999567370994 z_k1_s1122
                                 - 0.591997202673333 z_k1_s1123
                                 - 0.22999951642303 z_k1_s1124
                                 - 0.208999641387261 z_k1_s1125
                                 - 0.353999634598911 z_k1_s1126
                                 - 0.325997769889288 z_k1_s1127
                                 - 1.03600121864193 z_k1_s1130
                                 - 0.315000730696029 z_k1_s1131
                                 - 0.551000305126316 z_k1_s1132
                                 - 0.641999331680154 z_k1_s1133
                                 - 0.389001071571051 z_k1_s1134
                                 - 0.506002739032868 z_k1_s1135
                                 - 0.701002556795581 z_k1_s1136
                                 - 0.302000742845182 z_k1_s1137
                                 - 0.647002333922168 z_k1_s1140
                                 - 0.937005267929896 z_k1_s1141
                                 - 0.523001481526936 z_k1_s1142
                                 - 0.950999948852233 z_k1_s1143
                                 - 0.576001425105926 z_k1_s1144
                                 - 0.602001455333524 z_k1_s1145
                                 - 1.12300562582005 z_k1_s1146
                                 - 0.779000409902361 z_k1_s1147
                                 - 0.0840000260236421 z_k1_s1150
                                 - 0.143000244502415 z_k1_s1151
                                 - 0.391001122135169 z_k1_s1152
                                 - 0.141000243882804 z_k1_s1153
                                 - 0.301001094252695 z_k1_s1154
                                 - 0.343001107264516 z_k1_s1155
                                 - 0.361001112841011 z_k1_s1156
                                 - 0.238000073733653 z_k1_s1157
                                 - 0.347001350818288 z_k1_s1160
                                 - 0.230000357002584 z_k1_s1161
                                 - 0.402001649927956 z_k1_s1162
                                 - 0.33099974431114 z_k1_s1163
                                 - 0.257001531730851 z_k1_s1164
                                 - 0.287999421079797 z_k1_s1165
                                 - 0.274001087891133 z_k1_s1166
                                 - 0.329001387000801 z_k1_s1167
                                 - 0.597003682942837 z_k1_s1170
                                 - 0.277001430856732 z_k1_s1171
                                 - 0.179000858740998 z_k1_s1172
                                 - 0.584002572668833 z_k1_s1173
                                 - 0.186000959778161 z_k1_s1174
                                 - 0.422002561021341 z_k1_s1175
                                 - 0.386000998424578 z_k1_s1176
                                 - 0.177000990331247 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.658619 w_k1_ADM_s000
                                 + 0.625065 w_k1_AXP_s000
                                 + 0.594811 w_k1_DIS_s000
                                 + 0.857968 w_k1_FDX_s000
                                 + 0.643132 w_k1_IBM_s000
                                 + 0.860909 w_k1_INTC_s000
                                 + 0.551929 w_k1_KO_s000
                                 + 0.529308 w_k1_MAT_s000
                                 + 0.76744 w_k1_MSFT_s000
                                 + 0.744109 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.849936 w_k1_ADM_s000
                                 + 0.705087 w_k1_AXP_s000
                                 + 0.670913 w_k1_DIS_s000
                                 + 0.852803 w_k1_FDX_s000
                                 + 0.925611 w_k1_IBM_s000
                                 + 0.86174 w_k1_INTC_s000
                                 + 0.810513 w_k1_KO_s000
                                 + 0.947387 w_k1_MAT_s000
                                 + 0.854088 w_k1_MSFT_s000
                                 + 0.772549 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.74899 w_k1_ADM_s000
                                 + 0.871186 w_k1_AXP_s000
                                 + 1.00867 w_k1_DIS_s000
                                 + 0.992158 w_k1_FDX_s000
                                 + 0.843754 w_k1_IBM_s000
                                 + 1.19563 w_k1_INTC_s000
                                 + 0.895648 w_k1_KO_s000
                                 + 0.777035 w_k1_MAT_s000
                                 + 1.07709 w_k1_MSFT_s000
                                 + 0.950338 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.07419 w_k1_ADM_s000
                                 + 1.4421 w_k1_AXP_s000 + 1.77237 w_k1_DIS_s000
                                 + 1.30496 w_k1_FDX_s000
                                 + 1.64579 w_k1_IBM_s000
                                 + 1.1259 w_k1_INTC_s000 + 1.28472 w_k1_KO_s000
                                 + 1.81727 w_k1_MAT_s000
                                 + 1.21042 w_k1_MSFT_s000
                                 + 1.08077 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.85402 w_k1_ADM_s000
                                 + 1.49468 w_k1_AXP_s000
                                 + 1.36431 w_k1_DIS_s000
                                 + 1.33979 w_k1_FDX_s000
                                 + 2.13127 w_k1_IBM_s000
                                 + 1.53545 w_k1_INTC_s000
                                 + 1.62315 w_k1_KO_s000 + 1.97305 w_k1_MAT_s000
                                 + 1.24855 w_k1_MSFT_s000
                                 + 1.34074 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.850557 w_k1_ADM_s000
                                 + 1.13177 w_k1_AXP_s000
                                 + 0.756743 w_k1_DIS_s000
                                 + 0.976939 w_k1_FDX_s000
                                 + 0.759554 w_k1_IBM_s000
                                 + 1.00615 w_k1_INTC_s000
                                 + 0.766306 w_k1_KO_s000
                                 + 0.94458 w_k1_MAT_s000
                                 + 0.843342 w_k1_MSFT_s000
                                 + 0.89137 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.26278 w_k1_ADM_s000
                                 + 1.23626 w_k1_AXP_s000
                                 + 1.13459 w_k1_DIS_s000
                                 + 1.11108 w_k1_FDX_s000
                                 + 1.30148 w_k1_IBM_s000
                                 + 1.00647 w_k1_INTC_s000
                                 + 1.33358 w_k1_KO_s000 + 1.29555 w_k1_MAT_s000
                                 + 1.08891 w_k1_MSFT_s000
                                 + 1.20777 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.32769 w_k1_ADM_s000
                                 + 1.40507 w_k1_AXP_s000
                                 + 1.09259 w_k1_DIS_s000
                                 + 1.26793 w_k1_FDX_s000
                                 + 0.883033 w_k1_IBM_s000
                                 + 0.93913 w_k1_INTC_s000
                                 + 0.888407 w_k1_KO_s000
                                 + 1.34314 w_k1_MAT_s000
                                 + 0.969095 w_k1_MSFT_s000
                                 + 0.994618 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.45632 w_k1_ADM_s001
                                 + 1.1694 w_k1_AXP_s001 + 1.43653 w_k1_DIS_s001
                                 + 1.22249 w_k1_FDX_s001
                                 + 0.766063 w_k1_IBM_s001
                                 + 0.79955 w_k1_INTC_s001
                                 + 0.963251 w_k1_KO_s001
                                 + 0.553762 w_k1_MAT_s001
                                 + 0.987371 w_k1_MSFT_s001
                                 + 1.10087 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.02316 w_k1_ADM_s001
                                 + 0.935847 w_k1_AXP_s001
                                 + 1.20515 w_k1_DIS_s001
                                 + 0.973415 w_k1_FDX_s001
                                 + 1.3636 w_k1_IBM_s001
                                 + 0.921575 w_k1_INTC_s001
                                 + 1.10465 w_k1_KO_s001
                                 + 0.888751 w_k1_MAT_s001
                                 + 0.988918 w_k1_MSFT_s001
                                 + 1.14008 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.79329 w_k1_ADM_s001
                                 + 0.859853 w_k1_AXP_s001
                                 + 1.03501 w_k1_DIS_s001
                                 + 1.00104 w_k1_FDX_s001
                                 + 0.789096 w_k1_IBM_s001
                                 + 0.888576 w_k1_INTC_s001
                                 + 0.79238 w_k1_KO_s001
                                 + 0.530883 w_k1_MAT_s001
                                 + 0.922486 w_k1_MSFT_s001
                                 + 0.937479 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.662543 w_k1_ADM_s001
                                 + 0.726061 w_k1_AXP_s001
                                 + 0.718542 w_k1_DIS_s001
                                 + 0.775753 w_k1_FDX_s001
                                 + 0.800146 w_k1_IBM_s001
                                 + 0.824946 w_k1_INTC_s001
                                 + 0.814096 w_k1_KO_s001
                                 + 0.511526 w_k1_MAT_s001
                                 + 0.675534 w_k1_MSFT_s001
                                 + 0.881253 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.65565 w_k1_ADM_s001
                                 + 1.33312 w_k1_AXP_s001
                                 + 1.62673 w_k1_DIS_s001 + 1.2946 w_k1_FDX_s001
                                 + 1.37897 w_k1_IBM_s001
                                 + 0.774054 w_k1_INTC_s001
                                 + 1.22593 w_k1_KO_s001 + 1.04729 w_k1_MAT_s001
                                 + 1.04443 w_k1_MSFT_s001
                                 + 1.11661 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.976503 w_k1_ADM_s001
                                 + 1.39032 w_k1_AXP_s001
                                 + 1.01511 w_k1_DIS_s001
                                 + 1.09751 w_k1_FDX_s001
                                 + 1.12673 w_k1_IBM_s001
                                 + 0.928988 w_k1_INTC_s001
                                 + 0.905679 w_k1_KO_s001
                                 + 1.15932 w_k1_MAT_s001
                                 + 0.953409 w_k1_MSFT_s001
                                 + 1.18864 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.00018 w_k1_ADM_s001
                                 + 1.1029 w_k1_AXP_s001
                                 + 0.667994 w_k1_DIS_s001
                                 + 0.937292 w_k1_FDX_s001
                                 + 0.878853 w_k1_IBM_s001
                                 + 0.757227 w_k1_INTC_s001
                                 + 0.827194 w_k1_KO_s001 + 1.015 w_k1_MAT_s001
                                 + 0.807497 w_k1_MSFT_s001
                                 + 0.985209 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.842246 w_k1_ADM_s001
                                 + 1.18058 w_k1_AXP_s001
                                 + 1.22471 w_k1_DIS_s001
                                 + 1.03565 w_k1_FDX_s001
                                 + 1.94366 w_k1_IBM_s001
                                 + 0.781749 w_k1_INTC_s001
                                 + 1.15347 w_k1_KO_s001 + 1.80173 w_k1_MAT_s001
                                 + 0.82697 w_k1_MSFT_s001
                                 + 1.17339 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.951906 w_k1_ADM_s002
                                 + 1.29348 w_k1_AXP_s002 + 1.4159 w_k1_DIS_s002
                                 + 1.13893 w_k1_FDX_s002
                                 + 1.30261 w_k1_IBM_s002
                                 + 1.22862 w_k1_INTC_s002
                                 + 1.16399 w_k1_KO_s002 + 1.29657 w_k1_MAT_s002
                                 + 1.01799 w_k1_MSFT_s002
                                 + 1.11569 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.669129 w_k1_ADM_s002
                                 + 0.711258 w_k1_AXP_s002
                                 + 0.814826 w_k1_DIS_s002
                                 + 0.860746 w_k1_FDX_s002
                                 + 0.713583 w_k1_IBM_s002
                                 + 1.08113 w_k1_INTC_s002
                                 + 0.724422 w_k1_KO_s002
                                 + 0.54451 w_k1_MAT_s002
                                 + 0.904696 w_k1_MSFT_s002
                                 + 0.901236 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.738688 w_k1_ADM_s002
                                 + 0.85788 w_k1_AXP_s002
                                 + 0.631399 w_k1_DIS_s002
                                 + 0.726127 w_k1_FDX_s002
                                 + 0.868571 w_k1_IBM_s002
                                 + 0.958461 w_k1_INTC_s002
                                 + 0.764178 w_k1_KO_s002
                                 + 0.727574 w_k1_MAT_s002
                                 + 0.671335 w_k1_MSFT_s002
                                 + 0.826793 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.995635 w_k1_ADM_s002
                                 + 0.922829 w_k1_AXP_s002
                                 + 0.851648 w_k1_DIS_s002
                                 + 1.03575 w_k1_FDX_s002
                                 + 0.727031 w_k1_IBM_s002
                                 + 1.12371 w_k1_INTC_s002
                                 + 0.785776 w_k1_KO_s002
                                 + 0.781522 w_k1_MAT_s002
                                 + 1.1044 w_k1_MSFT_s002
                                 + 0.952121 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.701119 w_k1_ADM_s002
                                 + 0.89091 w_k1_AXP_s002
                                 + 0.847501 w_k1_DIS_s002
                                 + 0.917463 w_k1_FDX_s002
                                 + 0.987646 w_k1_IBM_s002
                                 + 1.23657 w_k1_INTC_s002
                                 + 0.890342 w_k1_KO_s002
                                 + 1.16896 w_k1_MAT_s002
                                 + 0.985382 w_k1_MSFT_s002
                                 + 0.936114 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.01624 w_k1_ADM_s002
                                 + 1.37453 w_k1_AXP_s002
                                 + 1.04474 w_k1_DIS_s002
                                 + 1.04764 w_k1_FDX_s002
                                 + 1.21169 w_k1_IBM_s002
                                 + 1.18061 w_k1_INTC_s002
                                 + 0.944451 w_k1_KO_s002
                                 + 0.981479 w_k1_MAT_s002
                                 + 0.981661 w_k1_MSFT_s002
                                 + 0.936316 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.952612 w_k1_ADM_s002
                                 + 0.838886 w_k1_AXP_s002
                                 + 1.11913 w_k1_DIS_s002
                                 + 1.00226 w_k1_FDX_s002
                                 + 1.07836 w_k1_IBM_s002
                                 + 1.16615 w_k1_INTC_s002
                                 + 1.00699 w_k1_KO_s002
                                 + 0.663101 w_k1_MAT_s002
                                 + 0.961531 w_k1_MSFT_s002
                                 + 1.01255 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.99286 w_k1_ADM_s002
                                 + 1.57605 w_k1_AXP_s002
                                 + 1.74922 w_k1_DIS_s002
                                 + 1.34378 w_k1_FDX_s002
                                 + 1.50676 w_k1_IBM_s002
                                 + 1.55845 w_k1_INTC_s002 + 1.2622 w_k1_KO_s002
                                 + 1.20301 w_k1_MAT_s002
                                 + 1.3973 w_k1_MSFT_s002 + 1.13335 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.851901 w_k1_ADM_s003
                                 + 0.642788 w_k1_AXP_s003
                                 + 0.68708 w_k1_DIS_s003
                                 + 0.789409 w_k1_FDX_s003
                                 + 0.810347 w_k1_IBM_s003
                                 + 0.914988 w_k1_INTC_s003
                                 + 0.619318 w_k1_KO_s003
                                 + 0.623599 w_k1_MAT_s003
                                 + 0.899807 w_k1_MSFT_s003
                                 + 0.800443 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.13573 w_k1_ADM_s003
                                 + 1.34649 w_k1_AXP_s003
                                 + 1.25804 w_k1_DIS_s003
                                 + 1.30683 w_k1_FDX_s003
                                 + 0.720148 w_k1_IBM_s003
                                 + 1.11773 w_k1_INTC_s003
                                 + 0.899273 w_k1_KO_s003
                                 + 0.688688 w_k1_MAT_s003
                                 + 1.08675 w_k1_MSFT_s003
                                 + 0.906633 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.949513 w_k1_ADM_s003
                                 + 0.749065 w_k1_AXP_s003
                                 + 0.756147 w_k1_DIS_s003
                                 + 0.780408 w_k1_FDX_s003
                                 + 1.07474 w_k1_IBM_s003
                                 + 0.800861 w_k1_INTC_s003
                                 + 0.861776 w_k1_KO_s003
                                 + 0.710605 w_k1_MAT_s003
                                 + 0.806377 w_k1_MSFT_s003
                                 + 0.932846 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.07265 w_k1_ADM_s003
                                 + 1.01605 w_k1_AXP_s003
                                 + 0.962899 w_k1_DIS_s003
                                 + 1.04477 w_k1_FDX_s003
                                 + 0.790553 w_k1_IBM_s003
                                 + 0.866974 w_k1_INTC_s003
                                 + 0.845905 w_k1_KO_s003
                                 + 0.759876 w_k1_MAT_s003
                                 + 0.839861 w_k1_MSFT_s003
                                 + 0.873653 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.10643 w_k1_ADM_s003
                                 + 0.843438 w_k1_AXP_s003
                                 + 0.879886 w_k1_DIS_s003
                                 + 0.86939 w_k1_FDX_s003
                                 + 0.87102 w_k1_IBM_s003
                                 + 1.10739 w_k1_INTC_s003
                                 + 0.899064 w_k1_KO_s003
                                 + 0.651198 w_k1_MAT_s003
                                 + 1.01527 w_k1_MSFT_s003
                                 + 0.842748 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.99477 w_k1_ADM_s003
                                 + 1.017 w_k1_AXP_s003 + 1.14127 w_k1_DIS_s003
                                 + 0.95496 w_k1_FDX_s003
                                 + 1.30627 w_k1_IBM_s003
                                 + 1.00862 w_k1_INTC_s003
                                 + 1.05722 w_k1_KO_s003 + 1.22131 w_k1_MAT_s003
                                 + 1.01052 w_k1_MSFT_s003
                                 + 0.916582 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.10782 w_k1_ADM_s003
                                 + 1.48355 w_k1_AXP_s003
                                 + 1.07928 w_k1_DIS_s003
                                 + 1.15409 w_k1_FDX_s003
                                 + 1.08901 w_k1_IBM_s003
                                 + 1.30968 w_k1_INTC_s003
                                 + 1.09366 w_k1_KO_s003 + 1.33039 w_k1_MAT_s003
                                 + 1.07669 w_k1_MSFT_s003
                                 + 1.06161 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.952988 w_k1_ADM_s003
                                 + 0.733657 w_k1_AXP_s003
                                 + 0.668598 w_k1_DIS_s003
                                 + 0.826149 w_k1_FDX_s003
                                 + 0.633403 w_k1_IBM_s003
                                 + 1.00958 w_k1_INTC_s003
                                 + 0.69402 w_k1_KO_s003
                                 + 0.524724 w_k1_MAT_s003
                                 + 0.89665 w_k1_MSFT_s003
                                 + 0.841259 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.03221 w_k1_ADM_s004
                                 + 1.10615 w_k1_AXP_s004
                                 + 1.06797 w_k1_DIS_s004
                                 + 1.01107 w_k1_FDX_s004 + 1.3122 w_k1_IBM_s004
                                 + 1.16866 w_k1_INTC_s004
                                 + 1.29288 w_k1_KO_s004 + 1.34661 w_k1_MAT_s004
                                 + 1.00936 w_k1_MSFT_s004
                                 + 1.15817 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.00632 w_k1_ADM_s004
                                 + 0.771784 w_k1_AXP_s004
                                 + 0.669029 w_k1_DIS_s004
                                 + 0.852365 w_k1_FDX_s004
                                 + 0.601566 w_k1_IBM_s004
                                 + 1.24589 w_k1_INTC_s004
                                 + 0.835505 w_k1_KO_s004
                                 + 0.404716 w_k1_MAT_s004
                                 + 1.01153 w_k1_MSFT_s004
                                 + 0.93645 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.00072 w_k1_ADM_s004
                                 + 1.14119 w_k1_AXP_s004
                                 + 0.944179 w_k1_DIS_s004
                                 + 1.10991 w_k1_FDX_s004
                                 + 0.67172 w_k1_IBM_s004
                                 + 1.19313 w_k1_INTC_s004
                                 + 0.864182 w_k1_KO_s004
                                 + 0.988118 w_k1_MAT_s004
                                 + 1.14273 w_k1_MSFT_s004
                                 + 0.948677 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.826206 w_k1_ADM_s004
                                 + 0.773021 w_k1_AXP_s004
                                 + 0.800553 w_k1_DIS_s004
                                 + 0.804921 w_k1_FDX_s004
                                 + 1.02469 w_k1_IBM_s004
                                 + 1.28513 w_k1_INTC_s004
                                 + 0.947499 w_k1_KO_s004
                                 + 0.761596 w_k1_MAT_s004
                                 + 0.993164 w_k1_MSFT_s004
                                 + 0.983714 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.19889 w_k1_ADM_s004
                                 + 1.36389 w_k1_AXP_s004
                                 + 1.46806 w_k1_DIS_s004
                                 + 1.21853 w_k1_FDX_s004
                                 + 1.13885 w_k1_IBM_s004
                                 + 1.46961 w_k1_INTC_s004
                                 + 1.53988 w_k1_KO_s004 + 1.07331 w_k1_MAT_s004
                                 + 1.49405 w_k1_MSFT_s004
                                 + 1.45157 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.894275 w_k1_ADM_s004
                                 + 0.832841 w_k1_AXP_s004
                                 + 0.65691 w_k1_DIS_s004
                                 + 0.791317 w_k1_FDX_s004
                                 + 0.953668 w_k1_IBM_s004
                                 + 1.07198 w_k1_INTC_s004
                                 + 0.896017 w_k1_KO_s004
                                 + 0.782761 w_k1_MAT_s004
                                 + 1.00043 w_k1_MSFT_s004
                                 + 0.946463 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.891752 w_k1_ADM_s004
                                 + 1.03572 w_k1_AXP_s004
                                 + 0.986023 w_k1_DIS_s004
                                 + 0.896762 w_k1_FDX_s004
                                 + 1.22425 w_k1_IBM_s004
                                 + 0.94838 w_k1_INTC_s004
                                 + 1.06767 w_k1_KO_s004 + 1.47081 w_k1_MAT_s004
                                 + 0.856268 w_k1_MSFT_s004
                                 + 1.00335 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.791824 w_k1_ADM_s004
                                 + 0.898084 w_k1_AXP_s004
                                 + 0.839874 w_k1_DIS_s004
                                 + 0.884845 w_k1_FDX_s004
                                 + 0.728485 w_k1_IBM_s004
                                 + 1.25984 w_k1_INTC_s004
                                 + 1.01766 w_k1_KO_s004
                                 + 0.775247 w_k1_MAT_s004
                                 + 1.13382 w_k1_MSFT_s004
                                 + 1.05202 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.06008 w_k1_ADM_s005
                                 + 0.980905 w_k1_AXP_s005
                                 + 1.32097 w_k1_DIS_s005
                                 + 1.19921 w_k1_FDX_s005
                                 + 0.942163 w_k1_IBM_s005
                                 + 1.36544 w_k1_INTC_s005
                                 + 1.09804 w_k1_KO_s005
                                 + 0.838275 w_k1_MAT_s005
                                 + 1.3035 w_k1_MSFT_s005 + 1.05017 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.700542 w_k1_ADM_s005
                                 + 0.538716 w_k1_AXP_s005
                                 + 0.62643 w_k1_DIS_s005
                                 + 0.681828 w_k1_FDX_s005
                                 + 0.801279 w_k1_IBM_s005
                                 + 0.903629 w_k1_INTC_s005
                                 + 0.801483 w_k1_KO_s005
                                 + 0.633546 w_k1_MAT_s005
                                 + 0.881206 w_k1_MSFT_s005
                                 + 0.798598 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.46897 w_k1_ADM_s005
                                 + 1.58453 w_k1_AXP_s005
                                 + 1.69823 w_k1_DIS_s005
                                 + 1.38266 w_k1_FDX_s005
                                 + 1.27071 w_k1_IBM_s005
                                 + 1.30445 w_k1_INTC_s005
                                 + 1.32509 w_k1_KO_s005 + 1.30532 w_k1_MAT_s005
                                 + 1.65901 w_k1_MSFT_s005
                                 + 1.26546 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.884959 w_k1_ADM_s005
                                 + 0.628774 w_k1_AXP_s005
                                 + 0.635768 w_k1_DIS_s005
                                 + 0.808358 w_k1_FDX_s005
                                 + 0.59314 w_k1_IBM_s005
                                 + 0.96996 w_k1_INTC_s005
                                 + 0.704062 w_k1_KO_s005
                                 + 0.455787 w_k1_MAT_s005
                                 + 0.84324 w_k1_MSFT_s005
                                 + 0.717984 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.32795 w_k1_ADM_s005
                                 + 1.31617 w_k1_AXP_s005
                                 + 1.26527 w_k1_DIS_s005
                                 + 1.08131 w_k1_FDX_s005 + 1.4233 w_k1_IBM_s005
                                 + 1.13669 w_k1_INTC_s005
                                 + 1.10344 w_k1_KO_s005 + 1.95261 w_k1_MAT_s005
                                 + 1.08272 w_k1_MSFT_s005
                                 + 1.05936 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.07086 w_k1_ADM_s005
                                 + 0.886307 w_k1_AXP_s005
                                 + 0.943977 w_k1_DIS_s005
                                 + 1.01393 w_k1_FDX_s005
                                 + 0.795956 w_k1_IBM_s005
                                 + 1.03588 w_k1_INTC_s005
                                 + 0.82263 w_k1_KO_s005
                                 + 0.703117 w_k1_MAT_s005
                                 + 1.1769 w_k1_MSFT_s005
                                 + 0.990275 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.880169 w_k1_ADM_s005
                                 + 0.913379 w_k1_AXP_s005
                                 + 1.06222 w_k1_DIS_s005
                                 + 0.906204 w_k1_FDX_s005
                                 + 1.22106 w_k1_IBM_s005
                                 + 1.15532 w_k1_INTC_s005
                                 + 1.31143 w_k1_KO_s005 + 1.17194 w_k1_MAT_s005
                                 + 1.18675 w_k1_MSFT_s005
                                 + 1.04471 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.04277 w_k1_ADM_s005
                                 + 0.907767 w_k1_AXP_s005
                                 + 0.727446 w_k1_DIS_s005
                                 + 0.744852 w_k1_FDX_s005
                                 + 1.19793 w_k1_IBM_s005
                                 + 0.890079 w_k1_INTC_s005
                                 + 0.989225 w_k1_KO_s005
                                 + 1.27861 w_k1_MAT_s005
                                 + 0.87031 w_k1_MSFT_s005
                                 + 0.842963 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.42359 w_k1_ADM_s006
                                 + 1.23262 w_k1_AXP_s006
                                 + 1.46693 w_k1_DIS_s006
                                 + 1.17047 w_k1_FDX_s006
                                 + 1.53836 w_k1_IBM_s006
                                 + 1.15028 w_k1_INTC_s006
                                 + 1.62217 w_k1_KO_s006 + 1.38815 w_k1_MAT_s006
                                 + 0.969859 w_k1_MSFT_s006
                                 + 1.22047 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.06841 w_k1_ADM_s006
                                 + 1.22821 w_k1_AXP_s006
                                 + 0.978958 w_k1_DIS_s006
                                 + 1.09329 w_k1_FDX_s006
                                 + 0.80011 w_k1_IBM_s006
                                 + 1.18408 w_k1_INTC_s006
                                 + 0.948358 w_k1_KO_s006
                                 + 0.704813 w_k1_MAT_s006
                                 + 0.900729 w_k1_MSFT_s006
                                 + 1.00147 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.67452 w_k1_ADM_s006
                                 + 1.56305 w_k1_AXP_s006
                                 + 1.93006 w_k1_DIS_s006
                                 + 1.52928 w_k1_FDX_s006
                                 + 1.21336 w_k1_IBM_s006
                                 + 1.26201 w_k1_INTC_s006
                                 + 1.36489 w_k1_KO_s006 + 1.20883 w_k1_MAT_s006
                                 + 1.16881 w_k1_MSFT_s006
                                 + 1.27511 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.789613 w_k1_ADM_s006
                                 + 0.730297 w_k1_AXP_s006
                                 + 0.749891 w_k1_DIS_s006
                                 + 0.794139 w_k1_FDX_s006
                                 + 0.718141 w_k1_IBM_s006
                                 + 0.857938 w_k1_INTC_s006
                                 + 0.864028 w_k1_KO_s006
                                 + 0.57279 w_k1_MAT_s006
                                 + 0.68797 w_k1_MSFT_s006
                                 + 0.762609 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.10293 w_k1_ADM_s006
                                 + 0.884284 w_k1_AXP_s006
                                 + 0.846616 w_k1_DIS_s006
                                 + 0.796003 w_k1_FDX_s006
                                 + 1.29477 w_k1_IBM_s006
                                 + 0.640787 w_k1_INTC_s006
                                 + 1.1733 w_k1_KO_s006 + 1.11062 w_k1_MAT_s006
                                 + 0.744214 w_k1_MSFT_s006
                                 + 0.932502 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.02045 w_k1_ADM_s006
                                 + 0.78763 w_k1_AXP_s006
                                 + 1.58374 w_k1_DIS_s006 + 1.1172 w_k1_FDX_s006
                                 + 0.912459 w_k1_IBM_s006
                                 + 1.14113 w_k1_INTC_s006
                                 + 1.31882 w_k1_KO_s006
                                 + 0.644011 w_k1_MAT_s006
                                 + 1.16424 w_k1_MSFT_s006
                                 + 1.10095 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.14434 w_k1_ADM_s006
                                 + 1.35019 w_k1_AXP_s006
                                 + 1.02287 w_k1_DIS_s006
                                 + 1.06814 w_k1_FDX_s006 + 1.1982 w_k1_IBM_s006
                                 + 0.927569 w_k1_INTC_s006
                                 + 1.16931 w_k1_KO_s006 + 1.37255 w_k1_MAT_s006
                                 + 0.872812 w_k1_MSFT_s006
                                 + 1.0006 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.38275 w_k1_ADM_s006
                                 + 1.12046 w_k1_AXP_s006
                                 + 1.39717 w_k1_DIS_s006
                                 + 1.30266 w_k1_FDX_s006
                                 + 0.979537 w_k1_IBM_s006
                                 + 0.986911 w_k1_INTC_s006
                                 + 1.15993 w_k1_KO_s006
                                 + 0.994621 w_k1_MAT_s006
                                 + 0.911067 w_k1_MSFT_s006
                                 + 0.975903 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.43941 w_k1_ADM_s007
                                 + 1.46493 w_k1_AXP_s007
                                 + 1.46184 w_k1_DIS_s007
                                 + 1.29825 w_k1_FDX_s007
                                 + 1.23857 w_k1_IBM_s007
                                 + 1.30822 w_k1_INTC_s007
                                 + 1.42586 w_k1_KO_s007 + 1.36375 w_k1_MAT_s007
                                 + 1.19791 w_k1_MSFT_s007
                                 + 1.22214 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.934187 w_k1_ADM_s007
                                 + 1.41289 w_k1_AXP_s007
                                 + 0.815681 w_k1_DIS_s007
                                 + 0.842446 w_k1_FDX_s007
                                 + 1.17375 w_k1_IBM_s007
                                 + 0.959396 w_k1_INTC_s007
                                 + 1.04794 w_k1_KO_s007 + 1.60797 w_k1_MAT_s007
                                 + 0.958187 w_k1_MSFT_s007
                                 + 0.994268 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.798776 w_k1_ADM_s007
                                 + 0.82031 w_k1_AXP_s007
                                 + 0.70301 w_k1_DIS_s007
                                 + 0.669712 w_k1_FDX_s007
                                 + 1.00821 w_k1_IBM_s007
                                 + 0.89421 w_k1_INTC_s007
                                 + 0.945002 w_k1_KO_s007
                                 + 0.924257 w_k1_MAT_s007
                                 + 0.823743 w_k1_MSFT_s007
                                 + 0.921497 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.02094 w_k1_ADM_s007
                                 + 1.28193 w_k1_AXP_s007
                                 + 0.912821 w_k1_DIS_s007
                                 + 0.890318 w_k1_FDX_s007
                                 + 1.71343 w_k1_IBM_s007
                                 + 1.01584 w_k1_INTC_s007
                                 + 1.19748 w_k1_KO_s007 + 1.82039 w_k1_MAT_s007
                                 + 0.994582 w_k1_MSFT_s007
                                 + 1.18763 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.977175 w_k1_ADM_s007
                                 + 0.95118 w_k1_AXP_s007
                                 + 1.23504 w_k1_DIS_s007 + 1.145 w_k1_FDX_s007
                                 + 0.86021 w_k1_IBM_s007
                                 + 1.34688 w_k1_INTC_s007
                                 + 1.07248 w_k1_KO_s007 + 0.92211 w_k1_MAT_s007
                                 + 1.25408 w_k1_MSFT_s007
                                 + 1.11419 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.912257 w_k1_ADM_s007
                                 + 1.12358 w_k1_AXP_s007
                                 + 0.794975 w_k1_DIS_s007
                                 + 1.02075 w_k1_FDX_s007
                                 + 0.711587 w_k1_IBM_s007
                                 + 1.06015 w_k1_INTC_s007
                                 + 0.78262 w_k1_KO_s007
                                 + 0.914591 w_k1_MAT_s007
                                 + 0.942051 w_k1_MSFT_s007
                                 + 0.963822 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.0079 w_k1_ADM_s007
                                 + 1.05266 w_k1_AXP_s007
                                 + 1.06163 w_k1_DIS_s007
                                 + 1.08985 w_k1_FDX_s007
                                 + 1.14272 w_k1_IBM_s007
                                 + 0.987703 w_k1_INTC_s007
                                 + 1.17589 w_k1_KO_s007 + 1.25657 w_k1_MAT_s007
                                 + 1.15238 w_k1_MSFT_s007
                                 + 1.00602 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.0729 w_k1_ADM_s007
                                 + 0.989804 w_k1_AXP_s007
                                 + 0.795423 w_k1_DIS_s007
                                 + 0.945039 w_k1_FDX_s007
                                 + 0.936467 w_k1_IBM_s007
                                 + 1.16503 w_k1_INTC_s007
                                 + 0.90875 w_k1_KO_s007 + 1.03939 w_k1_MAT_s007
                                 + 1.02543 w_k1_MSFT_s007
                                 + 0.961791 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.645016 w_k1_ADM_s010
                                 + 0.597624 w_k1_AXP_s010
                                 + 0.560613 w_k1_DIS_s010
                                 + 0.614869 w_k1_FDX_s010
                                 + 1.00942 w_k1_IBM_s010
                                 + 0.687945 w_k1_INTC_s010
                                 + 0.817509 w_k1_KO_s010
                                 + 0.982865 w_k1_MAT_s010
                                 + 0.736149 w_k1_MSFT_s010
                                 + 0.80619 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.4474 w_k1_ADM_s010
                                 + 1.0354 w_k1_AXP_s010 + 1.14311 w_k1_DIS_s010
                                 + 1.20136 w_k1_FDX_s010
                                 + 0.697296 w_k1_IBM_s010
                                 + 1.1215 w_k1_INTC_s010 + 1.05078 w_k1_KO_s010
                                 + 0.549499 w_k1_MAT_s010
                                 + 1.16079 w_k1_MSFT_s010
                                 + 1.09098 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.82325 w_k1_ADM_s010
                                 + 0.630965 w_k1_AXP_s010
                                 + 0.495412 w_k1_DIS_s010
                                 + 0.684136 w_k1_FDX_s010
                                 + 0.671712 w_k1_IBM_s010
                                 + 0.923392 w_k1_INTC_s010
                                 + 0.659726 w_k1_KO_s010
                                 + 0.632969 w_k1_MAT_s010
                                 + 0.745117 w_k1_MSFT_s010
                                 + 0.784417 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.02428 w_k1_ADM_s010
                                 + 0.81906 w_k1_AXP_s010
                                 + 0.787749 w_k1_DIS_s010
                                 + 0.896967 w_k1_FDX_s010
                                 + 0.711656 w_k1_IBM_s010
                                 + 0.820258 w_k1_INTC_s010
                                 + 0.833053 w_k1_KO_s010
                                 + 0.794877 w_k1_MAT_s010
                                 + 0.927965 w_k1_MSFT_s010
                                 + 1.00201 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.20045 w_k1_ADM_s010
                                 + 1.29844 w_k1_AXP_s010 + 1.4658 w_k1_DIS_s010
                                 + 1.33038 w_k1_FDX_s010
                                 + 1.07475 w_k1_IBM_s010
                                 + 1.24652 w_k1_INTC_s010
                                 + 1.23964 w_k1_KO_s010 + 1.31507 w_k1_MAT_s010
                                 + 1.19405 w_k1_MSFT_s010
                                 + 1.29764 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.08487 w_k1_ADM_s010
                                 + 1.03681 w_k1_AXP_s010
                                 + 0.870942 w_k1_DIS_s010
                                 + 0.99995 w_k1_FDX_s010
                                 + 0.976594 w_k1_IBM_s010
                                 + 1.10752 w_k1_INTC_s010
                                 + 0.979274 w_k1_KO_s010
                                 + 1.05119 w_k1_MAT_s010
                                 + 0.958773 w_k1_MSFT_s010
                                 + 0.960316 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.873328 w_k1_ADM_s010
                                 + 0.860871 w_k1_AXP_s010
                                 + 0.799849 w_k1_DIS_s010
                                 + 0.73261 w_k1_FDX_s010
                                 + 1.29271 w_k1_IBM_s010
                                 + 1.06205 w_k1_INTC_s010
                                 + 1.04904 w_k1_KO_s010 + 1.00135 w_k1_MAT_s010
                                 + 0.864505 w_k1_MSFT_s010
                                 + 0.944548 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.4104 w_k1_ADM_s010
                                 + 1.28365 w_k1_AXP_s010
                                 + 1.15527 w_k1_DIS_s010
                                 + 1.09666 w_k1_FDX_s010
                                 + 1.36403 w_k1_IBM_s010
                                 + 1.17491 w_k1_INTC_s010
                                 + 1.19783 w_k1_KO_s010 + 1.63223 w_k1_MAT_s010
                                 + 1.0993 w_k1_MSFT_s010 + 1.1649 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.903595 w_k1_ADM_s011
                                 + 0.849555 w_k1_AXP_s011
                                 + 0.788761 w_k1_DIS_s011
                                 + 0.81786 w_k1_FDX_s011
                                 + 0.977201 w_k1_IBM_s011
                                 + 0.842344 w_k1_INTC_s011
                                 + 0.843777 w_k1_KO_s011
                                 + 0.909878 w_k1_MAT_s011
                                 + 0.825165 w_k1_MSFT_s011
                                 + 0.927982 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.859018 w_k1_ADM_s011
                                 + 0.868545 w_k1_AXP_s011
                                 + 0.848669 w_k1_DIS_s011
                                 + 0.808798 w_k1_FDX_s011
                                 + 1.17746 w_k1_IBM_s011
                                 + 1.0544 w_k1_INTC_s011 + 1.07467 w_k1_KO_s011
                                 + 0.713829 w_k1_MAT_s011
                                 + 0.739758 w_k1_MSFT_s011
                                 + 0.958768 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.03907 w_k1_ADM_s011
                                 + 1.29146 w_k1_AXP_s011
                                 + 1.00427 w_k1_DIS_s011
                                 + 0.965451 w_k1_FDX_s011
                                 + 1.01894 w_k1_IBM_s011
                                 + 0.887534 w_k1_INTC_s011
                                 + 1.01079 w_k1_KO_s011 + 1.21792 w_k1_MAT_s011
                                 + 0.962913 w_k1_MSFT_s011
                                 + 1.0517 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.24487 w_k1_ADM_s011
                                 + 1.04561 w_k1_AXP_s011
                                 + 1.35494 w_k1_DIS_s011 + 1.1458 w_k1_FDX_s011
                                 + 0.925499 w_k1_IBM_s011
                                 + 1.05002 w_k1_INTC_s011
                                 + 1.07316 w_k1_KO_s011
                                 + 0.441693 w_k1_MAT_s011
                                 + 1.00321 w_k1_MSFT_s011
                                 + 1.05227 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.853732 w_k1_ADM_s011
                                 + 0.782391 w_k1_AXP_s011
                                 + 0.772654 w_k1_DIS_s011
                                 + 0.786875 w_k1_FDX_s011
                                 + 0.735093 w_k1_IBM_s011
                                 + 0.838773 w_k1_INTC_s011
                                 + 0.74269 w_k1_KO_s011
                                 + 0.593664 w_k1_MAT_s011
                                 + 0.753497 w_k1_MSFT_s011
                                 + 0.744799 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.08269 w_k1_ADM_s011
                                 + 0.744934 w_k1_AXP_s011
                                 + 0.577363 w_k1_DIS_s011
                                 + 0.710788 w_k1_FDX_s011
                                 + 0.610722 w_k1_IBM_s011
                                 + 0.833727 w_k1_INTC_s011
                                 + 0.690442 w_k1_KO_s011
                                 + 0.435391 w_k1_MAT_s011
                                 + 0.731254 w_k1_MSFT_s011
                                 + 0.705442 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.39804 w_k1_ADM_s011
                                 + 1.50207 w_k1_AXP_s011
                                 + 1.48626 w_k1_DIS_s011
                                 + 1.19569 w_k1_FDX_s011 + 1.4366 w_k1_IBM_s011
                                 + 1.16837 w_k1_INTC_s011
                                 + 1.22873 w_k1_KO_s011 + 1.30956 w_k1_MAT_s011
                                 + 1.03333 w_k1_MSFT_s011
                                 + 1.18884 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.36336 w_k1_ADM_s011
                                 + 1.22059 w_k1_AXP_s011 + 1.1761 w_k1_DIS_s011
                                 + 1.21339 w_k1_FDX_s011
                                 + 0.772612 w_k1_IBM_s011
                                 + 1.27008 w_k1_INTC_s011
                                 + 0.891073 w_k1_KO_s011
                                 + 0.637464 w_k1_MAT_s011
                                 + 1.06641 w_k1_MSFT_s011
                                 + 0.871759 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.916217 w_k1_ADM_s012
                                 + 0.822718 w_k1_AXP_s012
                                 + 0.949024 w_k1_DIS_s012
                                 + 1.01338 w_k1_FDX_s012
                                 + 0.899454 w_k1_IBM_s012
                                 + 0.890953 w_k1_INTC_s012
                                 + 0.880204 w_k1_KO_s012
                                 + 0.815994 w_k1_MAT_s012
                                 + 0.876644 w_k1_MSFT_s012
                                 + 1.00243 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.26074 w_k1_ADM_s012
                                 + 1.44699 w_k1_AXP_s012
                                 + 1.89781 w_k1_DIS_s012 + 1.5835 w_k1_FDX_s012
                                 + 1.15706 w_k1_IBM_s012
                                 + 1.67135 w_k1_INTC_s012
                                 + 1.43228 w_k1_KO_s012
                                 + 0.856321 w_k1_MAT_s012
                                 + 1.58739 w_k1_MSFT_s012
                                 + 1.26488 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.07844 w_k1_ADM_s012
                                 + 1.19043 w_k1_AXP_s012
                                 + 1.10912 w_k1_DIS_s012
                                 + 0.998816 w_k1_FDX_s012
                                 + 1.43286 w_k1_IBM_s012
                                 + 0.91273 w_k1_INTC_s012
                                 + 1.02931 w_k1_KO_s012 + 1.16593 w_k1_MAT_s012
                                 + 0.952123 w_k1_MSFT_s012
                                 + 1.04661 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.687934 w_k1_ADM_s012
                                 + 0.769369 w_k1_AXP_s012
                                 + 0.759528 w_k1_DIS_s012
                                 + 0.889684 w_k1_FDX_s012
                                 + 1.11944 w_k1_IBM_s012
                                 + 0.950091 w_k1_INTC_s012
                                 + 0.898498 w_k1_KO_s012
                                 + 0.923739 w_k1_MAT_s012
                                 + 0.928713 w_k1_MSFT_s012
                                 + 0.986214 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.594637 w_k1_ADM_s012
                                 + 0.759991 w_k1_AXP_s012
                                 + 0.675074 w_k1_DIS_s012
                                 + 0.756106 w_k1_FDX_s012
                                 + 0.809178 w_k1_IBM_s012
                                 + 0.840866 w_k1_INTC_s012
                                 + 0.750868 w_k1_KO_s012
                                 + 0.81994 w_k1_MAT_s012
                                 + 0.725814 w_k1_MSFT_s012
                                 + 0.769134 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.22674 w_k1_ADM_s012
                                 + 1.53563 w_k1_AXP_s012
                                 + 1.43425 w_k1_DIS_s012
                                 + 1.31701 w_k1_FDX_s012
                                 + 1.26622 w_k1_IBM_s012
                                 + 1.10381 w_k1_INTC_s012
                                 + 1.29506 w_k1_KO_s012 + 1.5672 w_k1_MAT_s012
                                 + 1.19976 w_k1_MSFT_s012
                                 + 1.16742 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.02341 w_k1_ADM_s012
                                 + 1.02626 w_k1_AXP_s012
                                 + 1.11039 w_k1_DIS_s012
                                 + 1.20531 w_k1_FDX_s012
                                 + 0.640228 w_k1_IBM_s012
                                 + 0.843463 w_k1_INTC_s012
                                 + 0.705844 w_k1_KO_s012
                                 + 0.655985 w_k1_MAT_s012
                                 + 0.907141 w_k1_MSFT_s012
                                 + 0.926969 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.00244 w_k1_ADM_s012
                                 + 1.10961 w_k1_AXP_s012
                                 + 0.902378 w_k1_DIS_s012
                                 + 1.11822 w_k1_FDX_s012
                                 + 0.856288 w_k1_IBM_s012
                                 + 1.05336 w_k1_INTC_s012
                                 + 0.833219 w_k1_KO_s012
                                 + 0.896559 w_k1_MAT_s012
                                 + 0.88591 w_k1_MSFT_s012
                                 + 0.949677 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.950422 w_k1_ADM_s013
                                 + 1.04089 w_k1_AXP_s013
                                 + 1.17383 w_k1_DIS_s013
                                 + 0.933864 w_k1_FDX_s013
                                 + 1.49074 w_k1_IBM_s013
                                 + 0.994246 w_k1_INTC_s013
                                 + 1.1043 w_k1_KO_s013 + 1.16755 w_k1_MAT_s013
                                 + 0.90179 w_k1_MSFT_s013
                                 + 1.04794 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.698009 w_k1_ADM_s013
                                 + 1.27522 w_k1_AXP_s013
                                 + 0.992581 w_k1_DIS_s013
                                 + 0.921383 w_k1_FDX_s013
                                 + 1.1962 w_k1_IBM_s013
                                 + 0.873784 w_k1_INTC_s013
                                 + 0.970456 w_k1_KO_s013
                                 + 1.36462 w_k1_MAT_s013
                                 + 0.749413 w_k1_MSFT_s013
                                 + 0.921329 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.18967 w_k1_ADM_s013
                                 + 1.82563 w_k1_AXP_s013 + 1.6275 w_k1_DIS_s013
                                 + 1.47332 w_k1_FDX_s013
                                 + 1.12594 w_k1_IBM_s013
                                 + 1.33084 w_k1_INTC_s013
                                 + 1.23467 w_k1_KO_s013 + 1.29331 w_k1_MAT_s013
                                 + 1.2264 w_k1_MSFT_s013 + 1.22104 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.870361 w_k1_ADM_s013
                                 + 1.21578 w_k1_AXP_s013
                                 + 1.31452 w_k1_DIS_s013
                                 + 1.15422 w_k1_FDX_s013
                                 + 0.803308 w_k1_IBM_s013
                                 + 1.02642 w_k1_INTC_s013
                                 + 0.876951 w_k1_KO_s013
                                 + 1.01911 w_k1_MAT_s013
                                 + 1.03667 w_k1_MSFT_s013
                                 + 1.04343 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.700407 w_k1_ADM_s013
                                 + 0.677492 w_k1_AXP_s013
                                 + 0.770635 w_k1_DIS_s013
                                 + 0.811079 w_k1_FDX_s013
                                 + 0.775008 w_k1_IBM_s013
                                 + 0.807425 w_k1_INTC_s013
                                 + 0.831943 w_k1_KO_s013
                                 + 0.742876 w_k1_MAT_s013
                                 + 0.724823 w_k1_MSFT_s013
                                 + 0.835633 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.810377 w_k1_ADM_s013
                                 + 1.43407 w_k1_AXP_s013 + 1.5746 w_k1_DIS_s013
                                 + 1.256 w_k1_FDX_s013 + 1.2759 w_k1_IBM_s013
                                 + 1.33082 w_k1_INTC_s013
                                 + 1.19557 w_k1_KO_s013 + 1.6049 w_k1_MAT_s013
                                 + 1.20792 w_k1_MSFT_s013
                                 + 1.07299 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.873203 w_k1_ADM_s013
                                 + 0.891977 w_k1_AXP_s013
                                 + 1.09628 w_k1_DIS_s013
                                 + 1.02721 w_k1_FDX_s013
                                 + 0.928262 w_k1_IBM_s013
                                 + 0.960985 w_k1_INTC_s013
                                 + 0.823052 w_k1_KO_s013
                                 + 0.573182 w_k1_MAT_s013
                                 + 0.948182 w_k1_MSFT_s013
                                 + 0.847022 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.842133 w_k1_ADM_s013
                                 + 0.964408 w_k1_AXP_s013
                                 + 1.2538 w_k1_DIS_s013 + 1.11515 w_k1_FDX_s013
                                 + 1.08471 w_k1_IBM_s013
                                 + 1.13023 w_k1_INTC_s013
                                 + 1.10943 w_k1_KO_s013
                                 + 0.974236 w_k1_MAT_s013
                                 + 1.0736 w_k1_MSFT_s013
                                 + 0.976857 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.08264 w_k1_ADM_s014
                                 + 1.15021 w_k1_AXP_s014
                                 + 1.21056 w_k1_DIS_s014
                                 + 1.15986 w_k1_FDX_s014
                                 + 1.31172 w_k1_IBM_s014
                                 + 0.895062 w_k1_INTC_s014
                                 + 1.12401 w_k1_KO_s014 + 1.11793 w_k1_MAT_s014
                                 + 1.07206 w_k1_MSFT_s014 + 1.0621 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.41147 w_k1_ADM_s014
                                 + 1.69995 w_k1_AXP_s014
                                 + 1.69855 w_k1_DIS_s014
                                 + 1.44386 w_k1_FDX_s014 + 1.2166 w_k1_IBM_s014
                                 + 1.21452 w_k1_INTC_s014
                                 + 1.34915 w_k1_KO_s014 + 1.34835 w_k1_MAT_s014
                                 + 1.41613 w_k1_MSFT_s014
                                 + 1.28053 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.32261 w_k1_ADM_s014
                                 + 1.14909 w_k1_AXP_s014
                                 + 0.936641 w_k1_DIS_s014
                                 + 1.05059 w_k1_FDX_s014
                                 + 0.884187 w_k1_IBM_s014
                                 + 0.863878 w_k1_INTC_s014
                                 + 0.883877 w_k1_KO_s014
                                 + 0.857349 w_k1_MAT_s014
                                 + 0.989794 w_k1_MSFT_s014
                                 + 1.02166 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.828211 w_k1_ADM_s014
                                 + 0.969864 w_k1_AXP_s014
                                 + 0.864122 w_k1_DIS_s014
                                 + 0.91516 w_k1_FDX_s014 + 1.0381 w_k1_IBM_s014
                                 + 0.901103 w_k1_INTC_s014
                                 + 0.883705 w_k1_KO_s014
                                 + 0.983368 w_k1_MAT_s014
                                 + 0.956832 w_k1_MSFT_s014
                                 + 0.886679 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.44127 w_k1_ADM_s014
                                 + 1.39582 w_k1_AXP_s014
                                 + 1.40507 w_k1_DIS_s014
                                 + 1.35684 w_k1_FDX_s014
                                 + 0.973789 w_k1_IBM_s014
                                 + 0.891602 w_k1_INTC_s014
                                 + 0.970029 w_k1_KO_s014
                                 + 1.14916 w_k1_MAT_s014
                                 + 1.19437 w_k1_MSFT_s014
                                 + 1.08084 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.993161 w_k1_ADM_s014
                                 + 0.834446 w_k1_AXP_s014
                                 + 0.890359 w_k1_DIS_s014
                                 + 0.858325 w_k1_FDX_s014
                                 + 1.47696 w_k1_IBM_s014
                                 + 1.01561 w_k1_INTC_s014
                                 + 1.16866 w_k1_KO_s014 + 1.38692 w_k1_MAT_s014
                                 + 1.10458 w_k1_MSFT_s014
                                 + 1.08852 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.838853 w_k1_ADM_s014
                                 + 0.664269 w_k1_AXP_s014
                                 + 0.887512 w_k1_DIS_s014
                                 + 0.941619 w_k1_FDX_s014
                                 + 0.792528 w_k1_IBM_s014
                                 + 0.898024 w_k1_INTC_s014
                                 + 0.86011 w_k1_KO_s014
                                 + 0.662505 w_k1_MAT_s014
                                 + 0.99346 w_k1_MSFT_s014
                                 + 0.907573 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.812191 w_k1_ADM_s014
                                 + 0.644429 w_k1_AXP_s014
                                 + 0.504619 w_k1_DIS_s014
                                 + 0.664708 w_k1_FDX_s014
                                 + 0.813005 w_k1_IBM_s014
                                 + 0.656144 w_k1_INTC_s014
                                 + 0.686996 w_k1_KO_s014
                                 + 0.758213 w_k1_MAT_s014
                                 + 0.822384 w_k1_MSFT_s014
                                 + 0.766697 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.959293 w_k1_ADM_s015
                                 + 0.883411 w_k1_AXP_s015 + 1.157 w_k1_DIS_s015
                                 + 1.06971 w_k1_FDX_s015
                                 + 1.22783 w_k1_IBM_s015
                                 + 1.1406 w_k1_INTC_s015 + 1.26281 w_k1_KO_s015
                                 + 1.08986 w_k1_MAT_s015
                                 + 1.16432 w_k1_MSFT_s015
                                 + 1.06749 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.44722 w_k1_ADM_s015
                                 + 1.29073 w_k1_AXP_s015
                                 + 1.32251 w_k1_DIS_s015
                                 + 1.32604 w_k1_FDX_s015
                                 + 1.02191 w_k1_IBM_s015
                                 + 1.0637 w_k1_INTC_s015 + 1.00945 w_k1_KO_s015
                                 + 1.13747 w_k1_MAT_s015
                                 + 1.10964 w_k1_MSFT_s015
                                 + 1.24583 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.939204 w_k1_ADM_s015
                                 + 0.928888 w_k1_AXP_s015
                                 + 0.74624 w_k1_DIS_s015
                                 + 1.03703 w_k1_FDX_s015
                                 + 0.69931 w_k1_IBM_s015
                                 + 1.03664 w_k1_INTC_s015
                                 + 0.810931 w_k1_KO_s015
                                 + 0.992865 w_k1_MAT_s015
                                 + 0.916387 w_k1_MSFT_s015
                                 + 1.02322 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.1328 w_k1_ADM_s015
                                 + 1.11313 w_k1_AXP_s015
                                 + 1.13512 w_k1_DIS_s015
                                 + 1.08008 w_k1_FDX_s015
                                 + 0.971242 w_k1_IBM_s015
                                 + 0.980729 w_k1_INTC_s015
                                 + 1.08683 w_k1_KO_s015 + 1.05435 w_k1_MAT_s015
                                 + 1.05119 w_k1_MSFT_s015
                                 + 1.16347 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.61464 w_k1_ADM_s015
                                 + 0.690883 w_k1_AXP_s015
                                 + 0.66462 w_k1_DIS_s015
                                 + 0.774986 w_k1_FDX_s015
                                 + 0.771484 w_k1_IBM_s015
                                 + 0.848197 w_k1_INTC_s015
                                 + 0.753184 w_k1_KO_s015
                                 + 1.09616 w_k1_MAT_s015
                                 + 0.777637 w_k1_MSFT_s015
                                 + 0.86839 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.21535 w_k1_ADM_s015
                                 + 1.21802 w_k1_AXP_s015
                                 + 1.42502 w_k1_DIS_s015
                                 + 1.16858 w_k1_FDX_s015
                                 + 1.53014 w_k1_IBM_s015
                                 + 1.11965 w_k1_INTC_s015 + 1.6604 w_k1_KO_s015
                                 + 1.73264 w_k1_MAT_s015
                                 + 1.19285 w_k1_MSFT_s015
                                 + 1.26409 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.03849 w_k1_ADM_s015
                                 + 0.782074 w_k1_AXP_s015
                                 + 0.984546 w_k1_DIS_s015
                                 + 0.930608 w_k1_FDX_s015
                                 + 0.787289 w_k1_IBM_s015
                                 + 0.862447 w_k1_INTC_s015
                                 + 1.01872 w_k1_KO_s015 + 0.85696 w_k1_MAT_s015
                                 + 1.02793 w_k1_MSFT_s015
                                 + 1.02604 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.958813 w_k1_ADM_s015
                                 + 0.95041 w_k1_AXP_s015
                                 + 0.935347 w_k1_DIS_s015
                                 + 0.88338 w_k1_FDX_s015
                                 + 1.24844 w_k1_IBM_s015
                                 + 1.06642 w_k1_INTC_s015 + 1.1787 w_k1_KO_s015
                                 + 1.31419 w_k1_MAT_s015
                                 + 0.891399 w_k1_MSFT_s015
                                 + 1.01879 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.04119 w_k1_ADM_s016
                                 + 0.666481 w_k1_AXP_s016
                                 + 0.715301 w_k1_DIS_s016
                                 + 0.747397 w_k1_FDX_s016
                                 + 0.834706 w_k1_IBM_s016
                                 + 0.792368 w_k1_INTC_s016
                                 + 0.806839 w_k1_KO_s016
                                 + 0.648076 w_k1_MAT_s016
                                 + 0.743368 w_k1_MSFT_s016
                                 + 0.7956 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.30777 w_k1_ADM_s016
                                 + 1.6697 w_k1_AXP_s016 + 1.75387 w_k1_DIS_s016
                                 + 1.39621 w_k1_FDX_s016
                                 + 1.44116 w_k1_IBM_s016
                                 + 1.35418 w_k1_INTC_s016
                                 + 1.44777 w_k1_KO_s016 + 1.44599 w_k1_MAT_s016
                                 + 1.43916 w_k1_MSFT_s016
                                 + 1.28931 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.941508 w_k1_ADM_s016
                                 + 0.873126 w_k1_AXP_s016
                                 + 0.907103 w_k1_DIS_s016
                                 + 0.870941 w_k1_FDX_s016
                                 + 1.09173 w_k1_IBM_s016
                                 + 0.934972 w_k1_INTC_s016
                                 + 1.06175 w_k1_KO_s016
                                 + 0.848034 w_k1_MAT_s016
                                 + 0.843002 w_k1_MSFT_s016
                                 + 0.973986 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.984906 w_k1_ADM_s016
                                 + 1.1343 w_k1_AXP_s016
                                 + 0.749941 w_k1_DIS_s016
                                 + 0.957078 w_k1_FDX_s016
                                 + 0.992251 w_k1_IBM_s016
                                 + 1.08092 w_k1_INTC_s016
                                 + 0.821168 w_k1_KO_s016
                                 + 1.10591 w_k1_MAT_s016
                                 + 0.910002 w_k1_MSFT_s016
                                 + 0.980663 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.94023 w_k1_ADM_s016
                                 + 0.872454 w_k1_AXP_s016
                                 + 0.876539 w_k1_DIS_s016
                                 + 0.91201 w_k1_FDX_s016
                                 + 0.709574 w_k1_IBM_s016
                                 + 1.08792 w_k1_INTC_s016
                                 + 0.808286 w_k1_KO_s016
                                 + 0.772909 w_k1_MAT_s016
                                 + 1.00457 w_k1_MSFT_s016
                                 + 0.897574 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.32829 w_k1_ADM_s016
                                 + 1.25371 w_k1_AXP_s016
                                 + 1.24532 w_k1_DIS_s016
                                 + 1.20727 w_k1_FDX_s016 + 1.1385 w_k1_IBM_s016
                                 + 1.03185 w_k1_INTC_s016
                                 + 1.26279 w_k1_KO_s016 + 1.21797 w_k1_MAT_s016
                                 + 1.22146 w_k1_MSFT_s016
                                 + 1.20918 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.09024 w_k1_ADM_s016
                                 + 1.08533 w_k1_AXP_s016
                                 + 1.00296 w_k1_DIS_s016
                                 + 1.11665 w_k1_FDX_s016
                                 + 0.891829 w_k1_IBM_s016
                                 + 1.28995 w_k1_INTC_s016
                                 + 1.00135 w_k1_KO_s016
                                 + 0.921656 w_k1_MAT_s016
                                 + 1.20992 w_k1_MSFT_s016
                                 + 0.959324 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.867218 w_k1_ADM_s016
                                 + 1.11651 w_k1_AXP_s016
                                 + 1.20059 w_k1_DIS_s016
                                 + 1.02673 w_k1_FDX_s016
                                 + 1.52223 w_k1_IBM_s016
                                 + 1.11122 w_k1_INTC_s016 + 1.3619 w_k1_KO_s016
                                 + 1.56087 w_k1_MAT_s016
                                 + 1.25539 w_k1_MSFT_s016
                                 + 1.06292 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.800978 w_k1_ADM_s017
                                 + 0.735648 w_k1_AXP_s017
                                 + 0.780738 w_k1_DIS_s017
                                 + 0.830821 w_k1_FDX_s017
                                 + 0.953318 w_k1_IBM_s017
                                 + 1.02015 w_k1_INTC_s017
                                 + 0.787207 w_k1_KO_s017
                                 + 1.24198 w_k1_MAT_s017
                                 + 0.794069 w_k1_MSFT_s017
                                 + 0.892696 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.956122 w_k1_ADM_s017
                                 + 0.807539 w_k1_AXP_s017
                                 + 1.07378 w_k1_DIS_s017
                                 + 0.948074 w_k1_FDX_s017
                                 + 1.07786 w_k1_IBM_s017
                                 + 1.05787 w_k1_INTC_s017
                                 + 1.00871 w_k1_KO_s017 + 1.34337 w_k1_MAT_s017
                                 + 1.08369 w_k1_MSFT_s017
                                 + 0.951945 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.11367 w_k1_ADM_s017
                                 + 1.2782 w_k1_AXP_s017 + 1.62317 w_k1_DIS_s017
                                 + 1.24119 w_k1_FDX_s017
                                 + 1.29219 w_k1_IBM_s017
                                 + 1.49069 w_k1_INTC_s017
                                 + 1.41812 w_k1_KO_s017 + 1.35095 w_k1_MAT_s017
                                 + 1.10125 w_k1_MSFT_s017 + 1.1944 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.14943 w_k1_ADM_s017
                                 + 1.142 w_k1_AXP_s017 + 1.43506 w_k1_DIS_s017
                                 + 1.12898 w_k1_FDX_s017
                                 + 1.43411 w_k1_IBM_s017
                                 + 1.2479 w_k1_INTC_s017 + 1.52922 w_k1_KO_s017
                                 + 1.71271 w_k1_MAT_s017
                                 + 1.15056 w_k1_MSFT_s017
                                 + 1.14577 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.640274 w_k1_ADM_s017
                                 + 0.738105 w_k1_AXP_s017
                                 + 0.885013 w_k1_DIS_s017
                                 + 0.773534 w_k1_FDX_s017
                                 + 1.25849 w_k1_IBM_s017
                                 + 0.977307 w_k1_INTC_s017
                                 + 1.08402 w_k1_KO_s017 + 1.53337 w_k1_MAT_s017
                                 + 0.823496 w_k1_MSFT_s017
                                 + 0.831165 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.972074 w_k1_ADM_s017
                                 + 0.809134 w_k1_AXP_s017
                                 + 0.815806 w_k1_DIS_s017
                                 + 0.859654 w_k1_FDX_s017
                                 + 0.684824 w_k1_IBM_s017
                                 + 0.89069 w_k1_INTC_s017
                                 + 0.859367 w_k1_KO_s017
                                 + 0.819359 w_k1_MAT_s017
                                 + 0.879098 w_k1_MSFT_s017
                                 + 0.905321 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.42336 w_k1_ADM_s017
                                 + 1.64139 w_k1_AXP_s017
                                 + 1.35233 w_k1_DIS_s017
                                 + 1.35263 w_k1_FDX_s017
                                 + 1.03222 w_k1_IBM_s017
                                 + 1.35688 w_k1_INTC_s017
                                 + 1.23846 w_k1_KO_s017 + 1.56636 w_k1_MAT_s017
                                 + 1.2338 w_k1_MSFT_s017 + 1.15855 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.27214 w_k1_ADM_s017
                                 + 1.18293 w_k1_AXP_s017
                                 + 1.04173 w_k1_DIS_s017
                                 + 1.13273 w_k1_FDX_s017
                                 + 0.862569 w_k1_IBM_s017
                                 + 1.11462 w_k1_INTC_s017
                                 + 0.905425 w_k1_KO_s017
                                 + 0.94886 w_k1_MAT_s017
                                 + 1.07867 w_k1_MSFT_s017
                                 + 0.992175 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.867589 w_k1_ADM_s100
                                 + 1.05681 w_k1_AXP_s100
                                 + 1.25233 w_k1_DIS_s100
                                 + 1.05413 w_k1_FDX_s100
                                 + 1.51689 w_k1_IBM_s100
                                 + 1.05189 w_k1_INTC_s100 + 1.4229 w_k1_KO_s100
                                 + 1.64503 w_k1_MAT_s100
                                 + 1.14487 w_k1_MSFT_s100
                                 + 1.13227 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.23102 w_k1_ADM_s100
                                 + 1.56703 w_k1_AXP_s100
                                 + 1.86475 w_k1_DIS_s100
                                 + 1.39641 w_k1_FDX_s100
                                 + 1.25739 w_k1_IBM_s100
                                 + 1.35471 w_k1_INTC_s100
                                 + 1.55227 w_k1_KO_s100 + 1.78773 w_k1_MAT_s100
                                 + 1.4987 w_k1_MSFT_s100 + 1.37782 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.917944 w_k1_ADM_s100
                                 + 0.819027 w_k1_AXP_s100
                                 + 0.758345 w_k1_DIS_s100
                                 + 0.815181 w_k1_FDX_s100
                                 + 0.938606 w_k1_IBM_s100
                                 + 0.905132 w_k1_INTC_s100
                                 + 0.974573 w_k1_KO_s100
                                 + 0.773947 w_k1_MAT_s100
                                 + 0.884844 w_k1_MSFT_s100
                                 + 0.954576 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.07291 w_k1_ADM_s100
                                 + 1.20993 w_k1_AXP_s100
                                 + 1.04503 w_k1_DIS_s100
                                 + 1.11006 w_k1_FDX_s100
                                 + 0.801795 w_k1_IBM_s100
                                 + 0.903676 w_k1_INTC_s100
                                 + 0.821344 w_k1_KO_s100
                                 + 1.02713 w_k1_MAT_s100
                                 + 1.11703 w_k1_MSFT_s100
                                 + 0.994939 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.925122 w_k1_ADM_s100
                                 + 1.03651 w_k1_AXP_s100
                                 + 0.862765 w_k1_DIS_s100
                                 + 0.990628 w_k1_FDX_s100
                                 + 1.2327 w_k1_IBM_s100
                                 + 1.09381 w_k1_INTC_s100
                                 + 1.08761 w_k1_KO_s100 + 1.02626 w_k1_MAT_s100
                                 + 1.02211 w_k1_MSFT_s100
                                 + 1.07592 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.570897 w_k1_ADM_s100
                                 + 0.790918 w_k1_AXP_s100
                                 + 0.630004 w_k1_DIS_s100
                                 + 0.673866 w_k1_FDX_s100
                                 + 1.2997 w_k1_IBM_s100
                                 + 0.885029 w_k1_INTC_s100
                                 + 0.895347 w_k1_KO_s100
                                 + 1.00099 w_k1_MAT_s100
                                 + 0.832045 w_k1_MSFT_s100
                                 + 0.774099 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.704111 w_k1_ADM_s100
                                 + 0.612792 w_k1_AXP_s100
                                 + 0.718451 w_k1_DIS_s100
                                 + 0.786309 w_k1_FDX_s100
                                 + 0.659744 w_k1_IBM_s100
                                 + 1.02791 w_k1_INTC_s100
                                 + 0.805178 w_k1_KO_s100
                                 + 0.511237 w_k1_MAT_s100
                                 + 0.887582 w_k1_MSFT_s100
                                 + 0.810896 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.957496 w_k1_ADM_s100
                                 + 1.06151 w_k1_AXP_s100
                                 + 1.31418 w_k1_DIS_s100
                                 + 1.29524 w_k1_FDX_s100 + 1.0254 w_k1_IBM_s100
                                 + 1.15522 w_k1_INTC_s100
                                 + 1.17284 w_k1_KO_s100
                                 + 0.831339 w_k1_MAT_s100
                                 + 1.28238 w_k1_MSFT_s100
                                 + 0.961392 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.778517 w_k1_ADM_s101
                                 + 0.569674 w_k1_AXP_s101
                                 + 0.632139 w_k1_DIS_s101
                                 + 0.707879 w_k1_FDX_s101
                                 + 0.649573 w_k1_IBM_s101
                                 + 0.701195 w_k1_INTC_s101
                                 + 0.631242 w_k1_KO_s101
                                 + 0.542095 w_k1_MAT_s101
                                 + 0.833418 w_k1_MSFT_s101
                                 + 0.817155 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.58736 w_k1_ADM_s101
                                 + 1.20793 w_k1_AXP_s101
                                 + 1.15086 w_k1_DIS_s101
                                 + 1.15854 w_k1_FDX_s101
                                 + 0.908222 w_k1_IBM_s101
                                 + 1.11278 w_k1_INTC_s101
                                 + 1.05014 w_k1_KO_s101
                                 + 0.997652 w_k1_MAT_s101
                                 + 1.10945 w_k1_MSFT_s101
                                 + 1.06724 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.41985 w_k1_ADM_s101
                                 + 1.21043 w_k1_AXP_s101 + 1.1311 w_k1_DIS_s101
                                 + 1.19468 w_k1_FDX_s101
                                 + 1.07973 w_k1_IBM_s101
                                 + 1.07902 w_k1_INTC_s101 + 1.1638 w_k1_KO_s101
                                 + 1.05057 w_k1_MAT_s101
                                 + 1.17757 w_k1_MSFT_s101
                                 + 1.10173 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.28464 w_k1_ADM_s101
                                 + 1.30673 w_k1_AXP_s101
                                 + 1.27119 w_k1_DIS_s101
                                 + 1.07762 w_k1_FDX_s101
                                 + 1.45931 w_k1_IBM_s101
                                 + 1.21104 w_k1_INTC_s101
                                 + 1.25495 w_k1_KO_s101 + 1.46078 w_k1_MAT_s101
                                 + 1.19077 w_k1_MSFT_s101
                                 + 1.06597 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.07395 w_k1_ADM_s101
                                 + 0.943131 w_k1_AXP_s101
                                 + 0.836732 w_k1_DIS_s101
                                 + 1.03528 w_k1_FDX_s101
                                 + 0.697385 w_k1_IBM_s101
                                 + 1.01915 w_k1_INTC_s101
                                 + 0.842225 w_k1_KO_s101
                                 + 0.536377 w_k1_MAT_s101
                                 + 1.02312 w_k1_MSFT_s101
                                 + 0.929787 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.06243 w_k1_ADM_s101
                                 + 0.838025 w_k1_AXP_s101
                                 + 1.22949 w_k1_DIS_s101
                                 + 1.02517 w_k1_FDX_s101
                                 + 0.806467 w_k1_IBM_s101
                                 + 1.10029 w_k1_INTC_s101
                                 + 1.00857 w_k1_KO_s101
                                 + 0.627188 w_k1_MAT_s101
                                 + 1.06888 w_k1_MSFT_s101
                                 + 1.03373 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.15709 w_k1_ADM_s101
                                 + 0.999708 w_k1_AXP_s101
                                 + 0.705268 w_k1_DIS_s101
                                 + 0.860474 w_k1_FDX_s101
                                 + 0.918981 w_k1_IBM_s101
                                 + 0.703984 w_k1_INTC_s101
                                 + 0.796594 w_k1_KO_s101
                                 + 0.927541 w_k1_MAT_s101
                                 + 0.788152 w_k1_MSFT_s101
                                 + 0.945206 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.1078 w_k1_ADM_s101
                                 + 0.765128 w_k1_AXP_s101
                                 + 0.807808 w_k1_DIS_s101
                                 + 0.79766 w_k1_FDX_s101
                                 + 1.02362 w_k1_IBM_s101
                                 + 0.962861 w_k1_INTC_s101
                                 + 0.966455 w_k1_KO_s101
                                 + 0.577401 w_k1_MAT_s101
                                 + 0.819395 w_k1_MSFT_s101
                                 + 0.86243 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.936306 w_k1_ADM_s102
                                 + 0.934047 w_k1_AXP_s102
                                 + 1.21842 w_k1_DIS_s102
                                 + 1.00783 w_k1_FDX_s102
                                 + 0.931084 w_k1_IBM_s102
                                 + 1.14822 w_k1_INTC_s102
                                 + 1.08195 w_k1_KO_s102
                                 + 0.921517 w_k1_MAT_s102
                                 + 1.10046 w_k1_MSFT_s102
                                 + 1.00722 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.18833 w_k1_ADM_s102
                                 + 1.59052 w_k1_AXP_s102
                                 + 1.70368 w_k1_DIS_s102
                                 + 1.23874 w_k1_FDX_s102
                                 + 1.36974 w_k1_IBM_s102
                                 + 1.61719 w_k1_INTC_s102 + 1.3609 w_k1_KO_s102
                                 + 1.55178 w_k1_MAT_s102
                                 + 1.43917 w_k1_MSFT_s102
                                 + 1.12136 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.725129 w_k1_ADM_s102
                                 + 0.933611 w_k1_AXP_s102
                                 + 0.739217 w_k1_DIS_s102
                                 + 0.800896 w_k1_FDX_s102
                                 + 1.05522 w_k1_IBM_s102
                                 + 0.984999 w_k1_INTC_s102
                                 + 1.02463 w_k1_KO_s102 + 1.44362 w_k1_MAT_s102
                                 + 0.860547 w_k1_MSFT_s102
                                 + 0.901303 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.798253 w_k1_ADM_s102
                                 + 0.724548 w_k1_AXP_s102
                                 + 1.0538 w_k1_DIS_s102
                                 + 0.870612 w_k1_FDX_s102
                                 + 1.02971 w_k1_IBM_s102
                                 + 1.08461 w_k1_INTC_s102
                                 + 1.02252 w_k1_KO_s102
                                 + 0.976352 w_k1_MAT_s102
                                 + 0.960095 w_k1_MSFT_s102
                                 + 0.957844 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.745364 w_k1_ADM_s102
                                 + 0.64363 w_k1_AXP_s102 + 0.6899 w_k1_DIS_s102
                                 + 0.748465 w_k1_FDX_s102
                                 + 0.828251 w_k1_IBM_s102
                                 + 0.982581 w_k1_INTC_s102
                                 + 0.856935 w_k1_KO_s102
                                 + 0.826165 w_k1_MAT_s102
                                 + 0.760391 w_k1_MSFT_s102
                                 + 0.857377 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04028 w_k1_ADM_s102
                                 + 1.09193 w_k1_AXP_s102
                                 + 0.841656 w_k1_DIS_s102
                                 + 1.04561 w_k1_FDX_s102
                                 + 0.838797 w_k1_IBM_s102
                                 + 1.11662 w_k1_INTC_s102
                                 + 0.830439 w_k1_KO_s102
                                 + 0.904499 w_k1_MAT_s102
                                 + 0.891204 w_k1_MSFT_s102
                                 + 0.93738 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.11582 w_k1_ADM_s102
                                 + 1.41074 w_k1_AXP_s102
                                 + 1.46568 w_k1_DIS_s102
                                 + 1.25922 w_k1_FDX_s102
                                 + 0.968448 w_k1_IBM_s102
                                 + 1.59843 w_k1_INTC_s102
                                 + 1.20712 w_k1_KO_s102 + 1.42143 w_k1_MAT_s102
                                 + 1.28555 w_k1_MSFT_s102
                                 + 1.29962 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.949294 w_k1_ADM_s102
                                 + 0.982643 w_k1_AXP_s102
                                 + 1.21672 w_k1_DIS_s102
                                 + 0.945339 w_k1_FDX_s102
                                 + 1.44373 w_k1_IBM_s102
                                 + 1.29633 w_k1_INTC_s102 + 1.4045 w_k1_KO_s102
                                 + 1.18631 w_k1_MAT_s102
                                 + 1.06925 w_k1_MSFT_s102
                                 + 1.14605 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.962599 w_k1_ADM_s103
                                 + 0.869434 w_k1_AXP_s103
                                 + 1.10663 w_k1_DIS_s103
                                 + 0.996894 w_k1_FDX_s103
                                 + 1.20775 w_k1_IBM_s103
                                 + 0.857453 w_k1_INTC_s103
                                 + 1.32109 w_k1_KO_s103 + 1.08595 w_k1_MAT_s103
                                 + 0.974303 w_k1_MSFT_s103
                                 + 1.03731 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.25613 w_k1_ADM_s103
                                 + 1.05542 w_k1_AXP_s103
                                 + 1.46411 w_k1_DIS_s103
                                 + 1.25801 w_k1_FDX_s103
                                 + 1.26391 w_k1_IBM_s103
                                 + 1.27536 w_k1_INTC_s103
                                 + 1.33814 w_k1_KO_s103 + 1.2116 w_k1_MAT_s103
                                 + 1.16856 w_k1_MSFT_s103
                                 + 1.09121 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.15399 w_k1_ADM_s103
                                 + 0.750265 w_k1_AXP_s103
                                 + 0.78551 w_k1_DIS_s103
                                 + 0.948306 w_k1_FDX_s103
                                 + 0.627987 w_k1_IBM_s103
                                 + 0.82648 w_k1_INTC_s103
                                 + 0.795016 w_k1_KO_s103
                                 + 0.447304 w_k1_MAT_s103
                                 + 0.87728 w_k1_MSFT_s103
                                 + 0.844658 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.0684 w_k1_ADM_s103
                                 + 1.17083 w_k1_AXP_s103
                                 + 0.937453 w_k1_DIS_s103
                                 + 1.1468 w_k1_FDX_s103 + 0.87646 w_k1_IBM_s103
                                 + 1.06567 w_k1_INTC_s103
                                 + 1.05122 w_k1_KO_s103
                                 + 0.958093 w_k1_MAT_s103
                                 + 0.952232 w_k1_MSFT_s103
                                 + 0.996432 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.810897 w_k1_ADM_s103
                                 + 0.803617 w_k1_AXP_s103
                                 + 0.736816 w_k1_DIS_s103
                                 + 0.872784 w_k1_FDX_s103
                                 + 0.80747 w_k1_IBM_s103
                                 + 0.884301 w_k1_INTC_s103
                                 + 0.796416 w_k1_KO_s103
                                 + 0.761797 w_k1_MAT_s103
                                 + 0.710798 w_k1_MSFT_s103
                                 + 0.861137 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.25995 w_k1_ADM_s103
                                 + 1.17237 w_k1_AXP_s103
                                 + 1.55442 w_k1_DIS_s103 + 1.3564 w_k1_FDX_s103
                                 + 1.00114 w_k1_IBM_s103
                                 + 1.23543 w_k1_INTC_s103
                                 + 1.15932 w_k1_KO_s103
                                 + 0.883713 w_k1_MAT_s103
                                 + 1.22329 w_k1_MSFT_s103
                                 + 1.09807 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.977426 w_k1_ADM_s103
                                 + 1.23018 w_k1_AXP_s103
                                 + 1.24737 w_k1_DIS_s103
                                 + 1.06449 w_k1_FDX_s103
                                 + 1.60292 w_k1_IBM_s103
                                 + 0.971967 w_k1_INTC_s103
                                 + 1.35715 w_k1_KO_s103 + 1.4811 w_k1_MAT_s103
                                 + 0.920256 w_k1_MSFT_s103
                                 + 1.09575 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.847357 w_k1_ADM_s103
                                 + 0.664472 w_k1_AXP_s103
                                 + 0.756206 w_k1_DIS_s103
                                 + 0.805544 w_k1_FDX_s103
                                 + 0.995834 w_k1_IBM_s103
                                 + 0.968596 w_k1_INTC_s103
                                 + 1.09111 w_k1_KO_s103
                                 + 0.739268 w_k1_MAT_s103
                                 + 0.793302 w_k1_MSFT_s103
                                 + 0.931798 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.837475 w_k1_ADM_s104
                                 + 0.73706 w_k1_AXP_s104
                                 + 0.616626 w_k1_DIS_s104
                                 + 0.790545 w_k1_FDX_s104
                                 + 0.616105 w_k1_IBM_s104
                                 + 0.768389 w_k1_INTC_s104
                                 + 0.671303 w_k1_KO_s104
                                 + 0.49185 w_k1_MAT_s104
                                 + 0.821077 w_k1_MSFT_s104
                                 + 0.686915 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.20699 w_k1_ADM_s104
                                 + 1.13768 w_k1_AXP_s104
                                 + 1.41166 w_k1_DIS_s104
                                 + 1.16073 w_k1_FDX_s104
                                 + 1.21556 w_k1_IBM_s104
                                 + 1.09098 w_k1_INTC_s104
                                 + 1.05933 w_k1_KO_s104
                                 + 0.909485 w_k1_MAT_s104
                                 + 1.14335 w_k1_MSFT_s104
                                 + 1.14334 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.35619 w_k1_ADM_s104
                                 + 1.34222 w_k1_AXP_s104
                                 + 1.33918 w_k1_DIS_s104
                                 + 1.35417 w_k1_FDX_s104
                                 + 0.841568 w_k1_IBM_s104
                                 + 1.08352 w_k1_INTC_s104
                                 + 1.04163 w_k1_KO_s104
                                 + 0.774371 w_k1_MAT_s104
                                 + 1.06528 w_k1_MSFT_s104
                                 + 1.23587 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.891902 w_k1_ADM_s104
                                 + 0.966555 w_k1_AXP_s104
                                 + 0.822142 w_k1_DIS_s104
                                 + 1.03253 w_k1_FDX_s104
                                 + 0.644626 w_k1_IBM_s104
                                 + 1.12618 w_k1_INTC_s104
                                 + 0.822893 w_k1_KO_s104
                                 + 0.689629 w_k1_MAT_s104
                                 + 0.942279 w_k1_MSFT_s104
                                 + 0.976054 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.790544 w_k1_ADM_s104
                                 + 1.28828 w_k1_AXP_s104
                                 + 0.942843 w_k1_DIS_s104
                                 + 0.956617 w_k1_FDX_s104
                                 + 1.24695 w_k1_IBM_s104
                                 + 0.74273 w_k1_INTC_s104
                                 + 0.893449 w_k1_KO_s104 + 1.5099 w_k1_MAT_s104
                                 + 0.850137 w_k1_MSFT_s104
                                 + 1.03596 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.954617 w_k1_ADM_s104
                                 + 1.12811 w_k1_AXP_s104
                                 + 1.02337 w_k1_DIS_s104
                                 + 1.06527 w_k1_FDX_s104
                                 + 0.906994 w_k1_IBM_s104
                                 + 1.0867 w_k1_INTC_s104 + 0.89354 w_k1_KO_s104
                                 + 0.986213 w_k1_MAT_s104
                                 + 1.03877 w_k1_MSFT_s104
                                 + 0.954751 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.957925 w_k1_ADM_s104
                                 + 0.778422 w_k1_AXP_s104
                                 + 0.833389 w_k1_DIS_s104
                                 + 0.829128 w_k1_FDX_s104
                                 + 0.962466 w_k1_IBM_s104
                                 + 0.857179 w_k1_INTC_s104
                                 + 0.78882 w_k1_KO_s104
                                 + 0.876112 w_k1_MAT_s104
                                 + 0.883484 w_k1_MSFT_s104
                                 + 0.972433 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.18099 w_k1_ADM_s104
                                 + 1.26221 w_k1_AXP_s104
                                 + 1.40981 w_k1_DIS_s104
                                 + 1.15294 w_k1_FDX_s104 + 1.6737 w_k1_IBM_s104
                                 + 1.01486 w_k1_INTC_s104
                                 + 1.38893 w_k1_KO_s104 + 1.62046 w_k1_MAT_s104
                                 + 1.13461 w_k1_MSFT_s104
                                 + 1.27138 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.05914 w_k1_ADM_s105
                                 + 1.22653 w_k1_AXP_s105 + 1.4438 w_k1_DIS_s105
                                 + 1.40472 w_k1_FDX_s105
                                 + 0.835327 w_k1_IBM_s105
                                 + 1.21299 w_k1_INTC_s105
                                 + 1.14661 w_k1_KO_s105 + 0.99515 w_k1_MAT_s105
                                 + 1.21463 w_k1_MSFT_s105
                                 + 1.23084 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11196 w_k1_ADM_s105
                                 + 1.23704 w_k1_AXP_s105
                                 + 1.12162 w_k1_DIS_s105
                                 + 1.19377 w_k1_FDX_s105
                                 + 1.03341 w_k1_IBM_s105
                                 + 0.937818 w_k1_INTC_s105
                                 + 1.14254 w_k1_KO_s105 + 1.31622 w_k1_MAT_s105
                                 + 1.02182 w_k1_MSFT_s105
                                 + 1.13998 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.697777 w_k1_ADM_s105
                                 + 0.692852 w_k1_AXP_s105
                                 + 0.641914 w_k1_DIS_s105
                                 + 0.743974 w_k1_FDX_s105
                                 + 0.967853 w_k1_IBM_s105
                                 + 0.728358 w_k1_INTC_s105
                                 + 0.855302 w_k1_KO_s105
                                 + 0.811883 w_k1_MAT_s105
                                 + 0.761524 w_k1_MSFT_s105
                                 + 0.921551 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.938633 w_k1_ADM_s105
                                 + 0.852136 w_k1_AXP_s105
                                 + 1.48264 w_k1_DIS_s105 + 1.1042 w_k1_FDX_s105
                                 + 1.07519 w_k1_IBM_s105
                                 + 0.995411 w_k1_INTC_s105
                                 + 1.22319 w_k1_KO_s105
                                 + 0.708354 w_k1_MAT_s105
                                 + 1.18412 w_k1_MSFT_s105
                                 + 1.07498 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.715832 w_k1_ADM_s105
                                 + 0.633394 w_k1_AXP_s105
                                 + 0.60488 w_k1_DIS_s105
                                 + 0.763991 w_k1_FDX_s105
                                 + 0.637455 w_k1_IBM_s105
                                 + 0.675786 w_k1_INTC_s105
                                 + 0.78303 w_k1_KO_s105
                                 + 0.705222 w_k1_MAT_s105
                                 + 0.795941 w_k1_MSFT_s105
                                 + 0.844265 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.738955 w_k1_ADM_s105
                                 + 0.758689 w_k1_AXP_s105
                                 + 0.866671 w_k1_DIS_s105
                                 + 0.679199 w_k1_FDX_s105
                                 + 1.34639 w_k1_IBM_s105
                                 + 0.749561 w_k1_INTC_s105
                                 + 1.15954 w_k1_KO_s105 + 1.09078 w_k1_MAT_s105
                                 + 0.811278 w_k1_MSFT_s105
                                 + 0.937523 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.90179 w_k1_ADM_s105
                                 + 0.927277 w_k1_AXP_s105
                                 + 0.876622 w_k1_DIS_s105
                                 + 0.944057 w_k1_FDX_s105
                                 + 0.737082 w_k1_IBM_s105
                                 + 0.841351 w_k1_INTC_s105
                                 + 0.825975 w_k1_KO_s105
                                 + 0.893035 w_k1_MAT_s105
                                 + 0.975177 w_k1_MSFT_s105
                                 + 0.910302 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.26307 w_k1_ADM_s105
                                 + 1.10612 w_k1_AXP_s105
                                 + 1.12918 w_k1_DIS_s105 + 1.0642 w_k1_FDX_s105
                                 + 1.36401 w_k1_IBM_s105
                                 + 0.963962 w_k1_INTC_s105
                                 + 1.30712 w_k1_KO_s105 + 1.33299 w_k1_MAT_s105
                                 + 1.12268 w_k1_MSFT_s105
                                 + 1.11598 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.979448 w_k1_ADM_s106
                                 + 0.896274 w_k1_AXP_s106
                                 + 1.02532 w_k1_DIS_s106
                                 + 1.17004 w_k1_FDX_s106
                                 + 0.59803 w_k1_IBM_s106
                                 + 1.16408 w_k1_INTC_s106
                                 + 0.838474 w_k1_KO_s106
                                 + 0.752207 w_k1_MAT_s106
                                 + 1.08936 w_k1_MSFT_s106
                                 + 0.955514 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.20774 w_k1_ADM_s106
                                 + 1.16513 w_k1_AXP_s106
                                 + 1.17854 w_k1_DIS_s106
                                 + 1.27686 w_k1_FDX_s106
                                 + 0.963694 w_k1_IBM_s106
                                 + 1.11849 w_k1_INTC_s106
                                 + 1.07322 w_k1_KO_s106 + 1.09423 w_k1_MAT_s106
                                 + 1.16251 w_k1_MSFT_s106
                                 + 1.03748 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.806264 w_k1_ADM_s106
                                 + 0.860611 w_k1_AXP_s106
                                 + 0.772769 w_k1_DIS_s106
                                 + 0.903582 w_k1_FDX_s106
                                 + 0.796088 w_k1_IBM_s106
                                 + 0.878366 w_k1_INTC_s106
                                 + 0.788945 w_k1_KO_s106
                                 + 1.19522 w_k1_MAT_s106
                                 + 0.824696 w_k1_MSFT_s106
                                 + 0.856383 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.0094 w_k1_ADM_s106
                                 + 1.05479 w_k1_AXP_s106 + 1.0812 w_k1_DIS_s106
                                 + 1.02085 w_k1_FDX_s106
                                 + 0.979523 w_k1_IBM_s106
                                 + 1.09647 w_k1_INTC_s106
                                 + 0.982097 w_k1_KO_s106
                                 + 0.994873 w_k1_MAT_s106
                                 + 0.956814 w_k1_MSFT_s106
                                 + 0.904538 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.07819 w_k1_ADM_s106
                                 + 0.868007 w_k1_AXP_s106
                                 + 0.992553 w_k1_DIS_s106
                                 + 1.01664 w_k1_FDX_s106
                                 + 0.881229 w_k1_IBM_s106
                                 + 1.0364 w_k1_INTC_s106 + 1.02045 w_k1_KO_s106
                                 + 1.14538 w_k1_MAT_s106
                                 + 0.985995 w_k1_MSFT_s106
                                 + 0.972451 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.981605 w_k1_ADM_s106
                                 + 0.785681 w_k1_AXP_s106
                                 + 1.1584 w_k1_DIS_s106
                                 + 0.927067 w_k1_FDX_s106
                                 + 1.20808 w_k1_IBM_s106
                                 + 0.913349 w_k1_INTC_s106
                                 + 1.10026 w_k1_KO_s106
                                 + 0.944963 w_k1_MAT_s106
                                 + 0.953399 w_k1_MSFT_s106
                                 + 0.967931 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.21469 w_k1_ADM_s106
                                 + 1.48175 w_k1_AXP_s106
                                 + 1.47292 w_k1_DIS_s106
                                 + 1.19727 w_k1_FDX_s106
                                 + 1.42537 w_k1_IBM_s106
                                 + 1.10513 w_k1_INTC_s106
                                 + 1.39874 w_k1_KO_s106 + 2.00274 w_k1_MAT_s106
                                 + 1.12923 w_k1_MSFT_s106
                                 + 1.24455 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.921439 w_k1_ADM_s106
                                 + 0.725994 w_k1_AXP_s106
                                 + 0.635842 w_k1_DIS_s106
                                 + 0.762341 w_k1_FDX_s106
                                 + 0.851974 w_k1_IBM_s106
                                 + 0.842503 w_k1_INTC_s106
                                 + 0.766251 w_k1_KO_s106
                                 + 0.602809 w_k1_MAT_s106
                                 + 0.756021 w_k1_MSFT_s106
                                 + 0.850308 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.01951 w_k1_ADM_s107
                                 + 1.53598 w_k1_AXP_s107
                                 + 1.43894 w_k1_DIS_s107
                                 + 1.07262 w_k1_FDX_s107
                                 + 1.81925 w_k1_IBM_s107
                                 + 0.820517 w_k1_INTC_s107
                                 + 1.30086 w_k1_KO_s107 + 2.56869 w_k1_MAT_s107
                                 + 0.954044 w_k1_MSFT_s107
                                 + 1.06275 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.625611 w_k1_ADM_s107
                                 + 0.549696 w_k1_AXP_s107
                                 + 0.576188 w_k1_DIS_s107
                                 + 0.761152 w_k1_FDX_s107
                                 + 0.716362 w_k1_IBM_s107
                                 + 0.738087 w_k1_INTC_s107
                                 + 0.619748 w_k1_KO_s107
                                 + 0.719342 w_k1_MAT_s107
                                 + 0.789973 w_k1_MSFT_s107
                                 + 0.660288 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.45298 w_k1_ADM_s107
                                 + 1.87773 w_k1_AXP_s107
                                 + 1.34243 w_k1_DIS_s107
                                 + 1.43669 w_k1_FDX_s107
                                 + 1.18463 w_k1_IBM_s107
                                 + 1.02607 w_k1_INTC_s107
                                 + 1.07981 w_k1_KO_s107 + 1.42929 w_k1_MAT_s107
                                 + 0.980377 w_k1_MSFT_s107
                                 + 1.08263 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.869902 w_k1_ADM_s107
                                 + 0.885979 w_k1_AXP_s107
                                 + 0.609401 w_k1_DIS_s107
                                 + 0.897961 w_k1_FDX_s107
                                 + 0.516929 w_k1_IBM_s107
                                 + 0.646382 w_k1_INTC_s107
                                 + 0.526873 w_k1_KO_s107
                                 + 0.634315 w_k1_MAT_s107
                                 + 0.688029 w_k1_MSFT_s107
                                 + 0.680707 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.950464 w_k1_ADM_s107
                                 + 0.788264 w_k1_AXP_s107
                                 + 0.876836 w_k1_DIS_s107
                                 + 0.996609 w_k1_FDX_s107
                                 + 0.684933 w_k1_IBM_s107
                                 + 0.848444 w_k1_INTC_s107
                                 + 0.864251 w_k1_KO_s107
                                 + 0.645445 w_k1_MAT_s107
                                 + 0.983986 w_k1_MSFT_s107
                                 + 0.876492 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.683167 w_k1_ADM_s107
                                 + 0.924742 w_k1_AXP_s107
                                 + 0.68461 w_k1_DIS_s107
                                 + 0.859332 w_k1_FDX_s107
                                 + 0.983087 w_k1_IBM_s107
                                 + 0.847239 w_k1_INTC_s107
                                 + 0.706473 w_k1_KO_s107
                                 + 1.21921 w_k1_MAT_s107
                                 + 0.718414 w_k1_MSFT_s107
                                 + 0.816992 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.0268 w_k1_ADM_s107
                                 + 0.945133 w_k1_AXP_s107
                                 + 1.00636 w_k1_DIS_s107
                                 + 0.946748 w_k1_FDX_s107
                                 + 1.28609 w_k1_IBM_s107
                                 + 0.701603 w_k1_INTC_s107
                                 + 1.16778 w_k1_KO_s107 + 1.337 w_k1_MAT_s107
                                 + 0.920127 w_k1_MSFT_s107
                                 + 1.08147 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.97483 w_k1_ADM_s107
                                 + 1.27003 w_k1_AXP_s107
                                 + 1.41135 w_k1_DIS_s107 + 1.2476 w_k1_FDX_s107
                                 + 1.12651 w_k1_IBM_s107
                                 + 1.11626 w_k1_INTC_s107
                                 + 1.20823 w_k1_KO_s107 + 1.35336 w_k1_MAT_s107
                                 + 1.33826 w_k1_MSFT_s107 + 1.1345 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.04157 w_k1_ADM_s110
                                 + 0.844581 w_k1_AXP_s110
                                 + 0.972245 w_k1_DIS_s110
                                 + 0.98237 w_k1_FDX_s110
                                 + 0.912746 w_k1_IBM_s110
                                 + 1.03814 w_k1_INTC_s110
                                 + 1.04124 w_k1_KO_s110 + 1.01654 w_k1_MAT_s110
                                 + 0.897472 w_k1_MSFT_s110
                                 + 1.04278 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.06632 w_k1_ADM_s110
                                 + 1.28579 w_k1_AXP_s110 + 1.0352 w_k1_DIS_s110
                                 + 1.09833 w_k1_FDX_s110
                                 + 1.37263 w_k1_IBM_s110
                                 + 1.15015 w_k1_INTC_s110
                                 + 1.28722 w_k1_KO_s110 + 2.2404 w_k1_MAT_s110
                                 + 1.04916 w_k1_MSFT_s110
                                 + 1.15289 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.28986 w_k1_ADM_s110
                                 + 1.00562 w_k1_AXP_s110 + 1.1731 w_k1_DIS_s110
                                 + 0.98721 w_k1_FDX_s110
                                 + 1.24032 w_k1_IBM_s110
                                 + 0.947829 w_k1_INTC_s110
                                 + 1.3956 w_k1_KO_s110 + 1.25639 w_k1_MAT_s110
                                 + 1.00079 w_k1_MSFT_s110
                                 + 1.17086 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.811906 w_k1_ADM_s110
                                 + 0.594464 w_k1_AXP_s110
                                 + 0.668419 w_k1_DIS_s110
                                 + 0.771251 w_k1_FDX_s110
                                 + 0.573317 w_k1_IBM_s110
                                 + 0.767357 w_k1_INTC_s110
                                 + 0.788873 w_k1_KO_s110
                                 + 0.543963 w_k1_MAT_s110
                                 + 0.742446 w_k1_MSFT_s110
                                 + 0.894712 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.19949 w_k1_ADM_s110
                                 + 1.19714 w_k1_AXP_s110
                                 + 1.31365 w_k1_DIS_s110
                                 + 1.36097 w_k1_FDX_s110
                                 + 0.688062 w_k1_IBM_s110
                                 + 1.1368 w_k1_INTC_s110
                                 + 0.984666 w_k1_KO_s110
                                 + 0.807793 w_k1_MAT_s110
                                 + 1.07399 w_k1_MSFT_s110
                                 + 1.18178 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.02016 w_k1_ADM_s110
                                 + 0.897894 w_k1_AXP_s110
                                 + 0.873282 w_k1_DIS_s110
                                 + 0.948585 w_k1_FDX_s110
                                 + 0.716266 w_k1_IBM_s110
                                 + 0.88947 w_k1_INTC_s110
                                 + 0.840442 w_k1_KO_s110
                                 + 0.798766 w_k1_MAT_s110
                                 + 0.883055 w_k1_MSFT_s110
                                 + 0.976383 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.33784 w_k1_ADM_s110
                                 + 1.15706 w_k1_AXP_s110
                                 + 1.42359 w_k1_DIS_s110
                                 + 1.11077 w_k1_FDX_s110 + 1.6672 w_k1_IBM_s110
                                 + 1.07515 w_k1_INTC_s110 + 1.6778 w_k1_KO_s110
                                 + 2.13103 w_k1_MAT_s110
                                 + 1.24424 w_k1_MSFT_s110
                                 + 1.43224 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.873355 w_k1_ADM_s110
                                 + 0.76067 w_k1_AXP_s110
                                 + 0.651071 w_k1_DIS_s110
                                 + 0.742283 w_k1_FDX_s110
                                 + 0.888719 w_k1_IBM_s110
                                 + 0.848074 w_k1_INTC_s110
                                 + 0.771403 w_k1_KO_s110
                                 + 0.89595 w_k1_MAT_s110
                                 + 0.772001 w_k1_MSFT_s110
                                 + 0.901153 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.686235 w_k1_ADM_s111
                                 + 0.624072 w_k1_AXP_s111
                                 + 0.66381 w_k1_DIS_s111
                                 + 0.698942 w_k1_FDX_s111
                                 + 0.763606 w_k1_IBM_s111
                                 + 0.771767 w_k1_INTC_s111
                                 + 0.725409 w_k1_KO_s111
                                 + 0.831768 w_k1_MAT_s111
                                 + 0.755965 w_k1_MSFT_s111
                                 + 0.699549 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.990963 w_k1_ADM_s111
                                 + 1.54265 w_k1_AXP_s111
                                 + 1.61245 w_k1_DIS_s111
                                 + 1.23474 w_k1_FDX_s111
                                 + 1.39308 w_k1_IBM_s111
                                 + 1.06848 w_k1_INTC_s111
                                 + 1.17973 w_k1_KO_s111 + 1.44878 w_k1_MAT_s111
                                 + 1.14986 w_k1_MSFT_s111
                                 + 1.26144 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.991354 w_k1_ADM_s111
                                 + 1.12328 w_k1_AXP_s111
                                 + 0.747208 w_k1_DIS_s111
                                 + 0.902329 w_k1_FDX_s111
                                 + 0.992029 w_k1_IBM_s111
                                 + 0.765126 w_k1_INTC_s111
                                 + 0.730743 w_k1_KO_s111
                                 + 1.03035 w_k1_MAT_s111
                                 + 0.748613 w_k1_MSFT_s111
                                 + 0.983938 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.589183 w_k1_ADM_s111
                                 + 0.785694 w_k1_AXP_s111
                                 + 0.909168 w_k1_DIS_s111
                                 + 0.873454 w_k1_FDX_s111
                                 + 1.04744 w_k1_IBM_s111
                                 + 1.14315 w_k1_INTC_s111
                                 + 0.945358 w_k1_KO_s111
                                 + 0.989656 w_k1_MAT_s111
                                 + 0.9664 w_k1_MSFT_s111
                                 + 0.969139 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.731635 w_k1_ADM_s111
                                 + 1.10973 w_k1_AXP_s111
                                 + 0.823497 w_k1_DIS_s111
                                 + 0.949259 w_k1_FDX_s111
                                 + 1.42898 w_k1_IBM_s111
                                 + 1.10283 w_k1_INTC_s111
                                 + 0.951355 w_k1_KO_s111
                                 + 1.61438 w_k1_MAT_s111
                                 + 0.901119 w_k1_MSFT_s111
                                 + 1.01181 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.978668 w_k1_ADM_s111
                                 + 1.17645 w_k1_AXP_s111
                                 + 1.24445 w_k1_DIS_s111
                                 + 1.19885 w_k1_FDX_s111
                                 + 0.979327 w_k1_IBM_s111
                                 + 1.02585 w_k1_INTC_s111
                                 + 1.03865 w_k1_KO_s111 + 1.23262 w_k1_MAT_s111
                                 + 1.15884 w_k1_MSFT_s111
                                 + 1.18012 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.981709 w_k1_ADM_s111
                                 + 1.12258 w_k1_AXP_s111
                                 + 1.04679 w_k1_DIS_s111
                                 + 1.06897 w_k1_FDX_s111
                                 + 0.902513 w_k1_IBM_s111
                                 + 0.984038 w_k1_INTC_s111
                                 + 0.879157 w_k1_KO_s111
                                 + 1.19024 w_k1_MAT_s111
                                 + 1.03163 w_k1_MSFT_s111
                                 + 0.935664 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.633616 w_k1_ADM_s111
                                 + 0.912432 w_k1_AXP_s111
                                 + 0.884891 w_k1_DIS_s111
                                 + 0.923473 w_k1_FDX_s111
                                 + 0.74026 w_k1_IBM_s111
                                 + 0.988121 w_k1_INTC_s111
                                 + 0.76389 w_k1_KO_s111
                                 + 0.857865 w_k1_MAT_s111
                                 + 0.965957 w_k1_MSFT_s111
                                 + 0.877377 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07293 w_k1_ADM_s112
                                 + 1.29159 w_k1_AXP_s112
                                 + 1.36946 w_k1_DIS_s112
                                 + 1.12127 w_k1_FDX_s112
                                 + 1.26929 w_k1_IBM_s112
                                 + 1.20274 w_k1_INTC_s112
                                 + 1.13862 w_k1_KO_s112
                                 + 0.924652 w_k1_MAT_s112
                                 + 1.10811 w_k1_MSFT_s112
                                 + 1.14791 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.892657 w_k1_ADM_s112
                                 + 1.01322 w_k1_AXP_s112
                                 + 1.09907 w_k1_DIS_s112
                                 + 0.870867 w_k1_FDX_s112
                                 + 1.19257 w_k1_IBM_s112
                                 + 0.933351 w_k1_INTC_s112
                                 + 1.08805 w_k1_KO_s112
                                 + 0.729919 w_k1_MAT_s112
                                 + 1.03333 w_k1_MSFT_s112
                                 + 1.00748 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.86236 w_k1_ADM_s112
                                 + 0.965773 w_k1_AXP_s112
                                 + 0.654111 w_k1_DIS_s112
                                 + 0.860659 w_k1_FDX_s112
                                 + 0.779225 w_k1_IBM_s112
                                 + 1.02398 w_k1_INTC_s112
                                 + 0.676428 w_k1_KO_s112
                                 + 0.558936 w_k1_MAT_s112
                                 + 0.82313 w_k1_MSFT_s112
                                 + 0.85662 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.06871 w_k1_ADM_s112
                                 + 1.29758 w_k1_AXP_s112
                                 + 1.34856 w_k1_DIS_s112
                                 + 1.22027 w_k1_FDX_s112
                                 + 0.81399 w_k1_IBM_s112
                                 + 1.13865 w_k1_INTC_s112
                                 + 0.976388 w_k1_KO_s112
                                 + 0.730799 w_k1_MAT_s112
                                 + 1.24692 w_k1_MSFT_s112 + 1.0518 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.815646 w_k1_ADM_s112
                                 + 0.679464 w_k1_AXP_s112
                                 + 0.70003 w_k1_DIS_s112
                                 + 0.685425 w_k1_FDX_s112
                                 + 0.905377 w_k1_IBM_s112
                                 + 0.811134 w_k1_INTC_s112
                                 + 0.790091 w_k1_KO_s112
                                 + 0.707843 w_k1_MAT_s112
                                 + 0.794656 w_k1_MSFT_s112
                                 + 0.820527 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.903534 w_k1_ADM_s112
                                 + 1.04716 w_k1_AXP_s112
                                 + 0.811305 w_k1_DIS_s112
                                 + 0.763611 w_k1_FDX_s112
                                 + 1.24987 w_k1_IBM_s112
                                 + 1.02029 w_k1_INTC_s112
                                 + 0.987611 w_k1_KO_s112
                                 + 0.989658 w_k1_MAT_s112
                                 + 0.801886 w_k1_MSFT_s112
                                 + 0.98687 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.878857 w_k1_ADM_s112
                                 + 1.18627 w_k1_AXP_s112
                                 + 1.02342 w_k1_DIS_s112
                                 + 0.983496 w_k1_FDX_s112
                                 + 0.883735 w_k1_IBM_s112
                                 + 0.988921 w_k1_INTC_s112
                                 + 0.940304 w_k1_KO_s112
                                 + 1.10242 w_k1_MAT_s112
                                 + 1.02459 w_k1_MSFT_s112
                                 + 1.02172 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.977717 w_k1_ADM_s112
                                 + 0.888166 w_k1_AXP_s112
                                 + 0.924535 w_k1_DIS_s112
                                 + 0.875079 w_k1_FDX_s112
                                 + 0.86181 w_k1_IBM_s112
                                 + 0.952583 w_k1_INTC_s112
                                 + 0.917459 w_k1_KO_s112
                                 + 0.698695 w_k1_MAT_s112
                                 + 0.95121 w_k1_MSFT_s112
                                 + 0.937341 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.912254 w_k1_ADM_s113
                                 + 0.956841 w_k1_AXP_s113
                                 + 0.804521 w_k1_DIS_s113
                                 + 0.924951 w_k1_FDX_s113
                                 + 0.819321 w_k1_IBM_s113
                                 + 0.88119 w_k1_INTC_s113
                                 + 0.730887 w_k1_KO_s113
                                 + 0.818073 w_k1_MAT_s113
                                 + 0.915319 w_k1_MSFT_s113
                                 + 0.989515 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.35363 w_k1_ADM_s113
                                 + 1.57324 w_k1_AXP_s113
                                 + 1.53488 w_k1_DIS_s113
                                 + 1.45658 w_k1_FDX_s113
                                 + 0.982585 w_k1_IBM_s113
                                 + 1.48826 w_k1_INTC_s113
                                 + 1.14421 w_k1_KO_s113 + 1.07868 w_k1_MAT_s113
                                 + 1.42773 w_k1_MSFT_s113
                                 + 1.22567 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.813199 w_k1_ADM_s113
                                 + 1.05765 w_k1_AXP_s113
                                 + 0.802986 w_k1_DIS_s113
                                 + 0.834257 w_k1_FDX_s113
                                 + 1.22337 w_k1_IBM_s113
                                 + 1.04008 w_k1_INTC_s113
                                 + 1.01451 w_k1_KO_s113 + 1.16926 w_k1_MAT_s113
                                 + 0.939009 w_k1_MSFT_s113
                                 + 1.00021 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.27425 w_k1_ADM_s113
                                 + 1.32339 w_k1_AXP_s113
                                 + 1.09344 w_k1_DIS_s113
                                 + 1.25673 w_k1_FDX_s113
                                 + 0.856728 w_k1_IBM_s113
                                 + 1.0681 w_k1_INTC_s113
                                 + 0.875228 w_k1_KO_s113
                                 + 0.840175 w_k1_MAT_s113
                                 + 1.11961 w_k1_MSFT_s113
                                 + 1.08112 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.935873 w_k1_ADM_s113
                                 + 1.07018 w_k1_AXP_s113
                                 + 0.680265 w_k1_DIS_s113
                                 + 0.899292 w_k1_FDX_s113
                                 + 1.01852 w_k1_IBM_s113
                                 + 1.0063 w_k1_INTC_s113
                                 + 0.988437 w_k1_KO_s113
                                 + 1.21745 w_k1_MAT_s113
                                 + 0.929165 w_k1_MSFT_s113
                                 + 0.979906 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.913153 w_k1_ADM_s113
                                 + 0.719205 w_k1_AXP_s113
                                 + 0.637124 w_k1_DIS_s113
                                 + 0.821957 w_k1_FDX_s113
                                 + 0.638245 w_k1_IBM_s113
                                 + 0.767731 w_k1_INTC_s113
                                 + 0.725309 w_k1_KO_s113
                                 + 0.655381 w_k1_MAT_s113
                                 + 0.791104 w_k1_MSFT_s113
                                 + 0.811354 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.41787 w_k1_ADM_s113
                                 + 1.50042 w_k1_AXP_s113
                                 + 1.45321 w_k1_DIS_s113
                                 + 1.25934 w_k1_FDX_s113
                                 + 1.32795 w_k1_IBM_s113
                                 + 0.945897 w_k1_INTC_s113
                                 + 1.33533 w_k1_KO_s113 + 1.68411 w_k1_MAT_s113
                                 + 1.16176 w_k1_MSFT_s113 + 1.261 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.12084 w_k1_ADM_s113
                                 + 1.24292 w_k1_AXP_s113
                                 + 0.999194 w_k1_DIS_s113
                                 + 0.911471 w_k1_FDX_s113
                                 + 1.69048 w_k1_IBM_s113
                                 + 0.977076 w_k1_INTC_s113
                                 + 1.28524 w_k1_KO_s113 + 2.04577 w_k1_MAT_s113
                                 + 1.09145 w_k1_MSFT_s113 + 1.0969 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.848673 w_k1_ADM_s114
                                 + 0.963778 w_k1_AXP_s114
                                 + 1.10982 w_k1_DIS_s114
                                 + 0.989053 w_k1_FDX_s114
                                 + 1.03661 w_k1_IBM_s114
                                 + 1.14444 w_k1_INTC_s114
                                 + 1.11748 w_k1_KO_s114 + 1.34565 w_k1_MAT_s114
                                 + 1.0711 w_k1_MSFT_s114 + 1.07101 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.972904 w_k1_ADM_s114
                                 + 0.984595 w_k1_AXP_s114
                                 + 1.30777 w_k1_DIS_s114
                                 + 0.964513 w_k1_FDX_s114
                                 + 1.43939 w_k1_IBM_s114
                                 + 0.984221 w_k1_INTC_s114
                                 + 1.38002 w_k1_KO_s114 + 1.6088 w_k1_MAT_s114
                                 + 1.11143 w_k1_MSFT_s114
                                 + 1.07633 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.993569 w_k1_ADM_s114
                                 + 0.860007 w_k1_AXP_s114
                                 + 0.889155 w_k1_DIS_s114
                                 + 0.882038 w_k1_FDX_s114
                                 + 0.744361 w_k1_IBM_s114
                                 + 0.906969 w_k1_INTC_s114
                                 + 0.850239 w_k1_KO_s114
                                 + 0.972156 w_k1_MAT_s114
                                 + 0.892919 w_k1_MSFT_s114
                                 + 0.930832 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.825645 w_k1_ADM_s114
                                 + 0.81569 w_k1_AXP_s114
                                 + 0.855046 w_k1_DIS_s114
                                 + 0.820523 w_k1_FDX_s114
                                 + 1.04063 w_k1_IBM_s114
                                 + 1.01601 w_k1_INTC_s114
                                 + 0.972095 w_k1_KO_s114
                                 + 0.823383 w_k1_MAT_s114
                                 + 0.896516 w_k1_MSFT_s114
                                 + 0.94453 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.90799 w_k1_ADM_s114
                                 + 1.0626 w_k1_AXP_s114 + 1.51671 w_k1_DIS_s114
                                 + 1.12656 w_k1_FDX_s114
                                 + 1.06975 w_k1_IBM_s114
                                 + 1.24057 w_k1_INTC_s114
                                 + 1.13333 w_k1_KO_s114 + 1.27097 w_k1_MAT_s114
                                 + 1.26656 w_k1_MSFT_s114
                                 + 1.03698 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.33903 w_k1_ADM_s114
                                 + 1.01551 w_k1_AXP_s114 + 1.0066 w_k1_DIS_s114
                                 + 1.04172 w_k1_FDX_s114
                                 + 0.87009 w_k1_IBM_s114
                                 + 0.999976 w_k1_INTC_s114
                                 + 0.99145 w_k1_KO_s114
                                 + 0.801362 w_k1_MAT_s114
                                 + 1.0343 w_k1_MSFT_s114
                                 + 0.964171 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.841193 w_k1_ADM_s114
                                 + 0.664056 w_k1_AXP_s114
                                 + 0.711086 w_k1_DIS_s114
                                 + 0.818386 w_k1_FDX_s114
                                 + 0.720741 w_k1_IBM_s114
                                 + 0.877299 w_k1_INTC_s114
                                 + 0.754463 w_k1_KO_s114
                                 + 0.742958 w_k1_MAT_s114
                                 + 0.87971 w_k1_MSFT_s114
                                 + 0.792668 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.45444 w_k1_ADM_s114
                                 + 1.34009 w_k1_AXP_s114
                                 + 1.48063 w_k1_DIS_s114 + 1.3091 w_k1_FDX_s114
                                 + 1.07383 w_k1_IBM_s114
                                 + 1.40556 w_k1_INTC_s114
                                 + 1.07729 w_k1_KO_s114
                                 + 0.946232 w_k1_MAT_s114
                                 + 1.23019 w_k1_MSFT_s114
                                 + 1.08857 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.33921 w_k1_ADM_s115
                                 + 1.2464 w_k1_AXP_s115 + 1.15866 w_k1_DIS_s115
                                 + 1.12939 w_k1_FDX_s115
                                 + 1.80123 w_k1_IBM_s115
                                 + 1.23929 w_k1_INTC_s115
                                 + 1.49157 w_k1_KO_s115 + 1.36468 w_k1_MAT_s115
                                 + 1.07495 w_k1_MSFT_s115
                                 + 1.03076 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.99707 w_k1_ADM_s115
                                 + 0.954308 w_k1_AXP_s115
                                 + 0.831307 w_k1_DIS_s115
                                 + 0.978463 w_k1_FDX_s115
                                 + 0.662362 w_k1_IBM_s115
                                 + 0.910363 w_k1_INTC_s115
                                 + 0.813732 w_k1_KO_s115
                                 + 0.672911 w_k1_MAT_s115
                                 + 0.860155 w_k1_MSFT_s115
                                 + 0.843998 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.0693 w_k1_ADM_s115
                                 + 0.792904 w_k1_AXP_s115
                                 + 0.867875 w_k1_DIS_s115
                                 + 0.947169 w_k1_FDX_s115
                                 + 0.871523 w_k1_IBM_s115
                                 + 0.875125 w_k1_INTC_s115
                                 + 0.835573 w_k1_KO_s115
                                 + 0.60447 w_k1_MAT_s115
                                 + 0.90442 w_k1_MSFT_s115
                                 + 0.873621 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.765239 w_k1_ADM_s115
                                 + 0.540282 w_k1_AXP_s115
                                 + 0.59294 w_k1_DIS_s115
                                 + 0.749874 w_k1_FDX_s115
                                 + 0.793525 w_k1_IBM_s115
                                 + 0.800383 w_k1_INTC_s115
                                 + 0.649648 w_k1_KO_s115
                                 + 0.655045 w_k1_MAT_s115
                                 + 0.710542 w_k1_MSFT_s115
                                 + 0.731721 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.05815 w_k1_ADM_s115
                                 + 0.94978 w_k1_AXP_s115
                                 + 0.962275 w_k1_DIS_s115
                                 + 0.889626 w_k1_FDX_s115
                                 + 1.24519 w_k1_IBM_s115
                                 + 0.930349 w_k1_INTC_s115
                                 + 1.11154 w_k1_KO_s115 + 1.01704 w_k1_MAT_s115
                                 + 0.824322 w_k1_MSFT_s115
                                 + 0.976244 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.17142 w_k1_ADM_s115
                                 + 1.09981 w_k1_AXP_s115
                                 + 1.17743 w_k1_DIS_s115
                                 + 1.15691 w_k1_FDX_s115
                                 + 0.795358 w_k1_IBM_s115
                                 + 1.09853 w_k1_INTC_s115
                                 + 1.02583 w_k1_KO_s115
                                 + 0.736127 w_k1_MAT_s115
                                 + 1.02415 w_k1_MSFT_s115
                                 + 1.05936 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.04636 w_k1_ADM_s115
                                 + 1.20573 w_k1_AXP_s115
                                 + 1.16765 w_k1_DIS_s115
                                 + 1.15436 w_k1_FDX_s115
                                 + 1.25417 w_k1_IBM_s115
                                 + 1.08678 w_k1_INTC_s115
                                 + 1.08355 w_k1_KO_s115 + 1.4336 w_k1_MAT_s115
                                 + 0.99098 w_k1_MSFT_s115
                                 + 1.10069 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.35162 w_k1_ADM_s115
                                 + 1.59128 w_k1_AXP_s115
                                 + 1.57605 w_k1_DIS_s115 + 1.2928 w_k1_FDX_s115
                                 + 1.27979 w_k1_IBM_s115
                                 + 1.38216 w_k1_INTC_s115
                                 + 1.24604 w_k1_KO_s115
                                 + 0.922458 w_k1_MAT_s115
                                 + 1.09539 w_k1_MSFT_s115 + 1.1629 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.32176 w_k1_ADM_s116
                                 + 1.08566 w_k1_AXP_s116
                                 + 1.22525 w_k1_DIS_s116
                                 + 1.00947 w_k1_FDX_s116
                                 + 1.57882 w_k1_IBM_s116
                                 + 1.19632 w_k1_INTC_s116
                                 + 1.17995 w_k1_KO_s116 + 1.28323 w_k1_MAT_s116
                                 + 1.13354 w_k1_MSFT_s116 + 1.0484 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.954462 w_k1_ADM_s116
                                 + 0.930045 w_k1_AXP_s116
                                 + 0.840471 w_k1_DIS_s116
                                 + 0.869504 w_k1_FDX_s116
                                 + 0.931901 w_k1_IBM_s116
                                 + 1.06128 w_k1_INTC_s116 + 1.0193 w_k1_KO_s116
                                 + 0.889325 w_k1_MAT_s116
                                 + 0.88569 w_k1_MSFT_s116
                                 + 0.932008 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.96241 w_k1_ADM_s116
                                 + 0.647596 w_k1_AXP_s116
                                 + 0.584663 w_k1_DIS_s116
                                 + 0.79083 w_k1_FDX_s116
                                 + 0.659718 w_k1_IBM_s116
                                 + 0.988744 w_k1_INTC_s116
                                 + 0.734968 w_k1_KO_s116
                                 + 0.515576 w_k1_MAT_s116
                                 + 0.880777 w_k1_MSFT_s116
                                 + 0.843135 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.985818 w_k1_ADM_s116
                                 + 1.08503 w_k1_AXP_s116
                                 + 1.25454 w_k1_DIS_s116
                                 + 1.18105 w_k1_FDX_s116 + 1.0315 w_k1_IBM_s116
                                 + 1.34835 w_k1_INTC_s116
                                 + 1.06555 w_k1_KO_s116 + 1.25396 w_k1_MAT_s116
                                 + 1.30275 w_k1_MSFT_s116
                                 + 1.07245 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.759217 w_k1_ADM_s116
                                 + 0.684951 w_k1_AXP_s116
                                 + 1.06099 w_k1_DIS_s116
                                 + 0.954198 w_k1_FDX_s116
                                 + 0.84555 w_k1_IBM_s116
                                 + 1.28888 w_k1_INTC_s116
                                 + 0.936761 w_k1_KO_s116
                                 + 0.889524 w_k1_MAT_s116
                                 + 1.18818 w_k1_MSFT_s116
                                 + 1.00923 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.28645 w_k1_ADM_s116
                                 + 1.0225 w_k1_AXP_s116 + 1.16832 w_k1_DIS_s116
                                 + 1.1036 w_k1_FDX_s116
                                 + 0.776361 w_k1_IBM_s116
                                 + 1.16888 w_k1_INTC_s116
                                 + 0.990026 w_k1_KO_s116
                                 + 0.896997 w_k1_MAT_s116
                                 + 1.14519 w_k1_MSFT_s116
                                 + 1.06238 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.766036 w_k1_ADM_s116
                                 + 0.702855 w_k1_AXP_s116
                                 + 0.673194 w_k1_DIS_s116
                                 + 0.693964 w_k1_FDX_s116
                                 + 1.07345 w_k1_IBM_s116
                                 + 0.85288 w_k1_INTC_s116
                                 + 0.888201 w_k1_KO_s116
                                 + 1.12301 w_k1_MAT_s116
                                 + 0.817319 w_k1_MSFT_s116
                                 + 0.850678 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.973294 w_k1_ADM_s116
                                 + 0.828434 w_k1_AXP_s116
                                 + 1.12646 w_k1_DIS_s116
                                 + 0.945284 w_k1_FDX_s116
                                 + 1.23335 w_k1_IBM_s116
                                 + 1.15132 w_k1_INTC_s116
                                 + 1.03279 w_k1_KO_s116 + 1.13357 w_k1_MAT_s116
                                 + 1.20806 w_k1_MSFT_s116 + 1.104 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.948084 w_k1_ADM_s117
                                 + 0.987494 w_k1_AXP_s117
                                 + 1.13829 w_k1_DIS_s117
                                 + 1.09313 w_k1_FDX_s117
                                 + 0.825955 w_k1_IBM_s117
                                 + 0.986399 w_k1_INTC_s117
                                 + 1.04101 w_k1_KO_s117
                                 + 0.516272 w_k1_MAT_s117
                                 + 0.993643 w_k1_MSFT_s117
                                 + 0.930503 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.06432 w_k1_ADM_s117
                                 + 1.43551 w_k1_AXP_s117
                                 + 1.60811 w_k1_DIS_s117
                                 + 1.38936 w_k1_FDX_s117
                                 + 1.06194 w_k1_IBM_s117
                                 + 1.42057 w_k1_INTC_s117
                                 + 1.29517 w_k1_KO_s117 + 1.03714 w_k1_MAT_s117
                                 + 1.42583 w_k1_MSFT_s117
                                 + 1.19551 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.18548 w_k1_ADM_s117
                                 + 1.17317 w_k1_AXP_s117
                                 + 1.33203 w_k1_DIS_s117
                                 + 1.28639 w_k1_FDX_s117
                                 + 0.66838 w_k1_IBM_s117
                                 + 1.22511 w_k1_INTC_s117
                                 + 1.01891 w_k1_KO_s117
                                 + 0.381941 w_k1_MAT_s117
                                 + 1.07277 w_k1_MSFT_s117
                                 + 1.01897 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.953973 w_k1_ADM_s117
                                 + 1.06816 w_k1_AXP_s117
                                 + 1.01717 w_k1_DIS_s117
                                 + 0.956864 w_k1_FDX_s117
                                 + 1.16377 w_k1_IBM_s117
                                 + 0.92037 w_k1_INTC_s117 + 1.2529 w_k1_KO_s117
                                 + 1.23415 w_k1_MAT_s117
                                 + 0.894375 w_k1_MSFT_s117
                                 + 1.00375 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.852132 w_k1_ADM_s117
                                 + 1.13537 w_k1_AXP_s117
                                 + 1.19055 w_k1_DIS_s117
                                 + 0.984517 w_k1_FDX_s117
                                 + 1.64501 w_k1_IBM_s117
                                 + 1.11769 w_k1_INTC_s117 + 1.4152 w_k1_KO_s117
                                 + 1.80421 w_k1_MAT_s117
                                 + 1.08668 w_k1_MSFT_s117
                                 + 1.04295 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.851473 w_k1_ADM_s117
                                 + 0.761563 w_k1_AXP_s117
                                 + 0.915249 w_k1_DIS_s117
                                 + 0.882719 w_k1_FDX_s117
                                 + 0.904452 w_k1_IBM_s117
                                 + 0.795747 w_k1_INTC_s117
                                 + 0.956237 w_k1_KO_s117
                                 + 0.674059 w_k1_MAT_s117
                                 + 0.96009 w_k1_MSFT_s117
                                 + 0.870554 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.730182 w_k1_ADM_s117
                                 + 0.85278 w_k1_AXP_s117
                                 + 0.629712 w_k1_DIS_s117
                                 + 0.691454 w_k1_FDX_s117
                                 + 0.8557 w_k1_IBM_s117
                                 + 0.746961 w_k1_INTC_s117
                                 + 0.751647 w_k1_KO_s117
                                 + 0.83522 w_k1_MAT_s117
                                 + 0.679678 w_k1_MSFT_s117
                                 + 0.726294 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.038 w_k1_ADM_s117
                                 + 1.50011 w_k1_AXP_s117
                                 + 1.31678 w_k1_DIS_s117
                                 + 1.25287 w_k1_FDX_s117
                                 + 1.51538 w_k1_IBM_s117
                                 + 1.02504 w_k1_INTC_s117
                                 + 1.30942 w_k1_KO_s117 + 1.40111 w_k1_MAT_s117
                                 + 1.04424 w_k1_MSFT_s117
                                 + 1.04596 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00322383551903029 w_k1_ADM_s000
                                 + 0.00921083509567112 w_k1_ADM_s001
                                 + 0.00447946204739476 w_k1_ADM_s002
                                 + 0.00387621404503145 w_k1_ADM_s003
                                 + 0.0058742457044692 w_k1_ADM_s004
                                 + 0.0150045170122073 w_k1_ADM_s005
                                 + 0.0109157371530106 w_k1_ADM_s006
                                 + 0.0115922927717699 w_k1_ADM_s007
                                 + 0.0739277410812521 w_k1_ADM_s010
                                 + 0.0355044078328588 w_k1_ADM_s011
                                 + 0.0230857694251613 w_k1_ADM_s012
                                 + 0.0124530932582315 w_k1_ADM_s013
                                 + 0.0474026329671118 w_k1_ADM_s014
                                 + 0.0224852366881163 w_k1_ADM_s015
                                 + 0.0331266217926865 w_k1_ADM_s016
                                 + 0.0262945191352058 w_k1_ADM_s017
                                 + 0.0171705945419069 w_k1_ADM_s100
                                 + 0.0808030981069158 w_k1_ADM_s101
                                 + 0.0384314563292447 w_k1_ADM_s102
                                 + 0.0843359673806816 w_k1_ADM_s103
                                 + 0.040813898345881 w_k1_ADM_s104
                                 + 0.0819478535490862 w_k1_ADM_s105
                                 + 0.0488223588661737 w_k1_ADM_s106
                                 + 0.0214364175937858 w_k1_ADM_s107
                                 + 0.0415643920849852 w_k1_ADM_s110
                                 + 0.0196407331026883 w_k1_ADM_s111
                                 + 0.0299256300568192 w_k1_ADM_s112
                                 + 0.0479616976714973 w_k1_ADM_s113
                                 + 0.0617231136699215 w_k1_ADM_s114
                                 + 0.022007967632124 w_k1_ADM_s115
                                 + 0.02487101580535 w_k1_ADM_s116
                                 + 0.026135515923992 w_k1_ADM_s117
                                 + 0.00340847987449354 w_k1_AXP_s000
                                 + 0.0100245261365296 w_k1_AXP_s001
                                 + 0.00506324805537125 w_k1_AXP_s002
                                 + 0.00396569176951238 w_k1_AXP_s003
                                 + 0.00619189056528283 w_k1_AXP_s004
                                 + 0.0137816441244513 w_k1_AXP_s005
                                 + 0.01012065085313 w_k1_AXP_s006
                                 + 0.0129644464433316 w_k1_AXP_s007
                                 + 0.0674976217292196 w_k1_AXP_s010
                                 + 0.0347896933667264 w_k1_AXP_s011
                                 + 0.0258200508097916 w_k1_AXP_s012
                                 + 0.0157768489839706 w_k1_AXP_s013
                                 + 0.0451124344589605 w_k1_AXP_s014
                                 + 0.0209918402971366 w_k1_AXP_s015
                                 + 0.0328434627609535 w_k1_AXP_s016
                                 + 0.0247209397158814 w_k1_AXP_s017
                                 + 0.018523824010155 w_k1_AXP_s100
                                 + 0.0694426192151968 w_k1_AXP_s101
                                 + 0.0420140587221439 w_k1_AXP_s102
                                 + 0.0803293704249978 w_k1_AXP_s103
                                 + 0.0427992795666763 w_k1_AXP_s104
                                 + 0.0817092290180319 w_k1_AXP_s105
                                 + 0.048766292686766 w_k1_AXP_s106
                                 + 0.023870084053204 w_k1_AXP_s107
                                 + 0.0370363583942507 w_k1_AXP_s110
                                 + 0.0250594875816741 w_k1_AXP_s111
                                 + 0.0345492583374346 w_k1_AXP_s112
                                 + 0.0512662304614023 w_k1_AXP_s113
                                 + 0.0578471029595201 w_k1_AXP_s114
                                 + 0.0210448565347402 w_k1_AXP_s115
                                 + 0.0214576731549753 w_k1_AXP_s116
                                 + 0.0294823883748896 w_k1_AXP_s117
                                 + 0.00310653824271963 w_k1_DIS_s000
                                 + 0.00987269543175325 w_k1_DIS_s001
                                 + 0.005127748749282 w_k1_DIS_s002
                                 + 0.00364947184065021 w_k1_DIS_s003
                                 + 0.0058700363653269 w_k1_DIS_s004
                                 + 0.0149752215567493 w_k1_DIS_s005
                                 + 0.0106463362913967 w_k1_DIS_s006
                                 + 0.0109594954548262 w_k1_DIS_s007
                                 + 0.0652741660927126 w_k1_DIS_s010
                                 + 0.034098536756079 w_k1_DIS_s011
                                 + 0.0248306140040372 w_k1_DIS_s012
                                 + 0.0170585054438512 w_k1_DIS_s013
                                 + 0.0453182406946666 w_k1_DIS_s014
                                 + 0.0221970121769734 w_k1_DIS_s015
                                 + 0.0321227578040029 w_k1_DIS_s016
                                 + 0.0271242903126207 w_k1_DIS_s017
                                 + 0.0188888077729553 w_k1_DIS_s100
                                 + 0.0675965831145249 w_k1_DIS_s101
                                 + 0.0429443978763293 w_k1_DIS_s102
                                 + 0.0851933429204094 w_k1_DIS_s103
                                 + 0.0415451491638692 w_k1_DIS_s104
                                 + 0.0858521025544461 w_k1_DIS_s105
                                 + 0.0499786228431418 w_k1_DIS_s106
                                 + 0.0229743961674518 w_k1_DIS_s107
                                 + 0.0392655825681122 w_k1_DIS_s110
                                 + 0.024708697046813 w_k1_DIS_s111
                                 + 0.0341496871423723 w_k1_DIS_s112
                                 + 0.0436887789706147 w_k1_DIS_s113
                                 + 0.0665318833581432 w_k1_DIS_s114
                                 + 0.0212927794520525 w_k1_DIS_s115
                                 + 0.0243297427457012 w_k1_DIS_s116
                                 + 0.0304129819334678 w_k1_DIS_s117
                                 + 0.00332240959081966 w_k1_FDX_s000
                                 + 0.00960863423203768 w_k1_FDX_s001
                                 + 0.00498331998130433 w_k1_FDX_s002
                                 + 0.00372903014977887 w_k1_FDX_s003
                                 + 0.00582401377485895 w_k1_FDX_s004
                                 + 0.0141199256553291 w_k1_FDX_s005
                                 + 0.00995237054167644 w_k1_FDX_s006
                                 + 0.0113525216374296 w_k1_FDX_s007
                                 + 0.066194659071977 w_k1_FDX_s010
                                 + 0.0311668511456965 w_k1_FDX_s011
                                 + 0.0260782459072923 w_k1_FDX_s012
                                 + 0.0154789742343234 w_k1_FDX_s013
                                 + 0.0462454793115749 w_k1_FDX_s014
                                 + 0.0220621780016354 w_k1_FDX_s015
                                 + 0.0317334039090793 w_k1_FDX_s016
                                 + 0.0256037621899012 w_k1_FDX_s017
                                 + 0.0186493268340426 w_k1_FDX_s100
                                 + 0.0678355462774269 w_k1_FDX_s101
                                 + 0.0400907791490351 w_k1_FDX_s102
                                 + 0.0865334631435957 w_k1_FDX_s103
                                 + 0.0414378703083849 w_k1_FDX_s104
                                 + 0.0847376467278659 w_k1_FDX_s105
                                 + 0.0493113445826074 w_k1_FDX_s106
                                 + 0.0233268777543633 w_k1_FDX_s107
                                 + 0.0380887148286567 w_k1_FDX_s110
                                 + 0.0237602221569342 w_k1_FDX_s111
                                 + 0.0306718270922732 w_k1_FDX_s112
                                 + 0.0460737333313676 w_k1_FDX_s113
                                 + 0.0600015651336765 w_k1_FDX_s114
                                 + 0.0209987734678348 w_k1_FDX_s115
                                 + 0.0232332904360204 w_k1_FDX_s116
                                 + 0.0287082524698153 w_k1_FDX_s117
                                 + 0.0033069253973916 w_k1_IBM_s000
                                 + 0.0102509661844905 w_k1_IBM_s001
                                 + 0.00503724476557357 w_k1_IBM_s002
                                 + 0.00374953988278951 w_k1_IBM_s003
                                 + 0.00601542721822284 w_k1_IBM_s004
                                 + 0.0147264174706071 w_k1_IBM_s005
                                 + 0.0102841387750636 w_k1_IBM_s006
                                 + 0.0122367937884757 w_k1_IBM_s007
                                 + 0.06905713688699 w_k1_IBM_s010
                                 + 0.0327537095007306 w_k1_IBM_s011
                                 + 0.0245185484560403 w_k1_IBM_s012
                                 + 0.0158248064592943 w_k1_IBM_s013
                                 + 0.0471625989134116 w_k1_IBM_s014
                                 + 0.0213374077519269 w_k1_IBM_s015
                                 + 0.0334058070585467 w_k1_IBM_s016
                                 + 0.0260395180888093 w_k1_IBM_s017
                                 + 0.0198104040398945 w_k1_IBM_s100
                                 + 0.0685053871289137 w_k1_IBM_s101
                                 + 0.041883241962893 w_k1_IBM_s102
                                 + 0.0866600882839163 w_k1_IBM_s103
                                 + 0.0408070535452048 w_k1_IBM_s104
                                 + 0.0835153463305519 w_k1_IBM_s105
                                 + 0.0479582502274754 w_k1_IBM_s106
                                 + 0.0224245560804638 w_k1_IBM_s107
                                 + 0.0394222684878999 w_k1_IBM_s110
                                 + 0.0245348234770726 w_k1_IBM_s111
                                 + 0.0311922431107148 w_k1_IBM_s112
                                 + 0.0454301024479613 w_k1_IBM_s113
                                 + 0.0618422926324048 w_k1_IBM_s114
                                 + 0.0210363409159699 w_k1_IBM_s115
                                 + 0.0250962368829224 w_k1_IBM_s116
                                 + 0.0287272165863479 w_k1_IBM_s117
                                 + 0.0032536278765702 w_k1_INTC_s000
                                 + 0.0082820281077391 w_k1_INTC_s001
                                 + 0.00579045874262976 w_k1_INTC_s002
                                 + 0.00395593316825636 w_k1_INTC_s003
                                 + 0.00725659461579617 w_k1_INTC_s004
                                 + 0.0158113688175012 w_k1_INTC_s005
                                 + 0.00932409452371893 w_k1_INTC_s006
                                 + 0.0124525957849492 w_k1_INTC_s007
                                 + 0.0708755135058905 w_k1_INTC_s010
                                 + 0.0318886005319386 w_k1_INTC_s011
                                 + 0.0243220163467159 w_k1_INTC_s012
                                 + 0.015223678169211 w_k1_INTC_s013
                                 + 0.040515440856697 w_k1_INTC_s014
                                 + 0.0212506181113273 w_k1_INTC_s015
                                 + 0.0332625615310415 w_k1_INTC_s016
                                 + 0.0281251414521388 w_k1_INTC_s017
                                 + 0.0193872382415968 w_k1_INTC_s100
                                 + 0.0691375681149248 w_k1_INTC_s101
                                 + 0.0486587914677782 w_k1_INTC_s102
                                 + 0.083831903960541 w_k1_INTC_s103
                                 + 0.0393649439167438 w_k1_INTC_s104
                                 + 0.0752956611102274 w_k1_INTC_s105
                                 + 0.0485982423440886 w_k1_INTC_s106
                                 + 0.0193465591285206 w_k1_INTC_s107
                                 + 0.0376003229356407 w_k1_INTC_s110
                                 + 0.0233933236155198 w_k1_INTC_s111
                                 + 0.0325182786805534 w_k1_INTC_s112
                                 + 0.0437860044528797 w_k1_INTC_s113
                                 + 0.064999478254844 w_k1_INTC_s114
                                 + 0.0206742762377023 w_k1_INTC_s115
                                 + 0.027833564817403 w_k1_INTC_s116
                                 + 0.0275263741991554 w_k1_INTC_s117
                                 + 0.00310519840591877 w_k1_KO_s000
                                 + 0.0090262875625773 w_k1_KO_s001
                                 + 0.00463975847090223 w_k1_KO_s002
                                 + 0.0034805793635312 w_k1_KO_s003
                                 + 0.00661363116891484 w_k1_KO_s004
                                 + 0.0146163486679015 w_k1_KO_s005
                                 + 0.0111605652906926 w_k1_KO_s006
                                 + 0.0119453135706138 w_k1_KO_s007
                                 + 0.0688775495547954 w_k1_KO_s010
                                 + 0.0310228382333885 w_k1_KO_s011
                                 + 0.0229333159786763 w_k1_KO_s012
                                 + 0.0148444729220931 w_k1_KO_s013
                                 + 0.0436116945013035 w_k1_KO_s014
                                 + 0.0224210994598595 w_k1_KO_s015
                                 + 0.0335268058972129 w_k1_KO_s016
                                 + 0.0262611135770905 w_k1_KO_s017
                                 + 0.0200113804858708 w_k1_KO_s100
                                 + 0.0681117608173846 w_k1_KO_s101
                                 + 0.0428849162670302 w_k1_KO_s102
                                 + 0.0923154462380874 w_k1_KO_s103
                                 + 0.0377459071003221 w_k1_KO_s104
                                 + 0.0883751855795271 w_k1_KO_s105
                                 + 0.0483739809019708 w_k1_KO_s106
                                 + 0.0216752393157991 w_k1_KO_s107
                                 + 0.042739262940981 w_k1_KO_s110
                                 + 0.0220428702476497 w_k1_KO_s111
                                 + 0.0300862216884155 w_k1_KO_s112
                                 + 0.0431424249110656 w_k1_KO_s113
                                 + 0.0632136631295397 w_k1_KO_s114
                                 + 0.0203412225070411 w_k1_KO_s115
                                 + 0.0240107204345198 w_k1_KO_s116
                                 + 0.0308300148590341 w_k1_KO_s117
                                 + 0.003533193578352 w_k1_MAT_s000
                                 + 0.00833443211790167 w_k1_MAT_s001
                                 + 0.00441034613063611 w_k1_MAT_s002
                                 + 0.00353546965156761 w_k1_MAT_s003
                                 + 0.00603611883968417 w_k1_MAT_s004
                                 + 0.0147694785520726 w_k1_MAT_s005
                                 + 0.0093410257283884 w_k1_MAT_s006
                                 + 0.0136098297433095 w_k1_MAT_s007
                                 + 0.0716870050369522 w_k1_MAT_s010
                                 + 0.0275225617607164 w_k1_MAT_s011
                                 + 0.0239516439268578 w_k1_MAT_s012
                                 + 0.0152983459258768 w_k1_MAT_s013
                                 + 0.0441907941241611 w_k1_MAT_s014
                                 + 0.02397224969414 w_k1_MAT_s015
                                 + 0.0323033313393175 w_k1_MAT_s016
                                 + 0.0318637045173167 w_k1_MAT_s017
                                 + 0.0189846145933631 w_k1_MAT_s100
                                 + 0.0617593119711757 w_k1_MAT_s101
                                 + 0.0451678155777383 w_k1_MAT_s102
                                 + 0.0796443025519848 w_k1_MAT_s103
                                 + 0.0398600399185769 w_k1_MAT_s104
                                 + 0.0863174764016301 w_k1_MAT_s105
                                 + 0.0565803228417829 w_k1_MAT_s106
                                 + 0.0258166917886918 w_k1_MAT_s107
                                 + 0.0473425873581658 w_k1_MAT_s110
                                 + 0.0276682507392968 w_k1_MAT_s111
                                 + 0.0253912681069969 w_k1_MAT_s112
                                 + 0.0497456742970129 w_k1_MAT_s113
                                 + 0.0645553173597569 w_k1_MAT_s114
                                 + 0.0183526834246471 w_k1_MAT_s115
                                 + 0.0243974616413832 w_k1_MAT_s116
                                 + 0.0257505311883468 w_k1_MAT_s117
                                 + 0.00310740753197761 w_k1_MSFT_s000
                                 + 0.00843916438022418 w_k1_MSFT_s001
                                 + 0.00490527041557631 w_k1_MSFT_s002
                                 + 0.00366755252498446 w_k1_MSFT_s003
                                 + 0.00661909222831332 w_k1_MSFT_s004
                                 + 0.0163220042626975 w_k1_MSFT_s005
                                 + 0.00841503009298678 w_k1_MSFT_s006
                                 + 0.0117350541827546 w_k1_MSFT_s007
                                 + 0.0667939734988488 w_k1_MSFT_s010
                                 + 0.0285187806133721 w_k1_MSFT_s011
                                 + 0.0232387570133349 w_k1_MSFT_s012
                                 + 0.0140972755397325 w_k1_MSFT_s013
                                 + 0.0469309426881669 w_k1_MSFT_s014
                                 + 0.0213059912401844 w_k1_MSFT_s015
                                 + 0.0330255812592971 w_k1_MSFT_s016
                                 + 0.0260100820079494 w_k1_MSFT_s017
                                 + 0.0199012883412946 w_k1_MSFT_s100
                                 + 0.070126681825566 w_k1_MSFT_s101
                                 + 0.0411428621277038 w_k1_MSFT_s102
                                 + 0.0773107681500633 w_k1_MSFT_s103
                                 + 0.039856979381756 w_k1_MSFT_s104
                                 + 0.0836151231019737 w_k1_MSFT_s105
                                 + 0.0466544432647582 w_k1_MSFT_s106
                                 + 0.0218212211871098 w_k1_MSFT_s107
                                 + 0.0365098620559719 w_k1_MSFT_s110
                                 + 0.0233778580511669 w_k1_MSFT_s111
                                 + 0.0316537066382058 w_k1_MSFT_s112
                                 + 0.0453995312262891 w_k1_MSFT_s113
                                 + 0.0635243285840969 w_k1_MSFT_s114
                                 + 0.0188496822504637 w_k1_MSFT_s115
                                 + 0.0263891049925273 w_k1_MSFT_s116
                                 + 0.0275698139538453 w_k1_MSFT_s117
                                 + 0.00309925839790955 w_k1_PG_s000
                                 + 0.0101082352799072 w_k1_PG_s001
                                 + 0.00488399829782263 w_k1_PG_s002
                                 + 0.00350709163244595 w_k1_PG_s003
                                 + 0.00654875535673061 w_k1_PG_s004
                                 + 0.0140785400859894 w_k1_PG_s005
                                 + 0.00954769012499808 w_k1_PG_s006
                                 + 0.0119773371075439 w_k1_PG_s007
                                 + 0.0705823792845592 w_k1_PG_s010
                                 + 0.0306978991107403 w_k1_PG_s011
                                 + 0.0241860901006673 w_k1_PG_s012
                                 + 0.0143845156655784 w_k1_PG_s013
                                 + 0.0447282249139176 w_k1_PG_s014
                                 + 0.0229348077118571 w_k1_PG_s015
                                 + 0.0317197899541925 w_k1_PG_s016
                                 + 0.0252162538174497 w_k1_PG_s017
                                 + 0.0188364588029375 w_k1_PG_s100
                                 + 0.066768260468781 w_k1_PG_s101
                                 + 0.040806079935576 w_k1_PG_s102
                                 + 0.0825776302887142 w_k1_PG_s103
                                 + 0.0411750677160683 w_k1_PG_s104
                                 + 0.0856768049294601 w_k1_PG_s105
                                 + 0.0473484397829338 w_k1_PG_s106
                                 + 0.0212784633351335 w_k1_PG_s107
                                 + 0.0416955250024933 w_k1_PG_s110
                                 + 0.0238585106426791 w_k1_PG_s111
                                 + 0.0312844907847629 w_k1_PG_s112
                                 + 0.046633806934234 w_k1_PG_s113
                                 + 0.0600244874903872 w_k1_PG_s114
                                 + 0.019833681838459 w_k1_PG_s115
                                 + 0.0243372343535507 w_k1_PG_s116
                                 + 0.026821126694452 w_k1_PG_s117 >= 1.041
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
