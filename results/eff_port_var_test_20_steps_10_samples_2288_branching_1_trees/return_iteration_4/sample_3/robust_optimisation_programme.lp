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
 w_k1_ADM_s0_return_constraint:  1.08532 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.924058 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.955023 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.17236 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.997872 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.8464 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.2898 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.723142 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.868353 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.943303 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.49329 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.98298 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.15512 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.956431 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.724097 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.19128 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.07665 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.86477 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.05137 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.12824 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.44558 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.18263 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.803973 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.950669 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.16603 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.1276 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.942783 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.36615 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.840984 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.33774 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.03811 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.01079 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.822899 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.08137 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.856704 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.239 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.717348 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.837132 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.15339 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.797991 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.803024 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24195 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.17381 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.732288 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.5424 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.680642 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.953366 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.972023 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.57669 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.2434 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.17806 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.13411 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.639251 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.16698 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.03426 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.803163 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.19163 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.24716 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.78402 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.1766 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.738744 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.632938 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.997558 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.07973 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.725862 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.46879 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.928044 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.45698 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.2952 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.64008 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.850559 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.17 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.1843 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.3257 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.749156 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.935031 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.21644 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.767904 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.891048 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.22878 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.20298 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.735361 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.80354 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.535559 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.708706 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.817652 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.72751 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.06028 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.09599 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.03673 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.559486 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.15817 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.868489 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.716057 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.936091 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.16742 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 2.07524 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.42818 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.631532 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.702258 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.02352 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.07863 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.03491 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.37433 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.840472 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.81817 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.08795 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.42044 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.703969 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.1858 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.917697 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.30178 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.696595 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.00494 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.11527 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.866192 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.940714 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.14685 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.08493 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.78068 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.32709 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.730077 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.774528 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.911963 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.50428 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.02001 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.19545 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.906155 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.668701 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.995051 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.0329 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.792098 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.938445 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.09726 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.50207 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.3011 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.787259 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.862853 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.14737 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.99148 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.903935 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.3157 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.811342 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.30997 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.00222 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.28209 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.909107 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.15653 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.97448 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.27733 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.762631 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.923531 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13439 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.848113 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.837114 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.34241 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.12567 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.895161 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.60385 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.703269 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.20533 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.717572 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.929979 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.09831 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.749466 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.77173 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.781645 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.47478 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.620514 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.0324 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.04917 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.931553 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.25197 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.783621 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.834974 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.534739 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.709128 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.21117 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.939454 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.998013 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.47672 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.94209 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.89484 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.49741 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.597832 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.816257 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.979498 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.08614 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.987439 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.29155 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.0202 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.93118 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.939876 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.17921 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13708 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.771698 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.43835 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.759765 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.896262 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.03069 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.62281 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.17877 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.14177 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.00921 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.660367 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.994695 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.00469 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.999069 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.03819 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.03711 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.58467 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.16756 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.736398 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.732618 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.02239 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.890486 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.800506 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.07871 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.818291 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.04276 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.10379 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.28467 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.949559 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.33858 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.01222 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.16795 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.827693 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.933547 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15783 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.835628 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.929784 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.15207 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.08891 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.876668 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.44691 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.607832 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.90316 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.742153 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.25357 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.05561 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.894918 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.13033 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.703024 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.30548 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.817917 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.940095 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.07799 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.09156 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.47691 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.15631 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.778945 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.710381 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.995278 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.11133 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.03181 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.03916 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.11081 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.81153 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.23191 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.27298 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.63569 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.86985 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.89061 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.07046 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.811427 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.03849 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.22409 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.783666 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.800419 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.39471 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.23097 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.867065 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.56405 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.558052 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.939799 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.64129 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.09934 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.09492 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.04483 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.66608 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.503825 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.47948 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.626035 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.972968 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.23563 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.915666 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.34704 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.771838 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.785654 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.579826 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.595739 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.12371 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.71938 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.01797 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.13818 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 2.07553 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.87299 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.65109 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.446922 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.778043 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.0301 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.31448 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.748567 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.988662 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09358 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.855757 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.936952 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.16754 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.15816 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.841499 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.39195 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.687681 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.807994 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.878224 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.49244 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.10114 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08679 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.957966 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.602463 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.06163 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.905898 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.841396 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.952988 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.02703 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.34184 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.12218 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.804206 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.901381 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.03128 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.958857 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.910242 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.09946 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.863221 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.27546 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.979192 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.2315 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.895025 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.19211 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.918793 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.08902 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.794331 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.907245 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08657 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.870197 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.949199 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.08027 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11548 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.876298 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.21582 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.762696 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.941391 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.899764 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.30027 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.997106 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.999861 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.04115 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.80179 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.1664 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.929208 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.945425 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.04581 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.04801 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.4202 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.1498 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.848127 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.793399 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.01778 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.977703 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.983548 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.12183 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.989265 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.40251 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.1095 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.27936 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.87622 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.06932 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.954748 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.14525 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.884387 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.966259 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.381000805560122 z_k1_s0000
                                 - 0.11800016322957 z_k1_s0001
                                 - 0.490999123552222 z_k1_s0002
                                 - 0.170000266862968 z_k1_s0003
                                 - 0.380001476706209 z_k1_s0004
                                 - 0.325000981709017 z_k1_s0005
                                 - 0.267000095348031 z_k1_s0006
                                 - 0.540000973423652 z_k1_s0007
                                 - 1.03799960333413 z_k1_s0010
                                 - 0.652998054494652 z_k1_s0011
                                 - 0.895002305223703 z_k1_s0012
                                 - 0.317000284354429 z_k1_s0013
                                 - 0.397000275883273 z_k1_s0014
                                 - 0.542002348836431 z_k1_s0015
                                 - 0.332000189259797 z_k1_s0016
                                 - 0.813001029753598 z_k1_s0017
                                 - 0.420999040173737 z_k1_s0020
                                 - 0.896998650026098 z_k1_s0021
                                 - 0.784996645747881 z_k1_s0022
                                 - 0.836999697336591 z_k1_s0023
                                 - 0.814000008822481 z_k1_s0024
                                 - 0.48799886049147 z_k1_s0025
                                 - 0.634996564530157 z_k1_s0026
                                 - 0.521998986880579 z_k1_s0027
                                 - 0.469000110843901 z_k1_s0030
                                 - 0.375000449114644 z_k1_s0031
                                 - 0.430000916425517 z_k1_s0032
                                 - 0.404000527078255 z_k1_s0033
                                 - 0.31900039439888 z_k1_s0034
                                 - 0.893000260434212 z_k1_s0035
                                 - 1.02699963038842 z_k1_s0036
                                 - 0.715001443032981 z_k1_s0037
                                 - 0.224000002689043 z_k1_s0040
                                 - 0.239000126172294 z_k1_s0041
                                 - 0.203999957611429 z_k1_s0042
                                 - 0.610001912917464 z_k1_s0043
                                 - 0.247000215943373 z_k1_s0044
                                 - 0.391001114422455 z_k1_s0045
                                 - 0.859000806177989 z_k1_s0046
                                 - 0.813001097069659 z_k1_s0047
                                 - 1.20600143203182 z_k1_s0050
                                 - 0.438000124429769 z_k1_s0051
                                 - 1.30200159548207 z_k1_s0052
                                 - 1.05900397765272 z_k1_s0053
                                 - 1.21299989067 z_k1_s0054
                                 - 1.78999497060992 z_k1_s0055
                                 - 1.09399627084366 z_k1_s0056
                                 - 1.83899412206962 z_k1_s0057
                                 - 0.485000819203315 z_k1_s0060
                                 - 0.863000922949273 z_k1_s0061
                                 - 0.894001346302724 z_k1_s0062
                                 - 0.401000628615025 z_k1_s0063
                                 - 0.708998981967899 z_k1_s0064
                                 - 0.349000534565174 z_k1_s0065
                                 - 0.717000388254977 z_k1_s0066
                                 - 0.608001331629441 z_k1_s0067
                                 - 0.407999278776205 z_k1_s0070
                                 - 0.520001397264379 z_k1_s0071
                                 - 0.190000045827251 z_k1_s0072
                                 - 0.314000089867145 z_k1_s0073
                                 - 0.325000537665152 z_k1_s0074
                                 - 0.230000112001424 z_k1_s0075
                                 - 0.498000501668366 z_k1_s0076
                                 - 0.200000062370794 z_k1_s0077
                                 - 0.208999110431198 z_k1_s0100
                                 - 0.170999319807137 z_k1_s0101
                                 - 0.524998555194548 z_k1_s0102
                                 - 0.383998628829381 z_k1_s0103
                                 - 0.438997429473371 z_k1_s0104
                                 - 0.502998510939233 z_k1_s0105
                                 - 0.169999270159375 z_k1_s0106
                                 - 0.218999082911094 z_k1_s0107
                                 - 0.691997849011239 z_k1_s0110
                                 - 0.202999661230631 z_k1_s0111
                                 - 0.704999706840714 z_k1_s0112
                                 - 0.390999006922869 z_k1_s0113
                                 - 0.301999151190238 z_k1_s0114
                                 - 0.848996686623787 z_k1_s0115
                                 - 0.406999199464167 z_k1_s0116
                                 - 0.771996651222906 z_k1_s0117
                                 - 0.533999606274025 z_k1_s0120
                                 - 0.428001903419239 z_k1_s0121
                                 - 0.248000234163051 z_k1_s0122
                                 - 0.315000112071928 z_k1_s0123
                                 - 0.344000511532712 z_k1_s0124
                                 - 0.263000288142691 z_k1_s0125
                                 - 0.295000153599132 z_k1_s0126
                                 - 0.978001612671569 z_k1_s0127
                                 - 0.883997247906793 z_k1_s0130
                                 - 0.524998786803901 z_k1_s0131
                                 - 0.731997510493191 z_k1_s0132
                                 - 0.701001749279141 z_k1_s0133
                                 - 0.347999698246593 z_k1_s0134
                                 - 0.78399674013545 z_k1_s0135
                                 - 0.624999294575243 z_k1_s0136
                                 - 0.573000064932984 z_k1_s0137
                                 - 0.159000594451627 z_k1_s0140
                                 - 0.137999921877521 z_k1_s0141
                                 - 0.130999801086042 z_k1_s0142
                                 - 0.241999168425094 z_k1_s0143
                                 - 0.30299993299401 z_k1_s0144
                                 - 0.296000277502032 z_k1_s0145
                                 - 0.189000004273442 z_k1_s0146
                                 - 0.0929998765457967 z_k1_s0147
                                 - 0.300000598179728 z_k1_s0150
                                 - 0.15099982407114 z_k1_s0151
                                 - 0.351999564372495 z_k1_s0152
                                 - 0.515000293009574 z_k1_s0153
                                 - 0.713999882549939 z_k1_s0154
                                 - 0.272999663704802 z_k1_s0155
                                 - 0.216999570391227 z_k1_s0156
                                 - 0.229999724738833 z_k1_s0157
                                 - 0.384999682665959 z_k1_s0160
                                 - 0.153999873066384 z_k1_s0161
                                 - 0.302999481085316 z_k1_s0162
                                 - 0.250998941790518 z_k1_s0163
                                 - 0.209999804999627 z_k1_s0164
                                 - 0.242000007104195 z_k1_s0165
                                 - 0.388000130894259 z_k1_s0166
                                 - 0.476998968342358 z_k1_s0167
                                 - 0.112999798911875 z_k1_s0170
                                 - 0.550000710065355 z_k1_s0171
                                 - 0.303999158786161 z_k1_s0172
                                 - 0.484998719931821 z_k1_s0173
                                 - 0.236000338951653 z_k1_s0174
                                 - 0.236000338951653 z_k1_s0175
                                 - 0.219999812826503 z_k1_s0176
                                 - 0.211999785363748 z_k1_s0177
                                 - 0.347001193061607 z_k1_s1000
                                 - 0.426001999304904 z_k1_s1001
                                 - 0.353000885278977 z_k1_s1002
                                 - 0.314000803559818 z_k1_s1003
                                 - 0.31900098877562 z_k1_s1004
                                 - 0.467000716426063 z_k1_s1005
                                 - 0.48900122849468 z_k1_s1006
                                 - 1.07100204027167 z_k1_s1007
                                 - 0.187000020042121 z_k1_s1010
                                 - 0.43299966593085 z_k1_s1011
                                 - 0.230999831134468 z_k1_s1012
                                 - 0.531000683817732 z_k1_s1013
                                 - 0.336998927184406 z_k1_s1014
                                 - 0.349000602027031 z_k1_s1015
                                 - 0.220999954631627 z_k1_s1016
                                 - 0.242999986777746 z_k1_s1017
                                 - 1.37199860220134 z_k1_s1020
                                 - 0.45100005049004 z_k1_s1021
                                 - 0.360000085931761 z_k1_s1022
                                 - 0.609001813498689 z_k1_s1023
                                 - 0.409000102824698 z_k1_s1024
                                 - 0.481001616651761 z_k1_s1025
                                 - 0.626998918053898 z_k1_s1026
                                 - 0.368000098234694 z_k1_s1027
                                 - 0.43800199591495 z_k1_s1030
                                 - 0.21500018978089 z_k1_s1031
                                 - 0.989998471655604 z_k1_s1032
                                 - 0.626999032048549 z_k1_s1033
                                 - 0.467000296993346 z_k1_s1034
                                 - 0.541998654300217 z_k1_s1035
                                 - 0.323000175743357 z_k1_s1036
                                 - 0.597001368726211 z_k1_s1037
                                 - 0.0969999321130005 z_k1_s1040
                                 - 0.183000368064668 z_k1_s1041
                                 - 0.156999956465363 z_k1_s1042
                                 - 0.174999488111134 z_k1_s1043
                                 - 0.222000285963717 z_k1_s1044
                                 - 0.127999949358387 z_k1_s1045
                                 - 0.205999925056466 z_k1_s1046
                                 - 0.231000051786602 z_k1_s1047
                                 - 0.666998566803359 z_k1_s1050
                                 - 0.498997397078693 z_k1_s1051
                                 - 0.372999578030364 z_k1_s1052
                                 - 0.414999164712645 z_k1_s1053
                                 - 0.420999441738155 z_k1_s1054
                                 - 1.43399704570597 z_k1_s1055
                                 - 0.344999226243168 z_k1_s1056
                                 - 0.550996974970899 z_k1_s1057
                                 - 0.437002261214545 z_k1_s1060
                                 - 0.576001342740393 z_k1_s1061
                                 - 0.212000693003624 z_k1_s1062
                                 - 0.461001465159649 z_k1_s1063
                                 - 0.712001375775515 z_k1_s1064
                                 - 0.483000167441053 z_k1_s1065
                                 - 0.303001025838299 z_k1_s1066
                                 - 0.224000635777141 z_k1_s1067
                                 - 0.101999942152004 z_k1_s1070
                                 - 0.154999962093704 z_k1_s1071
                                 - 0.334000510575775 z_k1_s1072
                                 - 0.254000255946922 z_k1_s1073
                                 - 0.182999796213947 z_k1_s1074
                                 - 0.283000539499772 z_k1_s1075
                                 - 0.304000627589834 z_k1_s1076
                                 - 0.0849999517933371 z_k1_s1077
                                 - 0.198000424700721 z_k1_s1100
                                 - 0.0440000451778905 z_k1_s1101
                                 - 0.0909999156450114 z_k1_s1102
                                 - 0.0909999156450114 z_k1_s1103
                                 - 0.0770000421612728 z_k1_s1104
                                 - 0.121000382539448 z_k1_s1105
                                 - 0.0490000737935825 z_k1_s1106
                                 - 0.0670000587299599 z_k1_s1107
                                 - 0.103000084581636 z_k1_s1110
                                 - 0.10899992523664 z_k1_s1111
                                 - 0.0910004032716281 z_k1_s1112
                                 - 0.0260000427060749 z_k1_s1113
                                 - 0.0520000854121497 z_k1_s1114
                                 - 0.043000042429237 z_k1_s1115
                                 - 0.0360000591314883 z_k1_s1116
                                 - 0.104000058024137 z_k1_s1117
                                 - 0.194000711857233 z_k1_s1120
                                 - 0.41400178817568 z_k1_s1121
                                 - 0.126000517422937 z_k1_s1122
                                 - 0.209000416275346 z_k1_s1123
                                 - 0.145000499220494 z_k1_s1124
                                 - 0.455001240552792 z_k1_s1125
                                 - 0.289001853849451 z_k1_s1126
                                 - 0.223001181602661 z_k1_s1127
                                 - 0.122000061561904 z_k1_s1130
                                 - 0.15200016462802 z_k1_s1131
                                 - 0.462001392178029 z_k1_s1132
                                 - 0.167000216161078 z_k1_s1133
                                 - 0.529999707899598 z_k1_s1134
                                 - 0.331999807823892 z_k1_s1135
                                 - 0.216000156955542 z_k1_s1136
                                 - 0.4570010499334 z_k1_s1137
                                 - 0.112000083239941 z_k1_s1140
                                 - 0.0820003712117685 z_k1_s1141
                                 - 0.0630000902600205 z_k1_s1142
                                 - 0.0480000257456768 z_k1_s1143
                                 - 0.095000362257453 z_k1_s1144
                                 - 0.0500000297142503 z_k1_s1145
                                 - 0.175000521000391 z_k1_s1146
                                 - 0.0699997913996416 z_k1_s1147
                                 - 0.0989996481590723 z_k1_s1150
                                 - 0.0960005540642666 z_k1_s1151
                                 - 0.0780000808893355 z_k1_s1152
                                 - 0.0979999501274704 z_k1_s1153
                                 - 0.328000066467693 z_k1_s1154
                                 - 0.0850000637127122 z_k1_s1155
                                 - 0.0930001257680842 z_k1_s1156
                                 - 0.0760001130255417 z_k1_s1157
                                 - 0.165999735765358 z_k1_s1160
                                 - 0.197000638861417 z_k1_s1161
                                 - 0.238000545857642 z_k1_s1162
                                 - 0.142000130902188 z_k1_s1163
                                 - 0.0830001748831057 z_k1_s1164
                                 - 0.176000274667657 z_k1_s1165
                                 - 0.30100010343679 z_k1_s1166
                                 - 0.232000222516263 z_k1_s1167
                                 - 0.167000545486317 z_k1_s1170
                                 - 0.229001222255809 z_k1_s1171
                                 - 0.19200035708486 z_k1_s1172
                                 - 0.207001124047913 z_k1_s1173
                                 - 0.0610002223034882 z_k1_s1174
                                 - 0.0830003205113841 z_k1_s1175
                                 - 0.114000108893675 z_k1_s1176
                                 - 0.0470002398079115 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.21117 w_k1_ADM_s000
                                 + 1.41896 w_k1_AXP_s000
                                 + 1.37481 w_k1_DIS_s000
                                 + 1.26741 w_k1_FDX_s000
                                 + 1.28955 w_k1_IBM_s000
                                 + 1.02819 w_k1_INTC_s000
                                 + 1.12378 w_k1_KO_s000 + 1.34587 w_k1_MAT_s000
                                 + 1.08397 w_k1_MSFT_s000
                                 + 1.22693 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.48625 w_k1_ADM_s000
                                 + 1.32044 w_k1_AXP_s000
                                 + 1.86943 w_k1_DIS_s000
                                 + 1.49557 w_k1_FDX_s000
                                 + 1.08286 w_k1_IBM_s000
                                 + 1.32684 w_k1_INTC_s000
                                 + 1.35933 w_k1_KO_s000
                                 + 0.966641 w_k1_MAT_s000
                                 + 1.47558 w_k1_MSFT_s000 + 1.3153 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.850822 w_k1_ADM_s000
                                 + 1.08676 w_k1_AXP_s000
                                 + 0.933794 w_k1_DIS_s000
                                 + 0.913269 w_k1_FDX_s000
                                 + 1.30972 w_k1_IBM_s000
                                 + 1.00702 w_k1_INTC_s000
                                 + 0.997267 w_k1_KO_s000
                                 + 1.36552 w_k1_MAT_s000
                                 + 0.875208 w_k1_MSFT_s000
                                 + 0.896423 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.980865 w_k1_ADM_s000
                                 + 0.707407 w_k1_AXP_s000
                                 + 0.66759 w_k1_DIS_s000
                                 + 0.863364 w_k1_FDX_s000
                                 + 0.78491 w_k1_IBM_s000
                                 + 0.884039 w_k1_INTC_s000
                                 + 0.890906 w_k1_KO_s000
                                 + 0.741567 w_k1_MAT_s000
                                 + 0.869513 w_k1_MSFT_s000
                                 + 0.887051 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.17663 w_k1_ADM_s000
                                 + 0.99001 w_k1_AXP_s000
                                 + 1.20802 w_k1_DIS_s000 + 1.1457 w_k1_FDX_s000
                                 + 0.868927 w_k1_IBM_s000
                                 + 1.00581 w_k1_INTC_s000
                                 + 1.00615 w_k1_KO_s000
                                 + 0.745139 w_k1_MAT_s000
                                 + 1.20708 w_k1_MSFT_s000
                                 + 1.04409 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.784771 w_k1_ADM_s000
                                 + 0.796974 w_k1_AXP_s000
                                 + 0.77237 w_k1_DIS_s000
                                 + 0.879632 w_k1_FDX_s000
                                 + 0.764051 w_k1_IBM_s000
                                 + 1.0171 w_k1_INTC_s000
                                 + 0.787251 w_k1_KO_s000
                                 + 0.646529 w_k1_MAT_s000
                                 + 0.880399 w_k1_MSFT_s000
                                 + 0.912987 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.741814 w_k1_ADM_s000
                                 + 0.790907 w_k1_AXP_s000
                                 + 0.500361 w_k1_DIS_s000
                                 + 0.676486 w_k1_FDX_s000
                                 + 0.784135 w_k1_IBM_s000
                                 + 0.883432 w_k1_INTC_s000
                                 + 0.739296 w_k1_KO_s000
                                 + 0.848859 w_k1_MAT_s000
                                 + 0.759499 w_k1_MSFT_s000
                                 + 0.900024 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.03857 w_k1_ADM_s000
                                 + 1.11965 w_k1_AXP_s000
                                 + 0.784641 w_k1_DIS_s000
                                 + 1.05542 w_k1_FDX_s000
                                 + 0.831689 w_k1_IBM_s000
                                 + 0.839993 w_k1_INTC_s000
                                 + 0.840546 w_k1_KO_s000
                                 + 0.889332 w_k1_MAT_s000
                                 + 0.854535 w_k1_MSFT_s000
                                 + 0.96199 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.16564 w_k1_ADM_s001
                                 + 1.11525 w_k1_AXP_s001
                                 + 1.17995 w_k1_DIS_s001
                                 + 1.01716 w_k1_FDX_s001
                                 + 1.35272 w_k1_IBM_s001
                                 + 1.11362 w_k1_INTC_s001 + 1.2483 w_k1_KO_s001
                                 + 1.66614 w_k1_MAT_s001
                                 + 1.27529 w_k1_MSFT_s001
                                 + 1.14581 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.912784 w_k1_ADM_s001
                                 + 0.711337 w_k1_AXP_s001
                                 + 0.654187 w_k1_DIS_s001
                                 + 0.816629 w_k1_FDX_s001
                                 + 0.634792 w_k1_IBM_s001
                                 + 1.06536 w_k1_INTC_s001
                                 + 0.751966 w_k1_KO_s001
                                 + 0.501961 w_k1_MAT_s001
                                 + 1.02017 w_k1_MSFT_s001
                                 + 0.793532 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.01435 w_k1_ADM_s001
                                 + 0.975577 w_k1_AXP_s001
                                 + 0.892932 w_k1_DIS_s001
                                 + 0.886402 w_k1_FDX_s001
                                 + 1.05353 w_k1_IBM_s001
                                 + 0.968085 w_k1_INTC_s001
                                 + 1.0197 w_k1_KO_s001 + 1.09512 w_k1_MAT_s001
                                 + 0.991504 w_k1_MSFT_s001
                                 + 1.0209 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.851884 w_k1_ADM_s001
                                 + 0.91132 w_k1_AXP_s001
                                 + 0.618345 w_k1_DIS_s001
                                 + 0.715457 w_k1_FDX_s001
                                 + 0.93147 w_k1_IBM_s001
                                 + 0.960387 w_k1_INTC_s001
                                 + 0.762225 w_k1_KO_s001
                                 + 0.686737 w_k1_MAT_s001
                                 + 0.791909 w_k1_MSFT_s001
                                 + 0.868105 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.43266 w_k1_ADM_s001
                                 + 1.2483 w_k1_AXP_s001 + 1.48658 w_k1_DIS_s001
                                 + 1.24321 w_k1_FDX_s001
                                 + 0.878909 w_k1_IBM_s001
                                 + 1.22826 w_k1_INTC_s001
                                 + 1.26336 w_k1_KO_s001 + 1.03636 w_k1_MAT_s001
                                 + 1.38305 w_k1_MSFT_s001
                                 + 1.28211 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.48606 w_k1_ADM_s001
                                 + 1.85806 w_k1_AXP_s001
                                 + 1.79206 w_k1_DIS_s001
                                 + 1.29547 w_k1_FDX_s001
                                 + 1.35168 w_k1_IBM_s001
                                 + 1.41331 w_k1_INTC_s001 + 1.4266 w_k1_KO_s001
                                 + 1.79042 w_k1_MAT_s001
                                 + 1.46054 w_k1_MSFT_s001
                                 + 1.17651 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.973016 w_k1_ADM_s001
                                 + 0.863677 w_k1_AXP_s001
                                 + 0.668652 w_k1_DIS_s001
                                 + 0.686505 w_k1_FDX_s001
                                 + 1.31754 w_k1_IBM_s001
                                 + 0.848591 w_k1_INTC_s001
                                 + 0.927558 w_k1_KO_s001 + 1.4713 w_k1_MAT_s001
                                 + 0.768106 w_k1_MSFT_s001
                                 + 0.864953 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.08802 w_k1_ADM_s001
                                 + 0.993467 w_k1_AXP_s001
                                 + 1.01144 w_k1_DIS_s001
                                 + 1.09819 w_k1_FDX_s001
                                 + 0.670213 w_k1_IBM_s001
                                 + 1.12186 w_k1_INTC_s001
                                 + 0.898024 w_k1_KO_s001
                                 + 0.809453 w_k1_MAT_s001
                                 + 1.16294 w_k1_MSFT_s001
                                 + 0.931518 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16476 w_k1_ADM_s002
                                 + 1.38888 w_k1_AXP_s002
                                 + 1.77303 w_k1_DIS_s002
                                 + 1.30222 w_k1_FDX_s002
                                 + 1.72123 w_k1_IBM_s002
                                 + 1.25922 w_k1_INTC_s002
                                 + 1.29604 w_k1_KO_s002 + 2.07867 w_k1_MAT_s002
                                 + 1.17104 w_k1_MSFT_s002
                                 + 1.09375 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.77102 w_k1_ADM_s002
                                 + 0.813108 w_k1_AXP_s002
                                 + 0.709715 w_k1_DIS_s002
                                 + 0.767393 w_k1_FDX_s002
                                 + 1.06568 w_k1_IBM_s002
                                 + 0.859444 w_k1_INTC_s002
                                 + 0.79374 w_k1_KO_s002 + 1.0855 w_k1_MAT_s002
                                 + 0.877999 w_k1_MSFT_s002
                                 + 0.810748 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.857859 w_k1_ADM_s002
                                 + 0.911064 w_k1_AXP_s002
                                 + 1.03197 w_k1_DIS_s002
                                 + 0.940172 w_k1_FDX_s002
                                 + 1.02021 w_k1_IBM_s002
                                 + 1.00119 w_k1_INTC_s002
                                 + 0.993506 w_k1_KO_s002
                                 + 0.937505 w_k1_MAT_s002
                                 + 1.03684 w_k1_MSFT_s002
                                 + 0.876057 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.843646 w_k1_ADM_s002
                                 + 0.686811 w_k1_AXP_s002
                                 + 0.766997 w_k1_DIS_s002
                                 + 0.812333 w_k1_FDX_s002
                                 + 0.663913 w_k1_IBM_s002
                                 + 0.836463 w_k1_INTC_s002
                                 + 0.710857 w_k1_KO_s002
                                 + 0.529422 w_k1_MAT_s002
                                 + 0.842353 w_k1_MSFT_s002
                                 + 0.782373 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.950759 w_k1_ADM_s002
                                 + 1.04162 w_k1_AXP_s002
                                 + 1.23792 w_k1_DIS_s002
                                 + 0.999291 w_k1_FDX_s002
                                 + 1.37781 w_k1_IBM_s002
                                 + 0.892526 w_k1_INTC_s002
                                 + 1.30909 w_k1_KO_s002 + 1.69303 w_k1_MAT_s002
                                 + 1.08217 w_k1_MSFT_s002
                                 + 1.04645 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.21198 w_k1_ADM_s002
                                 + 1.02849 w_k1_AXP_s002
                                 + 0.999822 w_k1_DIS_s002
                                 + 1.08858 w_k1_FDX_s002
                                 + 0.971701 w_k1_IBM_s002
                                 + 1.08315 w_k1_INTC_s002
                                 + 0.970519 w_k1_KO_s002
                                 + 1.16745 w_k1_MAT_s002
                                 + 1.10851 w_k1_MSFT_s002
                                 + 1.01724 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.15343 w_k1_ADM_s002
                                 + 1.3552 w_k1_AXP_s002 + 1.49532 w_k1_DIS_s002
                                 + 1.3398 w_k1_FDX_s002 + 1.02825 w_k1_IBM_s002
                                 + 1.22109 w_k1_INTC_s002
                                 + 0.995062 w_k1_KO_s002
                                 + 1.34337 w_k1_MAT_s002
                                 + 1.12122 w_k1_MSFT_s002
                                 + 1.14673 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.0753 w_k1_ADM_s002
                                 + 1.12473 w_k1_AXP_s002
                                 + 1.07874 w_k1_DIS_s002
                                 + 1.16966 w_k1_FDX_s002
                                 + 0.763067 w_k1_IBM_s002
                                 + 1.11372 w_k1_INTC_s002
                                 + 0.848447 w_k1_KO_s002
                                 + 1.05217 w_k1_MAT_s002
                                 + 0.991522 w_k1_MSFT_s002
                                 + 0.939263 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.917123 w_k1_ADM_s003
                                 + 0.740028 w_k1_AXP_s003
                                 + 0.612748 w_k1_DIS_s003
                                 + 0.819502 w_k1_FDX_s003
                                 + 0.762703 w_k1_IBM_s003
                                 + 0.828989 w_k1_INTC_s003
                                 + 0.698268 w_k1_KO_s003
                                 + 0.747989 w_k1_MAT_s003
                                 + 0.82117 w_k1_MSFT_s003
                                 + 0.862432 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.1706 w_k1_ADM_s003
                                 + 0.868234 w_k1_AXP_s003
                                 + 1.19687 w_k1_DIS_s003
                                 + 1.18343 w_k1_FDX_s003
                                 + 0.725272 w_k1_IBM_s003
                                 + 1.07287 w_k1_INTC_s003
                                 + 0.883684 w_k1_KO_s003
                                 + 0.514337 w_k1_MAT_s003
                                 + 1.19216 w_k1_MSFT_s003
                                 + 0.953804 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.821296 w_k1_ADM_s003
                                 + 0.743545 w_k1_AXP_s003
                                 + 0.779957 w_k1_DIS_s003
                                 + 0.800052 w_k1_FDX_s003
                                 + 0.985374 w_k1_IBM_s003
                                 + 0.807823 w_k1_INTC_s003
                                 + 0.965665 w_k1_KO_s003
                                 + 1.17148 w_k1_MAT_s003
                                 + 0.884386 w_k1_MSFT_s003
                                 + 0.819976 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.920426 w_k1_ADM_s003
                                 + 1.07123 w_k1_AXP_s003
                                 + 0.81049 w_k1_DIS_s003
                                 + 1.01399 w_k1_FDX_s003
                                 + 1.01954 w_k1_IBM_s003
                                 + 0.876471 w_k1_INTC_s003
                                 + 0.951624 w_k1_KO_s003
                                 + 1.47986 w_k1_MAT_s003
                                 + 0.988761 w_k1_MSFT_s003
                                 + 0.931641 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.955312 w_k1_ADM_s003
                                 + 0.985479 w_k1_AXP_s003
                                 + 0.881913 w_k1_DIS_s003
                                 + 1.05648 w_k1_FDX_s003
                                 + 0.731229 w_k1_IBM_s003
                                 + 0.868574 w_k1_INTC_s003
                                 + 0.869986 w_k1_KO_s003
                                 + 0.950742 w_k1_MAT_s003
                                 + 1.06604 w_k1_MSFT_s003
                                 + 0.894623 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.877696 w_k1_ADM_s003
                                 + 1.1202 w_k1_AXP_s003 + 1.01192 w_k1_DIS_s003
                                 + 0.991852 w_k1_FDX_s003
                                 + 1.37621 w_k1_IBM_s003
                                 + 0.96262 w_k1_INTC_s003
                                 + 1.09141 w_k1_KO_s003 + 1.64763 w_k1_MAT_s003
                                 + 1.11886 w_k1_MSFT_s003
                                 + 1.02165 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.20744 w_k1_ADM_s003
                                 + 1.27834 w_k1_AXP_s003
                                 + 1.55587 w_k1_DIS_s003
                                 + 1.29382 w_k1_FDX_s003
                                 + 1.25228 w_k1_IBM_s003
                                 + 1.11507 w_k1_INTC_s003
                                 + 1.36262 w_k1_KO_s003 + 1.56501 w_k1_MAT_s003
                                 + 1.26838 w_k1_MSFT_s003
                                 + 1.10719 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.1884 w_k1_ADM_s003
                                 + 0.864617 w_k1_AXP_s003
                                 + 0.996734 w_k1_DIS_s003
                                 + 1.04166 w_k1_FDX_s003
                                 + 0.906542 w_k1_IBM_s003
                                 + 0.824111 w_k1_INTC_s003
                                 + 1.03501 w_k1_KO_s003 + 1.38602 w_k1_MAT_s003
                                 + 1.07213 w_k1_MSFT_s003
                                 + 1.02787 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.32753 w_k1_ADM_s004
                                 + 1.28312 w_k1_AXP_s004 + 1.6069 w_k1_DIS_s004
                                 + 1.30792 w_k1_FDX_s004
                                 + 1.24772 w_k1_IBM_s004
                                 + 0.982573 w_k1_INTC_s004
                                 + 1.21358 w_k1_KO_s004 + 1.08338 w_k1_MAT_s004
                                 + 1.17977 w_k1_MSFT_s004
                                 + 1.00567 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.876586 w_k1_ADM_s004
                                 + 1.18679 w_k1_AXP_s004
                                 + 1.72363 w_k1_DIS_s004
                                 + 1.22075 w_k1_FDX_s004
                                 + 1.51962 w_k1_IBM_s004
                                 + 1.07549 w_k1_INTC_s004 + 1.3048 w_k1_KO_s004
                                 + 1.32312 w_k1_MAT_s004
                                 + 1.35772 w_k1_MSFT_s004
                                 + 0.983524 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.48575 w_k1_ADM_s004
                                 + 1.45587 w_k1_AXP_s004 + 1.6636 w_k1_DIS_s004
                                 + 1.52727 w_k1_FDX_s004
                                 + 1.23725 w_k1_IBM_s004
                                 + 0.957255 w_k1_INTC_s004
                                 + 1.32571 w_k1_KO_s004 + 1.49996 w_k1_MAT_s004
                                 + 1.22854 w_k1_MSFT_s004
                                 + 1.11776 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.949711 w_k1_ADM_s004
                                 + 1.10201 w_k1_AXP_s004
                                 + 1.06739 w_k1_DIS_s004
                                 + 0.996788 w_k1_FDX_s004
                                 + 1.48661 w_k1_IBM_s004
                                 + 0.839777 w_k1_INTC_s004
                                 + 1.13695 w_k1_KO_s004 + 1.41151 w_k1_MAT_s004
                                 + 0.871234 w_k1_MSFT_s004
                                 + 1.00696 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.02591 w_k1_ADM_s004
                                 + 0.945293 w_k1_AXP_s004
                                 + 1.07757 w_k1_DIS_s004
                                 + 1.19662 w_k1_FDX_s004
                                 + 0.986349 w_k1_IBM_s004
                                 + 1.05876 w_k1_INTC_s004
                                 + 0.973052 w_k1_KO_s004
                                 + 0.671147 w_k1_MAT_s004
                                 + 1.00091 w_k1_MSFT_s004
                                 + 0.857821 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.01272 w_k1_ADM_s004
                                 + 1.2392 w_k1_AXP_s004
                                 + 0.855838 w_k1_DIS_s004
                                 + 1.09856 w_k1_FDX_s004
                                 + 0.949994 w_k1_IBM_s004
                                 + 0.808842 w_k1_INTC_s004
                                 + 0.790363 w_k1_KO_s004
                                 + 1.52848 w_k1_MAT_s004
                                 + 0.998265 w_k1_MSFT_s004
                                 + 0.97555 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.79756 w_k1_ADM_s004
                                 + 0.786596 w_k1_AXP_s004
                                 + 1.03254 w_k1_DIS_s004
                                 + 0.953387 w_k1_FDX_s004
                                 + 1.02022 w_k1_IBM_s004
                                 + 0.820159 w_k1_INTC_s004
                                 + 0.906588 w_k1_KO_s004
                                 + 0.811931 w_k1_MAT_s004
                                 + 1.01841 w_k1_MSFT_s004
                                 + 0.853184 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.653839 w_k1_ADM_s004
                                 + 0.69707 w_k1_AXP_s004
                                 + 0.629403 w_k1_DIS_s004
                                 + 0.758006 w_k1_FDX_s004
                                 + 0.958758 w_k1_IBM_s004
                                 + 0.648478 w_k1_INTC_s004
                                 + 0.72386 w_k1_KO_s004
                                 + 0.840265 w_k1_MAT_s004
                                 + 0.71497 w_k1_MSFT_s004
                                 + 0.746946 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.02994 w_k1_ADM_s005
                                 + 0.879894 w_k1_AXP_s005
                                 + 0.723394 w_k1_DIS_s005
                                 + 0.835456 w_k1_FDX_s005
                                 + 1.03451 w_k1_IBM_s005
                                 + 0.668372 w_k1_INTC_s005
                                 + 0.794102 w_k1_KO_s005 + 1.0548 w_k1_MAT_s005
                                 + 0.722177 w_k1_MSFT_s005
                                 + 0.930959 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.841739 w_k1_ADM_s005
                                 + 0.896419 w_k1_AXP_s005
                                 + 0.747693 w_k1_DIS_s005
                                 + 0.886768 w_k1_FDX_s005
                                 + 0.768331 w_k1_IBM_s005
                                 + 0.792395 w_k1_INTC_s005
                                 + 0.71822 w_k1_KO_s005
                                 + 0.869974 w_k1_MAT_s005
                                 + 0.782436 w_k1_MSFT_s005
                                 + 0.867788 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.764859 w_k1_ADM_s005
                                 + 0.612521 w_k1_AXP_s005
                                 + 0.654385 w_k1_DIS_s005
                                 + 0.784437 w_k1_FDX_s005
                                 + 0.723468 w_k1_IBM_s005
                                 + 0.821099 w_k1_INTC_s005
                                 + 0.719562 w_k1_KO_s005
                                 + 0.621075 w_k1_MAT_s005
                                 + 0.798447 w_k1_MSFT_s005
                                 + 0.815583 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.23991 w_k1_ADM_s005
                                 + 1.42031 w_k1_AXP_s005
                                 + 1.43367 w_k1_DIS_s005
                                 + 1.16454 w_k1_FDX_s005
                                 + 1.56967 w_k1_IBM_s005
                                 + 1.11447 w_k1_INTC_s005
                                 + 1.19925 w_k1_KO_s005 + 1.43257 w_k1_MAT_s005
                                 + 1.19736 w_k1_MSFT_s005
                                 + 1.20014 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.01475 w_k1_ADM_s005
                                 + 0.76882 w_k1_AXP_s005
                                 + 0.998266 w_k1_DIS_s005
                                 + 0.925016 w_k1_FDX_s005
                                 + 0.819781 w_k1_IBM_s005
                                 + 0.879186 w_k1_INTC_s005
                                 + 0.91359 w_k1_KO_s005
                                 + 0.758072 w_k1_MAT_s005
                                 + 0.88535 w_k1_MSFT_s005
                                 + 1.04223 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.95944 w_k1_ADM_s005
                                 + 1.00655 w_k1_AXP_s005
                                 + 1.40285 w_k1_DIS_s005
                                 + 1.19199 w_k1_FDX_s005
                                 + 1.06887 w_k1_IBM_s005
                                 + 1.16451 w_k1_INTC_s005 + 1.1961 w_k1_KO_s005
                                 + 1.11706 w_k1_MAT_s005
                                 + 1.18127 w_k1_MSFT_s005
                                 + 1.05467 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.956351 w_k1_ADM_s005
                                 + 1.04125 w_k1_AXP_s005
                                 + 0.901696 w_k1_DIS_s005
                                 + 1.00008 w_k1_FDX_s005
                                 + 0.967717 w_k1_IBM_s005
                                 + 0.935326 w_k1_INTC_s005
                                 + 0.820407 w_k1_KO_s005
                                 + 1.03052 w_k1_MAT_s005
                                 + 0.912824 w_k1_MSFT_s005
                                 + 1.05329 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.24389 w_k1_ADM_s005
                                 + 1.14171 w_k1_AXP_s005
                                 + 1.02958 w_k1_DIS_s005 + 1.057 w_k1_FDX_s005
                                 + 1.13995 w_k1_IBM_s005
                                 + 0.783491 w_k1_INTC_s005
                                 + 1.02934 w_k1_KO_s005 + 1.10115 w_k1_MAT_s005
                                 + 1.01535 w_k1_MSFT_s005
                                 + 1.06313 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.955758 w_k1_ADM_s006
                                 + 1.05488 w_k1_AXP_s006
                                 + 0.906141 w_k1_DIS_s006
                                 + 0.994618 w_k1_FDX_s006
                                 + 0.798895 w_k1_IBM_s006
                                 + 1.03942 w_k1_INTC_s006
                                 + 0.842184 w_k1_KO_s006
                                 + 0.688251 w_k1_MAT_s006
                                 + 1.01972 w_k1_MSFT_s006
                                 + 0.994011 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.09029 w_k1_ADM_s006
                                 + 1.39038 w_k1_AXP_s006
                                 + 1.11346 w_k1_DIS_s006
                                 + 1.15338 w_k1_FDX_s006
                                 + 1.12835 w_k1_IBM_s006
                                 + 1.00574 w_k1_INTC_s006 + 1.0201 w_k1_KO_s006
                                 + 0.915658 w_k1_MAT_s006
                                 + 1.06994 w_k1_MSFT_s006
                                 + 1.13255 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.09212 w_k1_ADM_s006
                                 + 1.00543 w_k1_AXP_s006
                                 + 0.977043 w_k1_DIS_s006
                                 + 0.973173 w_k1_FDX_s006
                                 + 1.08764 w_k1_IBM_s006
                                 + 0.910072 w_k1_INTC_s006
                                 + 1.05457 w_k1_KO_s006
                                 + 0.927324 w_k1_MAT_s006
                                 + 0.943472 w_k1_MSFT_s006
                                 + 1.0894 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.826037 w_k1_ADM_s006
                                 + 0.763224 w_k1_AXP_s006
                                 + 0.652264 w_k1_DIS_s006
                                 + 0.716062 w_k1_FDX_s006
                                 + 0.768499 w_k1_IBM_s006
                                 + 0.859541 w_k1_INTC_s006
                                 + 0.763643 w_k1_KO_s006
                                 + 0.418813 w_k1_MAT_s006
                                 + 0.758084 w_k1_MSFT_s006
                                 + 0.757648 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.27534 w_k1_ADM_s006
                                 + 1.27206 w_k1_AXP_s006
                                 + 1.56442 w_k1_DIS_s006
                                 + 1.36397 w_k1_FDX_s006
                                 + 1.28008 w_k1_IBM_s006
                                 + 1.09521 w_k1_INTC_s006
                                 + 1.17782 w_k1_KO_s006
                                 + 0.845745 w_k1_MAT_s006
                                 + 1.29558 w_k1_MSFT_s006
                                 + 1.16482 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.888814 w_k1_ADM_s006
                                 + 0.869088 w_k1_AXP_s006
                                 + 0.732608 w_k1_DIS_s006
                                 + 0.965198 w_k1_FDX_s006
                                 + 0.600256 w_k1_IBM_s006
                                 + 0.794591 w_k1_INTC_s006
                                 + 0.724298 w_k1_KO_s006
                                 + 0.620672 w_k1_MAT_s006
                                 + 1.00632 w_k1_MSFT_s006
                                 + 0.937502 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.819161 w_k1_ADM_s006
                                 + 0.843763 w_k1_AXP_s006
                                 + 0.727967 w_k1_DIS_s006
                                 + 0.808403 w_k1_FDX_s006
                                 + 1.00729 w_k1_IBM_s006
                                 + 1.00956 w_k1_INTC_s006
                                 + 0.86196 w_k1_KO_s006 + 1.12084 w_k1_MAT_s006
                                 + 0.91195 w_k1_MSFT_s006
                                 + 0.875457 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.13141 w_k1_ADM_s006
                                 + 1.03353 w_k1_AXP_s006 + 1.3534 w_k1_DIS_s006
                                 + 1.03843 w_k1_FDX_s006
                                 + 1.16958 w_k1_IBM_s006
                                 + 1.04837 w_k1_INTC_s006
                                 + 1.09569 w_k1_KO_s006
                                 + 0.701701 w_k1_MAT_s006
                                 + 1.17751 w_k1_MSFT_s006
                                 + 1.01213 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.924111 w_k1_ADM_s007
                                 + 0.925727 w_k1_AXP_s007
                                 + 0.772895 w_k1_DIS_s007
                                 + 0.896829 w_k1_FDX_s007
                                 + 1.26044 w_k1_IBM_s007
                                 + 1.11248 w_k1_INTC_s007
                                 + 0.945463 w_k1_KO_s007 + 0.9108 w_k1_MAT_s007
                                 + 1.00706 w_k1_MSFT_s007
                                 + 0.928631 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.976794 w_k1_ADM_s007
                                 + 0.940384 w_k1_AXP_s007
                                 + 1.09492 w_k1_DIS_s007 + 1.0716 w_k1_FDX_s007
                                 + 0.894453 w_k1_IBM_s007
                                 + 1.28188 w_k1_INTC_s007
                                 + 1.03291 w_k1_KO_s007
                                 + 0.576035 w_k1_MAT_s007
                                 + 1.1261 w_k1_MSFT_s007 + 1.10113 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.30947 w_k1_ADM_s007
                                 + 1.2622 w_k1_AXP_s007 + 1.51002 w_k1_DIS_s007
                                 + 1.30966 w_k1_FDX_s007
                                 + 0.941565 w_k1_IBM_s007
                                 + 1.12497 w_k1_INTC_s007
                                 + 1.08889 w_k1_KO_s007
                                 + 0.826182 w_k1_MAT_s007
                                 + 1.15872 w_k1_MSFT_s007
                                 + 1.16737 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.853034 w_k1_ADM_s007
                                 + 1.04549 w_k1_AXP_s007
                                 + 1.13768 w_k1_DIS_s007
                                 + 0.909692 w_k1_FDX_s007
                                 + 1.09867 w_k1_IBM_s007
                                 + 1.09497 w_k1_INTC_s007
                                 + 1.21213 w_k1_KO_s007 + 1.0004 w_k1_MAT_s007
                                 + 1.07023 w_k1_MSFT_s007
                                 + 1.04479 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.11304 w_k1_ADM_s007
                                 + 1.04937 w_k1_AXP_s007
                                 + 1.04682 w_k1_DIS_s007
                                 + 1.16475 w_k1_FDX_s007
                                 + 0.752123 w_k1_IBM_s007
                                 + 1.20969 w_k1_INTC_s007
                                 + 0.901696 w_k1_KO_s007
                                 + 0.670468 w_k1_MAT_s007
                                 + 1.11742 w_k1_MSFT_s007
                                 + 0.946593 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.915632 w_k1_ADM_s007
                                 + 1.23056 w_k1_AXP_s007
                                 + 1.60773 w_k1_DIS_s007
                                 + 1.24075 w_k1_FDX_s007
                                 + 1.49685 w_k1_IBM_s007
                                 + 1.12814 w_k1_INTC_s007
                                 + 1.50861 w_k1_KO_s007 + 1.59499 w_k1_MAT_s007
                                 + 1.33064 w_k1_MSFT_s007
                                 + 1.08351 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.73048 w_k1_ADM_s007
                                 + 0.630436 w_k1_AXP_s007
                                 + 0.616805 w_k1_DIS_s007
                                 + 0.757163 w_k1_FDX_s007
                                 + 0.760612 w_k1_IBM_s007
                                 + 0.984267 w_k1_INTC_s007
                                 + 0.799261 w_k1_KO_s007
                                 + 0.727025 w_k1_MAT_s007
                                 + 0.949518 w_k1_MSFT_s007
                                 + 0.815069 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.39692 w_k1_ADM_s007
                                 + 1.67604 w_k1_AXP_s007
                                 + 1.56431 w_k1_DIS_s007
                                 + 1.40696 w_k1_FDX_s007
                                 + 1.12941 w_k1_IBM_s007
                                 + 1.58341 w_k1_INTC_s007
                                 + 1.27201 w_k1_KO_s007 + 1.18708 w_k1_MAT_s007
                                 + 1.35886 w_k1_MSFT_s007
                                 + 1.23511 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.816901 w_k1_ADM_s010
                                 + 1.18517 w_k1_AXP_s010
                                 + 1.15095 w_k1_DIS_s010
                                 + 0.968165 w_k1_FDX_s010
                                 + 1.76921 w_k1_IBM_s010
                                 + 1.20796 w_k1_INTC_s010
                                 + 1.41376 w_k1_KO_s010 + 2.17467 w_k1_MAT_s010
                                 + 1.23261 w_k1_MSFT_s010 + 1.1288 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.690011 w_k1_ADM_s010
                                 + 0.635674 w_k1_AXP_s010
                                 + 0.73501 w_k1_DIS_s010
                                 + 0.748346 w_k1_FDX_s010
                                 + 0.902638 w_k1_IBM_s010
                                 + 1.05939 w_k1_INTC_s010
                                 + 0.973911 w_k1_KO_s010 + 1.0879 w_k1_MAT_s010
                                 + 0.966092 w_k1_MSFT_s010
                                 + 0.927758 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.903002 w_k1_ADM_s010
                                 + 0.944181 w_k1_AXP_s010
                                 + 0.90138 w_k1_DIS_s010
                                 + 1.02157 w_k1_FDX_s010
                                 + 0.672141 w_k1_IBM_s010
                                 + 1.0805 w_k1_INTC_s010
                                 + 0.862727 w_k1_KO_s010
                                 + 1.09122 w_k1_MAT_s010
                                 + 1.06562 w_k1_MSFT_s010
                                 + 1.01163 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.946851 w_k1_ADM_s010
                                 + 1.0825 w_k1_AXP_s010
                                 + 0.991221 w_k1_DIS_s010
                                 + 0.990153 w_k1_FDX_s010
                                 + 1.13115 w_k1_IBM_s010
                                 + 1.18927 w_k1_INTC_s010
                                 + 1.21958 w_k1_KO_s010 + 1.37433 w_k1_MAT_s010
                                 + 1.20597 w_k1_MSFT_s010
                                 + 0.999372 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.985274 w_k1_ADM_s010
                                 + 0.900537 w_k1_AXP_s010
                                 + 0.879068 w_k1_DIS_s010
                                 + 0.964223 w_k1_FDX_s010
                                 + 0.945752 w_k1_IBM_s010
                                 + 1.0972 w_k1_INTC_s010
                                 + 0.994315 w_k1_KO_s010
                                 + 1.60442 w_k1_MAT_s010
                                 + 1.15487 w_k1_MSFT_s010
                                 + 1.00695 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.02519 w_k1_ADM_s010
                                 + 1.17795 w_k1_AXP_s010
                                 + 1.21779 w_k1_DIS_s010
                                 + 1.10558 w_k1_FDX_s010
                                 + 1.33398 w_k1_IBM_s010
                                 + 1.29922 w_k1_INTC_s010
                                 + 1.08814 w_k1_KO_s010 + 1.63276 w_k1_MAT_s010
                                 + 1.23975 w_k1_MSFT_s010 + 1.035 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.597942 w_k1_ADM_s010
                                 + 0.701084 w_k1_AXP_s010
                                 + 0.488819 w_k1_DIS_s010
                                 + 0.672573 w_k1_FDX_s010
                                 + 0.638739 w_k1_IBM_s010
                                 + 1.05339 w_k1_INTC_s010
                                 + 0.711696 w_k1_KO_s010
                                 + 0.962501 w_k1_MAT_s010
                                 + 0.806789 w_k1_MSFT_s010
                                 + 0.812421 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.18649 w_k1_ADM_s010
                                 + 1.24105 w_k1_AXP_s010
                                 + 1.07963 w_k1_DIS_s010
                                 + 1.24873 w_k1_FDX_s010
                                 + 0.928133 w_k1_IBM_s010
                                 + 1.23659 w_k1_INTC_s010
                                 + 1.03036 w_k1_KO_s010 + 1.11903 w_k1_MAT_s010
                                 + 1.21911 w_k1_MSFT_s010
                                 + 1.07607 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.654143 w_k1_ADM_s011
                                 + 0.826521 w_k1_AXP_s011
                                 + 0.843416 w_k1_DIS_s011
                                 + 0.899682 w_k1_FDX_s011
                                 + 0.630845 w_k1_IBM_s011
                                 + 0.922765 w_k1_INTC_s011
                                 + 0.688808 w_k1_KO_s011
                                 + 0.629504 w_k1_MAT_s011
                                 + 0.839534 w_k1_MSFT_s011
                                 + 0.83575 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.18108 w_k1_ADM_s011
                                 + 1.55518 w_k1_AXP_s011
                                 + 1.98769 w_k1_DIS_s011
                                 + 1.43191 w_k1_FDX_s011
                                 + 1.37935 w_k1_IBM_s011
                                 + 2.00375 w_k1_INTC_s011
                                 + 1.67683 w_k1_KO_s011 + 1.59448 w_k1_MAT_s011
                                 + 1.52019 w_k1_MSFT_s011
                                 + 1.24719 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.01644 w_k1_ADM_s011
                                 + 1.06657 w_k1_AXP_s011
                                 + 1.34019 w_k1_DIS_s011
                                 + 1.16666 w_k1_FDX_s011
                                 + 0.853425 w_k1_IBM_s011
                                 + 1.04806 w_k1_INTC_s011
                                 + 1.08754 w_k1_KO_s011
                                 + 0.970456 w_k1_MAT_s011
                                 + 1.16107 w_k1_MSFT_s011
                                 + 1.02839 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.992239 w_k1_ADM_s011
                                 + 1.26495 w_k1_AXP_s011
                                 + 1.13406 w_k1_DIS_s011
                                 + 1.03356 w_k1_FDX_s011
                                 + 1.35777 w_k1_IBM_s011
                                 + 1.16366 w_k1_INTC_s011
                                 + 1.16334 w_k1_KO_s011 + 1.63301 w_k1_MAT_s011
                                 + 1.00022 w_k1_MSFT_s011
                                 + 0.993053 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.748626 w_k1_ADM_s011
                                 + 1.28076 w_k1_AXP_s011
                                 + 1.27601 w_k1_DIS_s011
                                 + 0.995802 w_k1_FDX_s011
                                 + 1.91589 w_k1_IBM_s011
                                 + 0.930223 w_k1_INTC_s011
                                 + 1.43253 w_k1_KO_s011 + 2.16157 w_k1_MAT_s011
                                 + 0.928192 w_k1_MSFT_s011
                                 + 1.20057 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.892197 w_k1_ADM_s011
                                 + 0.985461 w_k1_AXP_s011
                                 + 0.871587 w_k1_DIS_s011
                                 + 0.914825 w_k1_FDX_s011
                                 + 1.16102 w_k1_IBM_s011
                                 + 0.979486 w_k1_INTC_s011
                                 + 1.15772 w_k1_KO_s011 + 1.09271 w_k1_MAT_s011
                                 + 0.92535 w_k1_MSFT_s011
                                 + 0.94319 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.990723 w_k1_ADM_s011
                                 + 1.51413 w_k1_AXP_s011
                                 + 1.28908 w_k1_DIS_s011
                                 + 1.31111 w_k1_FDX_s011
                                 + 1.08016 w_k1_IBM_s011
                                 + 1.08962 w_k1_INTC_s011
                                 + 1.14612 w_k1_KO_s011 + 1.67388 w_k1_MAT_s011
                                 + 1.04319 w_k1_MSFT_s011
                                 + 1.10784 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.69361 w_k1_ADM_s011
                                 + 0.811317 w_k1_AXP_s011
                                 + 0.791575 w_k1_DIS_s011
                                 + 0.81038 w_k1_FDX_s011
                                 + 0.949526 w_k1_IBM_s011
                                 + 1.05519 w_k1_INTC_s011
                                 + 0.905107 w_k1_KO_s011
                                 + 0.748494 w_k1_MAT_s011
                                 + 0.914079 w_k1_MSFT_s011
                                 + 0.87109 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.71997 w_k1_ADM_s012
                                 + 0.700429 w_k1_AXP_s012
                                 + 0.697782 w_k1_DIS_s012
                                 + 0.743637 w_k1_FDX_s012
                                 + 1.0695 w_k1_IBM_s012
                                 + 0.896388 w_k1_INTC_s012
                                 + 0.799515 w_k1_KO_s012
                                 + 0.688259 w_k1_MAT_s012
                                 + 0.78023 w_k1_MSFT_s012
                                 + 0.88075 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.837327 w_k1_ADM_s012
                                 + 0.72305 w_k1_AXP_s012
                                 + 0.675785 w_k1_DIS_s012
                                 + 0.812908 w_k1_FDX_s012
                                 + 0.749355 w_k1_IBM_s012
                                 + 0.819351 w_k1_INTC_s012
                                 + 0.762764 w_k1_KO_s012
                                 + 0.519698 w_k1_MAT_s012
                                 + 0.851374 w_k1_MSFT_s012
                                 + 0.920245 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.10574 w_k1_ADM_s012
                                 + 1.38723 w_k1_AXP_s012
                                 + 1.65165 w_k1_DIS_s012
                                 + 1.26554 w_k1_FDX_s012
                                 + 1.29109 w_k1_IBM_s012
                                 + 1.20172 w_k1_INTC_s012
                                 + 1.40369 w_k1_KO_s012 + 1.67537 w_k1_MAT_s012
                                 + 1.32348 w_k1_MSFT_s012
                                 + 1.29781 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.055 w_k1_ADM_s012
                                 + 1.54436 w_k1_AXP_s012
                                 + 1.20153 w_k1_DIS_s012
                                 + 1.23258 w_k1_FDX_s012
                                 + 1.08684 w_k1_IBM_s012
                                 + 0.963026 w_k1_INTC_s012
                                 + 0.907431 w_k1_KO_s012
                                 + 1.46406 w_k1_MAT_s012
                                 + 0.975795 w_k1_MSFT_s012
                                 + 1.07134 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.855708 w_k1_ADM_s012
                                 + 0.941644 w_k1_AXP_s012
                                 + 0.913967 w_k1_DIS_s012
                                 + 0.985663 w_k1_FDX_s012
                                 + 0.712107 w_k1_IBM_s012
                                 + 0.760083 w_k1_INTC_s012
                                 + 0.797502 w_k1_KO_s012 + 1.1007 w_k1_MAT_s012
                                 + 1.01095 w_k1_MSFT_s012 + 1.0194 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.890788 w_k1_ADM_s012
                                 + 1.23062 w_k1_AXP_s012
                                 + 0.843054 w_k1_DIS_s012
                                 + 0.977332 w_k1_FDX_s012 + 1.265 w_k1_IBM_s012
                                 + 0.823896 w_k1_INTC_s012
                                 + 1.04647 w_k1_KO_s012 + 1.34693 w_k1_MAT_s012
                                 + 0.972553 w_k1_MSFT_s012
                                 + 1.09748 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.826758 w_k1_ADM_s012
                                 + 0.947782 w_k1_AXP_s012
                                 + 1.11856 w_k1_DIS_s012
                                 + 0.891554 w_k1_FDX_s012
                                 + 1.65541 w_k1_IBM_s012
                                 + 0.817157 w_k1_INTC_s012
                                 + 1.2362 w_k1_KO_s012 + 2.00163 w_k1_MAT_s012
                                 + 0.947396 w_k1_MSFT_s012
                                 + 1.05072 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.845565 w_k1_ADM_s012
                                 + 1.01821 w_k1_AXP_s012
                                 + 1.30974 w_k1_DIS_s012
                                 + 1.13054 w_k1_FDX_s012
                                 + 1.09509 w_k1_IBM_s012
                                 + 0.992218 w_k1_INTC_s012
                                 + 1.02602 w_k1_KO_s012 + 1.02926 w_k1_MAT_s012
                                 + 1.16503 w_k1_MSFT_s012
                                 + 1.14483 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.910241 w_k1_ADM_s013
                                 + 0.897182 w_k1_AXP_s013 + 1.17 w_k1_DIS_s013
                                 + 0.991825 w_k1_FDX_s013
                                 + 1.05445 w_k1_IBM_s013
                                 + 0.893062 w_k1_INTC_s013
                                 + 0.952248 w_k1_KO_s013
                                 + 0.962387 w_k1_MAT_s013
                                 + 1.03052 w_k1_MSFT_s013
                                 + 1.03356 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.919167 w_k1_ADM_s013
                                 + 0.844224 w_k1_AXP_s013
                                 + 0.699027 w_k1_DIS_s013
                                 + 0.766253 w_k1_FDX_s013
                                 + 1.0916 w_k1_IBM_s013
                                 + 0.70281 w_k1_INTC_s013
                                 + 0.890077 w_k1_KO_s013
                                 + 1.00356 w_k1_MAT_s013
                                 + 0.755625 w_k1_MSFT_s013
                                 + 0.877431 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.00199 w_k1_ADM_s013
                                 + 1.09038 w_k1_AXP_s013
                                 + 0.957847 w_k1_DIS_s013
                                 + 0.82044 w_k1_FDX_s013
                                 + 1.43938 w_k1_IBM_s013
                                 + 0.841866 w_k1_INTC_s013
                                 + 1.14158 w_k1_KO_s013 + 1.20953 w_k1_MAT_s013
                                 + 0.816824 w_k1_MSFT_s013
                                 + 0.944161 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.1138 w_k1_ADM_s013
                                 + 1.25156 w_k1_AXP_s013
                                 + 1.53908 w_k1_DIS_s013
                                 + 1.27691 w_k1_FDX_s013
                                 + 0.781737 w_k1_IBM_s013
                                 + 0.947437 w_k1_INTC_s013
                                 + 1.12668 w_k1_KO_s013
                                 + 0.737063 w_k1_MAT_s013
                                 + 1.18231 w_k1_MSFT_s013
                                 + 1.06158 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.754901 w_k1_ADM_s013
                                 + 1.0137 w_k1_AXP_s013
                                 + 0.834942 w_k1_DIS_s013
                                 + 0.963918 w_k1_FDX_s013
                                 + 0.883118 w_k1_IBM_s013
                                 + 0.743769 w_k1_INTC_s013
                                 + 0.78153 w_k1_KO_s013 + 1.10884 w_k1_MAT_s013
                                 + 0.798729 w_k1_MSFT_s013
                                 + 0.83909 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.614458 w_k1_ADM_s013
                                 + 0.744659 w_k1_AXP_s013
                                 + 0.942207 w_k1_DIS_s013
                                 + 0.840661 w_k1_FDX_s013
                                 + 0.688301 w_k1_IBM_s013
                                 + 0.7862 w_k1_INTC_s013
                                 + 0.868573 w_k1_KO_s013
                                 + 0.658786 w_k1_MAT_s013
                                 + 0.818161 w_k1_MSFT_s013
                                 + 0.929794 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.976044 w_k1_ADM_s013
                                 + 1.35582 w_k1_AXP_s013
                                 + 0.985566 w_k1_DIS_s013
                                 + 1.06661 w_k1_FDX_s013
                                 + 0.828429 w_k1_IBM_s013
                                 + 0.875676 w_k1_INTC_s013
                                 + 0.763621 w_k1_KO_s013
                                 + 0.851686 w_k1_MAT_s013
                                 + 0.925144 w_k1_MSFT_s013
                                 + 0.939576 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.19328 w_k1_ADM_s013
                                 + 1.3056 w_k1_AXP_s013 + 1.72447 w_k1_DIS_s013
                                 + 1.18318 w_k1_FDX_s013
                                 + 1.43679 w_k1_IBM_s013
                                 + 1.00079 w_k1_INTC_s013
                                 + 1.50825 w_k1_KO_s013 + 1.65631 w_k1_MAT_s013
                                 + 1.18893 w_k1_MSFT_s013
                                 + 1.16958 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.740627 w_k1_ADM_s014
                                 + 0.677834 w_k1_AXP_s014
                                 + 0.696136 w_k1_DIS_s014
                                 + 0.837523 w_k1_FDX_s014
                                 + 0.738928 w_k1_IBM_s014
                                 + 0.81178 w_k1_INTC_s014
                                 + 0.599149 w_k1_KO_s014
                                 + 0.462114 w_k1_MAT_s014
                                 + 0.659433 w_k1_MSFT_s014
                                 + 0.742015 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.973148 w_k1_ADM_s014
                                 + 0.940814 w_k1_AXP_s014
                                 + 1.21098 w_k1_DIS_s014
                                 + 1.01973 w_k1_FDX_s014
                                 + 1.65389 w_k1_IBM_s014
                                 + 1.13054 w_k1_INTC_s014
                                 + 1.20373 w_k1_KO_s014 + 1.2794 w_k1_MAT_s014
                                 + 1.12429 w_k1_MSFT_s014
                                 + 1.04232 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.964632 w_k1_ADM_s014
                                 + 1.19779 w_k1_AXP_s014
                                 + 0.980921 w_k1_DIS_s014
                                 + 1.13281 w_k1_FDX_s014
                                 + 0.798075 w_k1_IBM_s014
                                 + 1.27335 w_k1_INTC_s014
                                 + 0.972864 w_k1_KO_s014
                                 + 0.870577 w_k1_MAT_s014
                                 + 1.09211 w_k1_MSFT_s014
                                 + 1.04106 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.3951 w_k1_ADM_s014
                                 + 1.52537 w_k1_AXP_s014
                                 + 1.51727 w_k1_DIS_s014
                                 + 1.35502 w_k1_FDX_s014
                                 + 1.20715 w_k1_IBM_s014
                                 + 1.21008 w_k1_INTC_s014
                                 + 1.22213 w_k1_KO_s014 + 1.09073 w_k1_MAT_s014
                                 + 1.19312 w_k1_MSFT_s014
                                 + 1.25805 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.985043 w_k1_ADM_s014
                                 + 1.07802 w_k1_AXP_s014
                                 + 1.05899 w_k1_DIS_s014
                                 + 1.08004 w_k1_FDX_s014
                                 + 1.13548 w_k1_IBM_s014
                                 + 0.943822 w_k1_INTC_s014
                                 + 1.01585 w_k1_KO_s014 + 1.08665 w_k1_MAT_s014
                                 + 0.942105 w_k1_MSFT_s014
                                 + 0.991977 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.920416 w_k1_ADM_s014
                                 + 0.832201 w_k1_AXP_s014
                                 + 0.868347 w_k1_DIS_s014
                                 + 0.920813 w_k1_FDX_s014
                                 + 1.05557 w_k1_IBM_s014
                                 + 0.806093 w_k1_INTC_s014
                                 + 0.873511 w_k1_KO_s014
                                 + 0.87282 w_k1_MAT_s014
                                 + 0.71658 w_k1_MSFT_s014
                                 + 0.881676 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.88586 w_k1_ADM_s014
                                 + 0.7904 w_k1_AXP_s014 + 1.08688 w_k1_DIS_s014
                                 + 1.10374 w_k1_FDX_s014
                                 + 0.940262 w_k1_IBM_s014
                                 + 0.986464 w_k1_INTC_s014
                                 + 1.00508 w_k1_KO_s014
                                 + 0.787577 w_k1_MAT_s014
                                 + 0.946854 w_k1_MSFT_s014
                                 + 1.01648 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.04931 w_k1_ADM_s014
                                 + 0.814338 w_k1_AXP_s014
                                 + 0.618085 w_k1_DIS_s014
                                 + 0.871779 w_k1_FDX_s014
                                 + 0.860835 w_k1_IBM_s014
                                 + 0.882809 w_k1_INTC_s014
                                 + 0.775197 w_k1_KO_s014
                                 + 0.819795 w_k1_MAT_s014
                                 + 0.884211 w_k1_MSFT_s014
                                 + 0.91612 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.680599 w_k1_ADM_s015
                                 + 0.804192 w_k1_AXP_s015
                                 + 0.93654 w_k1_DIS_s015
                                 + 0.899386 w_k1_FDX_s015
                                 + 1.28237 w_k1_IBM_s015
                                 + 1.06842 w_k1_INTC_s015 + 1.0145 w_k1_KO_s015
                                 + 1.4908 w_k1_MAT_s015
                                 + 1.07329 w_k1_MSFT_s015
                                 + 1.02343 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.741665 w_k1_ADM_s015
                                 + 0.724243 w_k1_AXP_s015
                                 + 0.776663 w_k1_DIS_s015
                                 + 0.762291 w_k1_FDX_s015
                                 + 0.970765 w_k1_IBM_s015
                                 + 0.884107 w_k1_INTC_s015
                                 + 0.951872 w_k1_KO_s015
                                 + 1.07236 w_k1_MAT_s015
                                 + 0.811756 w_k1_MSFT_s015
                                 + 0.921909 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.07491 w_k1_ADM_s015
                                 + 1.09955 w_k1_AXP_s015
                                 + 0.894673 w_k1_DIS_s015
                                 + 1.10383 w_k1_FDX_s015
                                 + 0.711444 w_k1_IBM_s015
                                 + 1.32375 w_k1_INTC_s015
                                 + 0.860035 w_k1_KO_s015
                                 + 1.00964 w_k1_MAT_s015
                                 + 1.13202 w_k1_MSFT_s015
                                 + 1.07632 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.11306 w_k1_ADM_s015
                                 + 1.03936 w_k1_AXP_s015
                                 + 1.04231 w_k1_DIS_s015 + 1.0257 w_k1_FDX_s015
                                 + 1.0415 w_k1_IBM_s015
                                 + 0.89456 w_k1_INTC_s015
                                 + 1.04944 w_k1_KO_s015 + 1.3486 w_k1_MAT_s015
                                 + 1.10674 w_k1_MSFT_s015
                                 + 1.05294 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.1029 w_k1_ADM_s015
                                 + 1.30232 w_k1_AXP_s015
                                 + 1.30997 w_k1_DIS_s015
                                 + 1.24868 w_k1_FDX_s015
                                 + 1.32373 w_k1_IBM_s015
                                 + 1.41157 w_k1_INTC_s015
                                 + 1.23571 w_k1_KO_s015 + 1.49813 w_k1_MAT_s015
                                 + 1.2244 w_k1_MSFT_s015 + 1.1913 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.665282 w_k1_ADM_s015
                                 + 0.732868 w_k1_AXP_s015
                                 + 0.615832 w_k1_DIS_s015
                                 + 0.784524 w_k1_FDX_s015
                                 + 0.659746 w_k1_IBM_s015
                                 + 0.889713 w_k1_INTC_s015
                                 + 0.68988 w_k1_KO_s015 + 1.00132 w_k1_MAT_s015
                                 + 0.841903 w_k1_MSFT_s015
                                 + 0.878757 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.947002 w_k1_ADM_s015
                                 + 0.925795 w_k1_AXP_s015
                                 + 0.785694 w_k1_DIS_s015
                                 + 0.938802 w_k1_FDX_s015
                                 + 0.964388 w_k1_IBM_s015
                                 + 1.03516 w_k1_INTC_s015
                                 + 0.902644 w_k1_KO_s015
                                 + 0.971232 w_k1_MAT_s015
                                 + 1.00024 w_k1_MSFT_s015
                                 + 1.03446 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.767143 w_k1_ADM_s015
                                 + 0.792979 w_k1_AXP_s015
                                 + 0.809529 w_k1_DIS_s015
                                 + 0.89077 w_k1_FDX_s015
                                 + 0.807959 w_k1_IBM_s015
                                 + 0.857315 w_k1_INTC_s015
                                 + 0.733401 w_k1_KO_s015
                                 + 0.868885 w_k1_MAT_s015
                                 + 0.919082 w_k1_MSFT_s015
                                 + 0.903267 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.894117 w_k1_ADM_s016
                                 + 0.722063 w_k1_AXP_s016
                                 + 0.713628 w_k1_DIS_s016
                                 + 0.849566 w_k1_FDX_s016
                                 + 0.989246 w_k1_IBM_s016
                                 + 0.779865 w_k1_INTC_s016
                                 + 0.897573 w_k1_KO_s016
                                 + 0.898276 w_k1_MAT_s016
                                 + 0.995708 w_k1_MSFT_s016
                                 + 0.823989 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.13452 w_k1_ADM_s016
                                 + 1.29084 w_k1_AXP_s016
                                 + 0.946837 w_k1_DIS_s016
                                 + 1.13883 w_k1_FDX_s016
                                 + 1.18464 w_k1_IBM_s016
                                 + 0.929185 w_k1_INTC_s016
                                 + 1.17416 w_k1_KO_s016 + 1.4082 w_k1_MAT_s016
                                 + 1.13085 w_k1_MSFT_s016
                                 + 1.09172 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.661017 w_k1_ADM_s016
                                 + 0.680956 w_k1_AXP_s016
                                 + 0.675072 w_k1_DIS_s016
                                 + 0.79282 w_k1_FDX_s016
                                 + 0.680255 w_k1_IBM_s016
                                 + 0.744239 w_k1_INTC_s016
                                 + 0.81062 w_k1_KO_s016
                                 + 0.765094 w_k1_MAT_s016
                                 + 0.948651 w_k1_MSFT_s016
                                 + 0.885517 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.3527 w_k1_ADM_s016
                                 + 1.60686 w_k1_AXP_s016
                                 + 1.59929 w_k1_DIS_s016
                                 + 1.51779 w_k1_FDX_s016
                                 + 1.19028 w_k1_IBM_s016
                                 + 0.999528 w_k1_INTC_s016
                                 + 1.23955 w_k1_KO_s016 + 1.50678 w_k1_MAT_s016
                                 + 1.29936 w_k1_MSFT_s016
                                 + 1.09251 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.13056 w_k1_ADM_s016
                                 + 0.77576 w_k1_AXP_s016
                                 + 0.723047 w_k1_DIS_s016
                                 + 1.00486 w_k1_FDX_s016
                                 + 0.607736 w_k1_IBM_s016
                                 + 0.902138 w_k1_INTC_s016
                                 + 0.812535 w_k1_KO_s016
                                 + 0.546101 w_k1_MAT_s016
                                 + 0.961314 w_k1_MSFT_s016
                                 + 0.770622 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.06517 w_k1_ADM_s016
                                 + 1.02038 w_k1_AXP_s016
                                 + 1.19077 w_k1_DIS_s016
                                 + 0.940022 w_k1_FDX_s016
                                 + 1.88882 w_k1_IBM_s016
                                 + 0.883897 w_k1_INTC_s016
                                 + 1.79694 w_k1_KO_s016 + 1.87926 w_k1_MAT_s016
                                 + 1.03947 w_k1_MSFT_s016
                                 + 1.18855 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.831948 w_k1_ADM_s016
                                 + 0.866345 w_k1_AXP_s016
                                 + 1.11004 w_k1_DIS_s016
                                 + 0.997303 w_k1_FDX_s016
                                 + 1.19574 w_k1_IBM_s016
                                 + 0.93031 w_k1_INTC_s016
                                 + 1.29621 w_k1_KO_s016 + 1.15032 w_k1_MAT_s016
                                 + 1.14496 w_k1_MSFT_s016
                                 + 0.922601 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.19538 w_k1_ADM_s016
                                 + 1.10946 w_k1_AXP_s016
                                 + 1.00685 w_k1_DIS_s016
                                 + 1.11959 w_k1_FDX_s016
                                 + 0.814443 w_k1_IBM_s016
                                 + 0.91216 w_k1_INTC_s016
                                 + 0.876291 w_k1_KO_s016
                                 + 0.862531 w_k1_MAT_s016
                                 + 1.12924 w_k1_MSFT_s016
                                 + 1.00402 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.926844 w_k1_ADM_s017
                                 + 1.24118 w_k1_AXP_s017
                                 + 0.891851 w_k1_DIS_s017
                                 + 1.00579 w_k1_FDX_s017
                                 + 1.01756 w_k1_IBM_s017
                                 + 0.99264 w_k1_INTC_s017 + 1.0105 w_k1_KO_s017
                                 + 1.71249 w_k1_MAT_s017
                                 + 1.18536 w_k1_MSFT_s017
                                 + 1.10452 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.39642 w_k1_ADM_s017
                                 + 1.31687 w_k1_AXP_s017 + 1.404 w_k1_DIS_s017
                                 + 1.29783 w_k1_FDX_s017
                                 + 1.07509 w_k1_IBM_s017
                                 + 1.36776 w_k1_INTC_s017
                                 + 1.19872 w_k1_KO_s017
                                 + 0.941802 w_k1_MAT_s017
                                 + 1.31247 w_k1_MSFT_s017
                                 + 1.16743 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.09577 w_k1_ADM_s017
                                 + 1.19322 w_k1_AXP_s017 + 1.0552 w_k1_DIS_s017
                                 + 0.999076 w_k1_FDX_s017
                                 + 1.3585 w_k1_IBM_s017
                                 + 1.05416 w_k1_INTC_s017
                                 + 1.08915 w_k1_KO_s017 + 1.48875 w_k1_MAT_s017
                                 + 1.12172 w_k1_MSFT_s017
                                 + 0.998302 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.917844 w_k1_ADM_s017
                                 + 0.701074 w_k1_AXP_s017
                                 + 0.816611 w_k1_DIS_s017
                                 + 0.762876 w_k1_FDX_s017
                                 + 0.950252 w_k1_IBM_s017
                                 + 0.790775 w_k1_INTC_s017
                                 + 0.846628 w_k1_KO_s017
                                 + 0.716701 w_k1_MAT_s017
                                 + 0.888106 w_k1_MSFT_s017
                                 + 0.83157 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.902823 w_k1_ADM_s017
                                 + 0.870477 w_k1_AXP_s017
                                 + 0.771908 w_k1_DIS_s017
                                 + 0.948014 w_k1_FDX_s017
                                 + 0.630291 w_k1_IBM_s017
                                 + 0.936881 w_k1_INTC_s017
                                 + 0.766899 w_k1_KO_s017
                                 + 0.610082 w_k1_MAT_s017
                                 + 0.918215 w_k1_MSFT_s017
                                 + 0.906531 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.34072 w_k1_ADM_s017
                                 + 1.10875 w_k1_AXP_s017
                                 + 1.01391 w_k1_DIS_s017
                                 + 1.04787 w_k1_FDX_s017
                                 + 0.839486 w_k1_IBM_s017
                                 + 1.1097 w_k1_INTC_s017 + 0.92848 w_k1_KO_s017
                                 + 0.535227 w_k1_MAT_s017
                                 + 1.16378 w_k1_MSFT_s017
                                 + 1.00162 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.989413 w_k1_ADM_s017
                                 + 1.04451 w_k1_AXP_s017
                                 + 1.24444 w_k1_DIS_s017
                                 + 0.949606 w_k1_FDX_s017
                                 + 1.57469 w_k1_IBM_s017
                                 + 1.12339 w_k1_INTC_s017
                                 + 1.25313 w_k1_KO_s017
                                 + 0.932143 w_k1_MAT_s017
                                 + 1.17425 w_k1_MSFT_s017
                                 + 1.04696 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.768049 w_k1_ADM_s017
                                 + 1.0981 w_k1_AXP_s017 + 0.82044 w_k1_DIS_s017
                                 + 0.930274 w_k1_FDX_s017
                                 + 0.993312 w_k1_IBM_s017
                                 + 0.942129 w_k1_INTC_s017
                                 + 0.849039 w_k1_KO_s017
                                 + 1.06785 w_k1_MAT_s017
                                 + 1.00832 w_k1_MSFT_s017
                                 + 0.948957 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.907558 w_k1_ADM_s100
                                 + 1.07659 w_k1_AXP_s100
                                 + 1.02324 w_k1_DIS_s100
                                 + 0.917675 w_k1_FDX_s100
                                 + 1.32941 w_k1_IBM_s100
                                 + 1.17692 w_k1_INTC_s100
                                 + 1.14366 w_k1_KO_s100 + 1.46356 w_k1_MAT_s100
                                 + 0.859664 w_k1_MSFT_s100
                                 + 0.973933 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.17634 w_k1_ADM_s100
                                 + 1.33559 w_k1_AXP_s100 + 1.3287 w_k1_DIS_s100
                                 + 1.02383 w_k1_FDX_s100
                                 + 1.46643 w_k1_IBM_s100
                                 + 1.19076 w_k1_INTC_s100
                                 + 1.32425 w_k1_KO_s100 + 1.37325 w_k1_MAT_s100
                                 + 1.07243 w_k1_MSFT_s100
                                 + 1.06574 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.38828 w_k1_ADM_s100
                                 + 1.70225 w_k1_AXP_s100
                                 + 1.47985 w_k1_DIS_s100
                                 + 1.29633 w_k1_FDX_s100
                                 + 1.35844 w_k1_IBM_s100
                                 + 1.16216 w_k1_INTC_s100
                                 + 1.30936 w_k1_KO_s100 + 1.71048 w_k1_MAT_s100
                                 + 0.976205 w_k1_MSFT_s100
                                 + 1.16872 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.41297 w_k1_ADM_s100
                                 + 1.48716 w_k1_AXP_s100
                                 + 1.38197 w_k1_DIS_s100
                                 + 1.31851 w_k1_FDX_s100
                                 + 0.940107 w_k1_IBM_s100
                                 + 1.11804 w_k1_INTC_s100
                                 + 1.08131 w_k1_KO_s100 + 1.09894 w_k1_MAT_s100
                                 + 1.1768 w_k1_MSFT_s100 + 1.11227 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.07745 w_k1_ADM_s100
                                 + 1.02082 w_k1_AXP_s100
                                 + 0.961601 w_k1_DIS_s100
                                 + 1.04496 w_k1_FDX_s100
                                 + 0.698327 w_k1_IBM_s100
                                 + 1.02195 w_k1_INTC_s100
                                 + 0.868896 w_k1_KO_s100
                                 + 0.596964 w_k1_MAT_s100
                                 + 0.994653 w_k1_MSFT_s100
                                 + 0.879184 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.984334 w_k1_ADM_s100
                                 + 0.781551 w_k1_AXP_s100
                                 + 0.905176 w_k1_DIS_s100
                                 + 0.76304 w_k1_FDX_s100
                                 + 1.21646 w_k1_IBM_s100
                                 + 0.819033 w_k1_INTC_s100
                                 + 1.06897 w_k1_KO_s100
                                 + 0.929876 w_k1_MAT_s100
                                 + 0.825659 w_k1_MSFT_s100
                                 + 0.980251 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.31705 w_k1_ADM_s100
                                 + 1.23241 w_k1_AXP_s100
                                 + 0.971401 w_k1_DIS_s100
                                 + 1.02294 w_k1_FDX_s100
                                 + 0.919206 w_k1_IBM_s100
                                 + 1.02859 w_k1_INTC_s100 + 1.0918 w_k1_KO_s100
                                 + 1.25956 w_k1_MAT_s100
                                 + 0.995718 w_k1_MSFT_s100
                                 + 1.06395 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.899018 w_k1_ADM_s100
                                 + 0.880964 w_k1_AXP_s100
                                 + 0.784064 w_k1_DIS_s100
                                 + 0.805184 w_k1_FDX_s100
                                 + 0.849034 w_k1_IBM_s100
                                 + 0.872416 w_k1_INTC_s100
                                 + 0.783739 w_k1_KO_s100
                                 + 0.862438 w_k1_MAT_s100
                                 + 0.775503 w_k1_MSFT_s100
                                 + 0.899506 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.787167 w_k1_ADM_s101
                                 + 0.830433 w_k1_AXP_s101
                                 + 0.975398 w_k1_DIS_s101
                                 + 1.03173 w_k1_FDX_s101
                                 + 0.747864 w_k1_IBM_s101
                                 + 1.29691 w_k1_INTC_s101
                                 + 0.931454 w_k1_KO_s101
                                 + 0.503271 w_k1_MAT_s101
                                 + 1.04843 w_k1_MSFT_s101
                                 + 0.993035 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.2326 w_k1_ADM_s101
                                 + 1.17021 w_k1_AXP_s101
                                 + 0.922297 w_k1_DIS_s101
                                 + 1.19615 w_k1_FDX_s101
                                 + 0.769683 w_k1_IBM_s101
                                 + 1.27702 w_k1_INTC_s101
                                 + 0.947681 w_k1_KO_s101
                                 + 0.584441 w_k1_MAT_s101
                                 + 0.988436 w_k1_MSFT_s101
                                 + 1.07157 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.41987 w_k1_ADM_s101
                                 + 1.73189 w_k1_AXP_s101
                                 + 1.41481 w_k1_DIS_s101
                                 + 1.47905 w_k1_FDX_s101 + 1.2494 w_k1_IBM_s101
                                 + 1.16152 w_k1_INTC_s101
                                 + 1.29977 w_k1_KO_s101 + 1.56669 w_k1_MAT_s101
                                 + 1.15155 w_k1_MSFT_s101
                                 + 1.30133 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.891055 w_k1_ADM_s101
                                 + 0.997375 w_k1_AXP_s101
                                 + 0.957096 w_k1_DIS_s101
                                 + 0.917514 w_k1_FDX_s101 + 1.1 w_k1_IBM_s101
                                 + 1.0968 w_k1_INTC_s101 + 1.06713 w_k1_KO_s101
                                 + 0.837098 w_k1_MAT_s101
                                 + 0.944901 w_k1_MSFT_s101
                                 + 0.98621 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.781664 w_k1_ADM_s101
                                 + 0.730415 w_k1_AXP_s101
                                 + 0.722975 w_k1_DIS_s101
                                 + 0.765943 w_k1_FDX_s101
                                 + 1.00301 w_k1_IBM_s101
                                 + 0.734625 w_k1_INTC_s101
                                 + 0.849335 w_k1_KO_s101
                                 + 0.844709 w_k1_MAT_s101
                                 + 0.704999 w_k1_MSFT_s101
                                 + 0.796582 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.06261 w_k1_ADM_s101
                                 + 1.22384 w_k1_AXP_s101
                                 + 1.54744 w_k1_DIS_s101
                                 + 1.25307 w_k1_FDX_s101
                                 + 1.09012 w_k1_IBM_s101
                                 + 1.30312 w_k1_INTC_s101
                                 + 1.30415 w_k1_KO_s101
                                 + 0.749847 w_k1_MAT_s101
                                 + 1.22299 w_k1_MSFT_s101
                                 + 1.13554 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.957779 w_k1_ADM_s101
                                 + 1.14541 w_k1_AXP_s101
                                 + 1.18794 w_k1_DIS_s101
                                 + 0.961026 w_k1_FDX_s101
                                 + 1.58031 w_k1_IBM_s101
                                 + 0.888571 w_k1_INTC_s101
                                 + 1.22059 w_k1_KO_s101 + 1.55617 w_k1_MAT_s101
                                 + 0.91643 w_k1_MSFT_s101 + 1.0735 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.687924 w_k1_ADM_s101
                                 + 0.805666 w_k1_AXP_s101
                                 + 0.553738 w_k1_DIS_s101
                                 + 0.779948 w_k1_FDX_s101
                                 + 0.710767 w_k1_IBM_s101
                                 + 0.715331 w_k1_INTC_s101
                                 + 0.524389 w_k1_KO_s101
                                 + 0.697979 w_k1_MAT_s101
                                 + 0.596587 w_k1_MSFT_s101
                                 + 0.78568 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.793696 w_k1_ADM_s102
                                 + 0.715174 w_k1_AXP_s102
                                 + 0.687275 w_k1_DIS_s102
                                 + 0.814907 w_k1_FDX_s102
                                 + 0.790045 w_k1_IBM_s102
                                 + 0.721261 w_k1_INTC_s102
                                 + 0.664406 w_k1_KO_s102
                                 + 1.04282 w_k1_MAT_s102
                                 + 0.863756 w_k1_MSFT_s102
                                 + 0.82956 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.804433 w_k1_ADM_s102
                                 + 1.03111 w_k1_AXP_s102
                                 + 0.880151 w_k1_DIS_s102
                                 + 0.856172 w_k1_FDX_s102
                                 + 1.17816 w_k1_IBM_s102
                                 + 0.827471 w_k1_INTC_s102
                                 + 0.792362 w_k1_KO_s102
                                 + 1.37189 w_k1_MAT_s102
                                 + 0.802028 w_k1_MSFT_s102
                                 + 0.89472 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.35992 w_k1_ADM_s102
                                 + 1.18078 w_k1_AXP_s102
                                 + 1.41972 w_k1_DIS_s102
                                 + 1.34837 w_k1_FDX_s102
                                 + 0.905827 w_k1_IBM_s102
                                 + 1.12149 w_k1_INTC_s102
                                 + 1.01179 w_k1_KO_s102
                                 + 0.876487 w_k1_MAT_s102
                                 + 1.24559 w_k1_MSFT_s102
                                 + 1.19995 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.10514 w_k1_ADM_s102
                                 + 0.891969 w_k1_AXP_s102
                                 + 1.15401 w_k1_DIS_s102
                                 + 0.962365 w_k1_FDX_s102
                                 + 1.15927 w_k1_IBM_s102
                                 + 0.807597 w_k1_INTC_s102
                                 + 1.04624 w_k1_KO_s102 + 1.33504 w_k1_MAT_s102
                                 + 1.02407 w_k1_MSFT_s102
                                 + 1.00425 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.713791 w_k1_ADM_s102
                                 + 0.863114 w_k1_AXP_s102
                                 + 0.792511 w_k1_DIS_s102
                                 + 0.827764 w_k1_FDX_s102
                                 + 1.43306 w_k1_IBM_s102
                                 + 0.88534 w_k1_INTC_s102
                                 + 1.03813 w_k1_KO_s102 + 2.05849 w_k1_MAT_s102
                                 + 0.851046 w_k1_MSFT_s102
                                 + 0.957999 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04898 w_k1_ADM_s102
                                 + 1.19631 w_k1_AXP_s102
                                 + 0.963566 w_k1_DIS_s102
                                 + 1.04735 w_k1_FDX_s102
                                 + 0.965952 w_k1_IBM_s102
                                 + 0.9472 w_k1_INTC_s102
                                 + 0.866373 w_k1_KO_s102
                                 + 1.27514 w_k1_MAT_s102
                                 + 1.06533 w_k1_MSFT_s102
                                 + 0.976554 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.40897 w_k1_ADM_s102
                                 + 1.27603 w_k1_AXP_s102
                                 + 1.69043 w_k1_DIS_s102
                                 + 1.32238 w_k1_FDX_s102
                                 + 1.40724 w_k1_IBM_s102
                                 + 1.25694 w_k1_INTC_s102
                                 + 1.29843 w_k1_KO_s102 + 1.37465 w_k1_MAT_s102
                                 + 1.31969 w_k1_MSFT_s102
                                 + 1.14738 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.932236 w_k1_ADM_s102
                                 + 0.897608 w_k1_AXP_s102
                                 + 1.15554 w_k1_DIS_s102
                                 + 1.09213 w_k1_FDX_s102
                                 + 0.74528 w_k1_IBM_s102
                                 + 1.05337 w_k1_INTC_s102
                                 + 0.879763 w_k1_KO_s102
                                 + 0.712425 w_k1_MAT_s102
                                 + 1.06493 w_k1_MSFT_s102
                                 + 0.859157 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.706294 w_k1_ADM_s103
                                 + 0.660713 w_k1_AXP_s103
                                 + 0.778677 w_k1_DIS_s103
                                 + 0.796993 w_k1_FDX_s103
                                 + 0.737326 w_k1_IBM_s103
                                 + 0.95185 w_k1_INTC_s103
                                 + 0.905099 w_k1_KO_s103
                                 + 0.738325 w_k1_MAT_s103
                                 + 0.971187 w_k1_MSFT_s103
                                 + 0.949232 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.811681 w_k1_ADM_s103
                                 + 0.891768 w_k1_AXP_s103
                                 + 1.03136 w_k1_DIS_s103
                                 + 1.03648 w_k1_FDX_s103
                                 + 0.723249 w_k1_IBM_s103
                                 + 1.06334 w_k1_INTC_s103
                                 + 0.923549 w_k1_KO_s103
                                 + 0.81608 w_k1_MAT_s103
                                 + 0.994016 w_k1_MSFT_s103
                                 + 1.05297 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05177 w_k1_ADM_s103
                                 + 0.862811 w_k1_AXP_s103
                                 + 1.00295 w_k1_DIS_s103
                                 + 1.04693 w_k1_FDX_s103
                                 + 0.938961 w_k1_IBM_s103
                                 + 0.910132 w_k1_INTC_s103
                                 + 0.974083 w_k1_KO_s103
                                 + 0.957921 w_k1_MAT_s103
                                 + 1.04593 w_k1_MSFT_s103
                                 + 1.12586 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.17484 w_k1_ADM_s103
                                 + 1.1881 w_k1_AXP_s103 + 1.37487 w_k1_DIS_s103
                                 + 1.13314 w_k1_FDX_s103
                                 + 1.51869 w_k1_IBM_s103
                                 + 1.23689 w_k1_INTC_s103 + 1.462 w_k1_KO_s103
                                 + 1.30539 w_k1_MAT_s103
                                 + 1.18063 w_k1_MSFT_s103
                                 + 1.26196 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.788358 w_k1_ADM_s103
                                 + 0.647255 w_k1_AXP_s103
                                 + 0.627655 w_k1_DIS_s103
                                 + 0.77533 w_k1_FDX_s103
                                 + 0.774765 w_k1_IBM_s103
                                 + 0.775571 w_k1_INTC_s103
                                 + 0.791401 w_k1_KO_s103
                                 + 0.813473 w_k1_MAT_s103
                                 + 0.782113 w_k1_MSFT_s103
                                 + 0.808374 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.33663 w_k1_ADM_s103
                                 + 1.21032 w_k1_AXP_s103
                                 + 1.54561 w_k1_DIS_s103
                                 + 1.40741 w_k1_FDX_s103
                                 + 1.06831 w_k1_IBM_s103
                                 + 1.28428 w_k1_INTC_s103
                                 + 1.29483 w_k1_KO_s103 + 1.02602 w_k1_MAT_s103
                                 + 1.2263 w_k1_MSFT_s103 + 1.22517 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.900181 w_k1_ADM_s103
                                 + 1.22593 w_k1_AXP_s103
                                 + 0.805648 w_k1_DIS_s103
                                 + 1.07119 w_k1_FDX_s103
                                 + 0.825243 w_k1_IBM_s103
                                 + 1.08318 w_k1_INTC_s103
                                 + 0.810526 w_k1_KO_s103
                                 + 1.03689 w_k1_MAT_s103
                                 + 0.900534 w_k1_MSFT_s103
                                 + 1.00439 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.914777 w_k1_ADM_s103
                                 + 0.927589 w_k1_AXP_s103
                                 + 0.860984 w_k1_DIS_s103
                                 + 0.858526 w_k1_FDX_s103
                                 + 1.16839 w_k1_IBM_s103
                                 + 0.99089 w_k1_INTC_s103
                                 + 1.08328 w_k1_KO_s103 + 1.06266 w_k1_MAT_s103
                                 + 0.844548 w_k1_MSFT_s103
                                 + 0.944718 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.02031 w_k1_ADM_s104
                                 + 0.865653 w_k1_AXP_s104
                                 + 1.00054 w_k1_DIS_s104
                                 + 0.935862 w_k1_FDX_s104
                                 + 1.22205 w_k1_IBM_s104
                                 + 1.14137 w_k1_INTC_s104
                                 + 1.08724 w_k1_KO_s104 + 1.07997 w_k1_MAT_s104
                                 + 1.1446 w_k1_MSFT_s104
                                 + 0.956524 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.24065 w_k1_ADM_s104
                                 + 1.41736 w_k1_AXP_s104 + 1.5846 w_k1_DIS_s104
                                 + 1.26498 w_k1_FDX_s104 + 1.3354 w_k1_IBM_s104
                                 + 1.32932 w_k1_INTC_s104
                                 + 1.51179 w_k1_KO_s104 + 1.84582 w_k1_MAT_s104
                                 + 1.25828 w_k1_MSFT_s104
                                 + 1.35725 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.948178 w_k1_ADM_s104
                                 + 0.946736 w_k1_AXP_s104
                                 + 0.765265 w_k1_DIS_s104
                                 + 0.915957 w_k1_FDX_s104
                                 + 0.775246 w_k1_IBM_s104
                                 + 0.931592 w_k1_INTC_s104
                                 + 0.71937 w_k1_KO_s104
                                 + 0.505386 w_k1_MAT_s104
                                 + 0.79142 w_k1_MSFT_s104
                                 + 0.835006 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.06374 w_k1_ADM_s104
                                 + 1.33542 w_k1_AXP_s104
                                 + 1.16277 w_k1_DIS_s104
                                 + 1.21763 w_k1_FDX_s104
                                 + 1.23284 w_k1_IBM_s104
                                 + 1.31057 w_k1_INTC_s104
                                 + 1.18154 w_k1_KO_s104 + 1.66464 w_k1_MAT_s104
                                 + 1.19248 w_k1_MSFT_s104
                                 + 1.12809 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.822666 w_k1_ADM_s104
                                 + 1.10472 w_k1_AXP_s104
                                 + 0.795723 w_k1_DIS_s104
                                 + 0.939506 w_k1_FDX_s104
                                 + 0.717353 w_k1_IBM_s104
                                 + 0.999762 w_k1_INTC_s104
                                 + 0.713088 w_k1_KO_s104
                                 + 1.07208 w_k1_MAT_s104
                                 + 1.02547 w_k1_MSFT_s104
                                 + 0.871412 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.940163 w_k1_ADM_s104
                                 + 0.719393 w_k1_AXP_s104
                                 + 0.601179 w_k1_DIS_s104
                                 + 0.733696 w_k1_FDX_s104
                                 + 0.76571 w_k1_IBM_s104
                                 + 0.952412 w_k1_INTC_s104
                                 + 0.773227 w_k1_KO_s104
                                 + 0.784759 w_k1_MAT_s104
                                 + 0.895939 w_k1_MSFT_s104
                                 + 0.863646 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.19238 w_k1_ADM_s104
                                 + 1.10497 w_k1_AXP_s104
                                 + 1.16594 w_k1_DIS_s104 + 1.1963 w_k1_FDX_s104
                                 + 0.869917 w_k1_IBM_s104
                                 + 1.21353 w_k1_INTC_s104
                                 + 0.945183 w_k1_KO_s104
                                 + 1.22771 w_k1_MAT_s104
                                 + 1.21692 w_k1_MSFT_s104
                                 + 0.911275 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.00413 w_k1_ADM_s104
                                 + 1.19757 w_k1_AXP_s104
                                 + 0.789915 w_k1_DIS_s104
                                 + 0.940883 w_k1_FDX_s104
                                 + 1.08403 w_k1_IBM_s104
                                 + 1.11118 w_k1_INTC_s104
                                 + 1.04907 w_k1_KO_s104 + 1.27063 w_k1_MAT_s104
                                 + 0.904042 w_k1_MSFT_s104
                                 + 0.991836 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.12849 w_k1_ADM_s105
                                 + 1.11433 w_k1_AXP_s105
                                 + 1.34081 w_k1_DIS_s105
                                 + 1.23509 w_k1_FDX_s105
                                 + 1.06994 w_k1_IBM_s105
                                 + 0.929368 w_k1_INTC_s105
                                 + 1.32054 w_k1_KO_s105 + 1.28665 w_k1_MAT_s105
                                 + 1.33535 w_k1_MSFT_s105
                                 + 1.10208 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.709751 w_k1_ADM_s105
                                 + 0.889987 w_k1_AXP_s105
                                 + 0.683437 w_k1_DIS_s105
                                 + 0.789765 w_k1_FDX_s105
                                 + 1.18462 w_k1_IBM_s105
                                 + 0.864795 w_k1_INTC_s105
                                 + 1.09939 w_k1_KO_s105 + 1.73107 w_k1_MAT_s105
                                 + 0.854883 w_k1_MSFT_s105
                                 + 0.949629 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.945603 w_k1_ADM_s105
                                 + 1.0505 w_k1_AXP_s105
                                 + 0.952266 w_k1_DIS_s105
                                 + 0.98097 w_k1_FDX_s105
                                 + 1.30373 w_k1_IBM_s105
                                 + 1.06337 w_k1_INTC_s105
                                 + 1.26919 w_k1_KO_s105 + 1.8049 w_k1_MAT_s105
                                 + 1.02075 w_k1_MSFT_s105
                                 + 0.986067 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.0716 w_k1_ADM_s105
                                 + 1.14873 w_k1_AXP_s105
                                 + 0.966347 w_k1_DIS_s105
                                 + 1.22925 w_k1_FDX_s105
                                 + 0.714157 w_k1_IBM_s105
                                 + 1.14075 w_k1_INTC_s105
                                 + 0.859176 w_k1_KO_s105
                                 + 0.593342 w_k1_MAT_s105
                                 + 1.1689 w_k1_MSFT_s105 + 0.94215 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.979431 w_k1_ADM_s105
                                 + 1.3758 w_k1_AXP_s105 + 1.29936 w_k1_DIS_s105
                                 + 1.18834 w_k1_FDX_s105
                                 + 1.45423 w_k1_IBM_s105
                                 + 1.0202 w_k1_INTC_s105 + 1.30407 w_k1_KO_s105
                                 + 1.65259 w_k1_MAT_s105
                                 + 1.09696 w_k1_MSFT_s105
                                 + 1.07396 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.832773 w_k1_ADM_s105
                                 + 0.797917 w_k1_AXP_s105
                                 + 0.714174 w_k1_DIS_s105
                                 + 0.839739 w_k1_FDX_s105
                                 + 0.76823 w_k1_IBM_s105
                                 + 0.847796 w_k1_INTC_s105
                                 + 0.931806 w_k1_KO_s105
                                 + 1.07723 w_k1_MAT_s105
                                 + 0.853966 w_k1_MSFT_s105
                                 + 0.880194 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.03707 w_k1_ADM_s105
                                 + 1.43673 w_k1_AXP_s105
                                 + 1.87304 w_k1_DIS_s105
                                 + 1.49217 w_k1_FDX_s105 + 1.448 w_k1_IBM_s105
                                 + 1.37741 w_k1_INTC_s105
                                 + 1.86524 w_k1_KO_s105 + 1.72106 w_k1_MAT_s105
                                 + 1.59387 w_k1_MSFT_s105 + 1.2681 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.938013 w_k1_ADM_s105
                                 + 0.807605 w_k1_AXP_s105
                                 + 1.02968 w_k1_DIS_s105
                                 + 0.98285 w_k1_FDX_s105
                                 + 0.841551 w_k1_IBM_s105
                                 + 0.902507 w_k1_INTC_s105
                                 + 1.13519 w_k1_KO_s105 + 1.0254 w_k1_MAT_s105
                                 + 1.15423 w_k1_MSFT_s105
                                 + 0.929365 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.971325 w_k1_ADM_s106
                                 + 1.05927 w_k1_AXP_s106
                                 + 0.975292 w_k1_DIS_s106
                                 + 1.03341 w_k1_FDX_s106
                                 + 0.917859 w_k1_IBM_s106
                                 + 1.06078 w_k1_INTC_s106
                                 + 0.937294 w_k1_KO_s106
                                 + 1.12494 w_k1_MAT_s106
                                 + 0.840355 w_k1_MSFT_s106
                                 + 0.933968 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.947495 w_k1_ADM_s106
                                 + 0.718996 w_k1_AXP_s106
                                 + 0.756493 w_k1_DIS_s106
                                 + 0.73258 w_k1_FDX_s106
                                 + 0.919802 w_k1_IBM_s106
                                 + 0.738208 w_k1_INTC_s106
                                 + 0.907879 w_k1_KO_s106
                                 + 1.15315 w_k1_MAT_s106
                                 + 0.778946 w_k1_MSFT_s106
                                 + 0.820692 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.12218 w_k1_ADM_s106
                                 + 1.07579 w_k1_AXP_s106
                                 + 1.03431 w_k1_DIS_s106 + 1.0199 w_k1_FDX_s106
                                 + 1.12245 w_k1_IBM_s106
                                 + 1.05489 w_k1_INTC_s106
                                 + 0.975959 w_k1_KO_s106
                                 + 1.10506 w_k1_MAT_s106
                                 + 0.942742 w_k1_MSFT_s106
                                 + 0.963994 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.999056 w_k1_ADM_s106
                                 + 0.907757 w_k1_AXP_s106
                                 + 1.03159 w_k1_DIS_s106
                                 + 1.04149 w_k1_FDX_s106
                                 + 0.747763 w_k1_IBM_s106
                                 + 1.02114 w_k1_INTC_s106
                                 + 0.870001 w_k1_KO_s106
                                 + 0.870652 w_k1_MAT_s106
                                 + 0.920915 w_k1_MSFT_s106
                                 + 0.893687 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.41848 w_k1_ADM_s106
                                 + 1.40368 w_k1_AXP_s106
                                 + 1.53077 w_k1_DIS_s106
                                 + 1.33356 w_k1_FDX_s106
                                 + 1.02909 w_k1_IBM_s106
                                 + 1.30519 w_k1_INTC_s106
                                 + 1.35223 w_k1_KO_s106 + 1.52449 w_k1_MAT_s106
                                 + 1.25518 w_k1_MSFT_s106
                                 + 1.18514 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.10853 w_k1_ADM_s106
                                 + 1.50715 w_k1_AXP_s106
                                 + 1.46171 w_k1_DIS_s106
                                 + 1.30228 w_k1_FDX_s106
                                 + 1.49348 w_k1_IBM_s106
                                 + 1.27933 w_k1_INTC_s106
                                 + 1.49963 w_k1_KO_s106 + 2.59821 w_k1_MAT_s106
                                 + 1.18698 w_k1_MSFT_s106
                                 + 1.23403 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.891313 w_k1_ADM_s106
                                 + 1.10749 w_k1_AXP_s106
                                 + 0.919909 w_k1_DIS_s106
                                 + 0.889249 w_k1_FDX_s106
                                 + 1.39233 w_k1_IBM_s106
                                 + 0.933868 w_k1_INTC_s106
                                 + 1.10498 w_k1_KO_s106 + 1.59777 w_k1_MAT_s106
                                 + 0.814618 w_k1_MSFT_s106
                                 + 0.907213 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.745115 w_k1_ADM_s106
                                 + 0.591696 w_k1_AXP_s106
                                 + 0.566343 w_k1_DIS_s106
                                 + 0.716679 w_k1_FDX_s106
                                 + 0.616837 w_k1_IBM_s106
                                 + 0.701834 w_k1_INTC_s106
                                 + 0.74092 w_k1_KO_s106
                                 + 0.725196 w_k1_MAT_s106
                                 + 0.718021 w_k1_MSFT_s106
                                 + 0.795028 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.10938 w_k1_ADM_s107
                                 + 0.992968 w_k1_AXP_s107
                                 + 1.11363 w_k1_DIS_s107 + 1.0144 w_k1_FDX_s107
                                 + 0.999848 w_k1_IBM_s107
                                 + 1.19714 w_k1_INTC_s107
                                 + 0.89097 w_k1_KO_s107
                                 + 0.655225 w_k1_MAT_s107
                                 + 0.958969 w_k1_MSFT_s107
                                 + 0.874939 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.02503 w_k1_ADM_s107
                                 + 1.13999 w_k1_AXP_s107
                                 + 0.869952 w_k1_DIS_s107
                                 + 0.884175 w_k1_FDX_s107
                                 + 1.74865 w_k1_IBM_s107
                                 + 0.990899 w_k1_INTC_s107
                                 + 1.24314 w_k1_KO_s107 + 1.82519 w_k1_MAT_s107
                                 + 0.874738 w_k1_MSFT_s107
                                 + 0.956738 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.03641 w_k1_ADM_s107
                                 + 1.26419 w_k1_AXP_s107
                                 + 1.03975 w_k1_DIS_s107
                                 + 1.08002 w_k1_FDX_s107
                                 + 1.05159 w_k1_IBM_s107
                                 + 1.07828 w_k1_INTC_s107
                                 + 0.901668 w_k1_KO_s107
                                 + 1.06381 w_k1_MAT_s107
                                 + 0.958148 w_k1_MSFT_s107
                                 + 0.880874 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.05679 w_k1_ADM_s107
                                 + 1.01758 w_k1_AXP_s107
                                 + 1.02653 w_k1_DIS_s107
                                 + 1.10805 w_k1_FDX_s107
                                 + 0.714121 w_k1_IBM_s107
                                 + 1.1643 w_k1_INTC_s107
                                 + 0.787586 w_k1_KO_s107
                                 + 0.661239 w_k1_MAT_s107
                                 + 1.08955 w_k1_MSFT_s107
                                 + 0.897096 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.39717 w_k1_ADM_s107
                                 + 1.82867 w_k1_AXP_s107
                                 + 1.65675 w_k1_DIS_s107
                                 + 1.55705 w_k1_FDX_s107
                                 + 1.16737 w_k1_IBM_s107
                                 + 1.32771 w_k1_INTC_s107
                                 + 1.30493 w_k1_KO_s107 + 1.03492 w_k1_MAT_s107
                                 + 1.24715 w_k1_MSFT_s107
                                 + 1.07284 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.768794 w_k1_ADM_s107
                                 + 0.755904 w_k1_AXP_s107
                                 + 0.557536 w_k1_DIS_s107
                                 + 0.706192 w_k1_FDX_s107
                                 + 1.15203 w_k1_IBM_s107
                                 + 0.949972 w_k1_INTC_s107
                                 + 0.760988 w_k1_KO_s107
                                 + 0.89445 w_k1_MAT_s107
                                 + 0.696297 w_k1_MSFT_s107
                                 + 0.768516 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.939541 w_k1_ADM_s107
                                 + 0.85211 w_k1_AXP_s107
                                 + 0.728168 w_k1_DIS_s107
                                 + 0.807924 w_k1_FDX_s107
                                 + 0.734806 w_k1_IBM_s107
                                 + 0.830635 w_k1_INTC_s107
                                 + 0.698156 w_k1_KO_s107
                                 + 0.626168 w_k1_MAT_s107
                                 + 0.717742 w_k1_MSFT_s107
                                 + 0.768019 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.2341 w_k1_ADM_s107
                                 + 1.7221 w_k1_AXP_s107 + 1.46879 w_k1_DIS_s107
                                 + 1.64401 w_k1_FDX_s107
                                 + 0.746194 w_k1_IBM_s107
                                 + 1.42792 w_k1_INTC_s107
                                 + 1.00236 w_k1_KO_s107 + 1.00752 w_k1_MAT_s107
                                 + 1.32187 w_k1_MSFT_s107
                                 + 1.03832 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.929995 w_k1_ADM_s110
                                 + 1.16029 w_k1_AXP_s110
                                 + 1.16645 w_k1_DIS_s110
                                 + 1.09831 w_k1_FDX_s110
                                 + 1.00113 w_k1_IBM_s110
                                 + 0.986696 w_k1_INTC_s110
                                 + 1.11599 w_k1_KO_s110
                                 + 0.969754 w_k1_MAT_s110
                                 + 1.14655 w_k1_MSFT_s110
                                 + 1.11306 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.96496 w_k1_ADM_s110
                                 + 1.75042 w_k1_AXP_s110
                                 + 1.93648 w_k1_DIS_s110
                                 + 1.38246 w_k1_FDX_s110
                                 + 1.61289 w_k1_IBM_s110
                                 + 1.39082 w_k1_INTC_s110
                                 + 1.63399 w_k1_KO_s110 + 1.55172 w_k1_MAT_s110
                                 + 1.56271 w_k1_MSFT_s110
                                 + 1.28046 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.695165 w_k1_ADM_s110
                                 + 0.70709 w_k1_AXP_s110
                                 + 0.71208 w_k1_DIS_s110
                                 + 0.719584 w_k1_FDX_s110
                                 + 0.697016 w_k1_IBM_s110
                                 + 0.984515 w_k1_INTC_s110
                                 + 0.900565 w_k1_KO_s110
                                 + 0.598925 w_k1_MAT_s110
                                 + 0.888001 w_k1_MSFT_s110
                                 + 0.901577 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.763762 w_k1_ADM_s110
                                 + 0.730279 w_k1_AXP_s110
                                 + 0.918561 w_k1_DIS_s110
                                 + 0.804114 w_k1_FDX_s110
                                 + 0.877968 w_k1_IBM_s110
                                 + 0.951319 w_k1_INTC_s110
                                 + 1.02027 w_k1_KO_s110
                                 + 0.627583 w_k1_MAT_s110
                                 + 0.947578 w_k1_MSFT_s110
                                 + 0.863933 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.23588 w_k1_ADM_s110
                                 + 1.15757 w_k1_AXP_s110
                                 + 1.74799 w_k1_DIS_s110 + 1.3813 w_k1_FDX_s110
                                 + 1.07705 w_k1_IBM_s110
                                 + 0.914543 w_k1_INTC_s110
                                 + 1.32736 w_k1_KO_s110 + 1.21507 w_k1_MAT_s110
                                 + 1.08388 w_k1_MSFT_s110
                                 + 1.24801 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.823285 w_k1_ADM_s110
                                 + 0.805757 w_k1_AXP_s110
                                 + 1.09805 w_k1_DIS_s110
                                 + 1.04508 w_k1_FDX_s110
                                 + 0.704026 w_k1_IBM_s110
                                 + 0.964226 w_k1_INTC_s110
                                 + 1.08607 w_k1_KO_s110
                                 + 0.838023 w_k1_MAT_s110
                                 + 1.22497 w_k1_MSFT_s110
                                 + 1.07538 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.80586 w_k1_ADM_s110
                                 + 0.671588 w_k1_AXP_s110
                                 + 0.820886 w_k1_DIS_s110
                                 + 0.716339 w_k1_FDX_s110
                                 + 1.21253 w_k1_IBM_s110
                                 + 0.884051 w_k1_INTC_s110
                                 + 0.981922 w_k1_KO_s110
                                 + 0.833403 w_k1_MAT_s110
                                 + 0.838365 w_k1_MSFT_s110
                                 + 0.911488 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.808614 w_k1_ADM_s110
                                 + 1.21956 w_k1_AXP_s110
                                 + 1.19626 w_k1_DIS_s110
                                 + 0.944633 w_k1_FDX_s110
                                 + 1.52934 w_k1_IBM_s110
                                 + 0.991181 w_k1_INTC_s110
                                 + 1.19724 w_k1_KO_s110 + 1.30559 w_k1_MAT_s110
                                 + 0.910659 w_k1_MSFT_s110
                                 + 0.939105 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.01252 w_k1_ADM_s111
                                 + 0.767179 w_k1_AXP_s111
                                 + 0.745853 w_k1_DIS_s111
                                 + 0.785057 w_k1_FDX_s111
                                 + 0.818751 w_k1_IBM_s111
                                 + 0.706507 w_k1_INTC_s111
                                 + 0.812466 w_k1_KO_s111
                                 + 0.76292 w_k1_MAT_s111
                                 + 0.701841 w_k1_MSFT_s111
                                 + 0.833651 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.27898 w_k1_ADM_s111
                                 + 0.934391 w_k1_AXP_s111
                                 + 1.17524 w_k1_DIS_s111
                                 + 1.05593 w_k1_FDX_s111
                                 + 0.93886 w_k1_IBM_s111
                                 + 0.932405 w_k1_INTC_s111
                                 + 1.07239 w_k1_KO_s111
                                 + 0.910243 w_k1_MAT_s111
                                 + 1.01745 w_k1_MSFT_s111
                                 + 1.00318 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.935511 w_k1_ADM_s111
                                 + 0.883681 w_k1_AXP_s111
                                 + 0.753725 w_k1_DIS_s111
                                 + 0.767795 w_k1_FDX_s111
                                 + 1.03805 w_k1_IBM_s111
                                 + 0.861417 w_k1_INTC_s111
                                 + 0.857845 w_k1_KO_s111
                                 + 0.934121 w_k1_MAT_s111
                                 + 0.745679 w_k1_MSFT_s111
                                 + 0.90756 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.37795 w_k1_ADM_s111
                                 + 1.5888 w_k1_AXP_s111 + 2.32092 w_k1_DIS_s111
                                 + 1.52312 w_k1_FDX_s111
                                 + 1.35295 w_k1_IBM_s111
                                 + 1.47513 w_k1_INTC_s111
                                 + 1.63846 w_k1_KO_s111 + 2.3042 w_k1_MAT_s111
                                 + 1.44999 w_k1_MSFT_s111
                                 + 1.29926 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.1158 w_k1_ADM_s111
                                 + 0.80873 w_k1_AXP_s111
                                 + 0.819898 w_k1_DIS_s111
                                 + 0.976051 w_k1_FDX_s111
                                 + 0.692477 w_k1_IBM_s111
                                 + 0.992486 w_k1_INTC_s111
                                 + 0.851712 w_k1_KO_s111
                                 + 0.624509 w_k1_MAT_s111
                                 + 0.953113 w_k1_MSFT_s111
                                 + 0.929836 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.6018 w_k1_ADM_s111
                                 + 1.29589 w_k1_AXP_s111
                                 + 1.58591 w_k1_DIS_s111
                                 + 1.17941 w_k1_FDX_s111
                                 + 1.12875 w_k1_IBM_s111
                                 + 1.01375 w_k1_INTC_s111
                                 + 1.35251 w_k1_KO_s111 + 1.30856 w_k1_MAT_s111
                                 + 1.27321 w_k1_MSFT_s111
                                 + 1.11976 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.08587 w_k1_ADM_s111
                                 + 1.10613 w_k1_AXP_s111 + 1.0305 w_k1_DIS_s111
                                 + 1.18073 w_k1_FDX_s111
                                 + 0.743444 w_k1_IBM_s111
                                 + 1.27691 w_k1_INTC_s111
                                 + 0.914502 w_k1_KO_s111
                                 + 0.874653 w_k1_MAT_s111
                                 + 1.20447 w_k1_MSFT_s111
                                 + 1.00612 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.28922 w_k1_ADM_s111
                                 + 1.32721 w_k1_AXP_s111
                                 + 1.03868 w_k1_DIS_s111
                                 + 1.09733 w_k1_FDX_s111 + 1.2768 w_k1_IBM_s111
                                 + 0.99615 w_k1_INTC_s111
                                 + 1.26566 w_k1_KO_s111 + 1.50918 w_k1_MAT_s111
                                 + 1.0239 w_k1_MSFT_s111 + 1.13654 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.35649 w_k1_ADM_s112
                                 + 1.10439 w_k1_AXP_s112
                                 + 1.34026 w_k1_DIS_s112
                                 + 1.24206 w_k1_FDX_s112
                                 + 0.726009 w_k1_IBM_s112
                                 + 1.20062 w_k1_INTC_s112
                                 + 1.21086 w_k1_KO_s112 + 1.04178 w_k1_MAT_s112
                                 + 1.24622 w_k1_MSFT_s112
                                 + 1.14516 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.05913 w_k1_ADM_s112
                                 + 1.24202 w_k1_AXP_s112
                                 + 1.18237 w_k1_DIS_s112
                                 + 1.05263 w_k1_FDX_s112
                                 + 1.22355 w_k1_IBM_s112
                                 + 0.962886 w_k1_INTC_s112
                                 + 1.41518 w_k1_KO_s112 + 2.13387 w_k1_MAT_s112
                                 + 1.08525 w_k1_MSFT_s112
                                 + 1.06182 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.592 w_k1_ADM_s112
                                 + 1.4883 w_k1_AXP_s112 + 1.83769 w_k1_DIS_s112
                                 + 1.4744 w_k1_FDX_s112 + 1.25924 w_k1_IBM_s112
                                 + 1.34002 w_k1_INTC_s112
                                 + 1.55203 w_k1_KO_s112 + 1.63036 w_k1_MAT_s112
                                 + 1.21596 w_k1_MSFT_s112 + 1.3108 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.17915 w_k1_ADM_s112
                                 + 1.29439 w_k1_AXP_s112
                                 + 1.43169 w_k1_DIS_s112
                                 + 1.23028 w_k1_FDX_s112
                                 + 1.08255 w_k1_IBM_s112
                                 + 1.1927 w_k1_INTC_s112 + 1.23386 w_k1_KO_s112
                                 + 1.62403 w_k1_MAT_s112
                                 + 1.1855 w_k1_MSFT_s112 + 1.07978 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.629639 w_k1_ADM_s112
                                 + 0.540111 w_k1_AXP_s112
                                 + 0.449069 w_k1_DIS_s112
                                 + 0.627701 w_k1_FDX_s112
                                 + 0.96856 w_k1_IBM_s112
                                 + 0.776223 w_k1_INTC_s112
                                 + 0.808034 w_k1_KO_s112
                                 + 1.05202 w_k1_MAT_s112
                                 + 0.690991 w_k1_MSFT_s112
                                 + 0.780657 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.842538 w_k1_ADM_s112
                                 + 0.974004 w_k1_AXP_s112
                                 + 0.970441 w_k1_DIS_s112
                                 + 0.872928 w_k1_FDX_s112
                                 + 1.19903 w_k1_IBM_s112
                                 + 0.954649 w_k1_INTC_s112
                                 + 1.15066 w_k1_KO_s112 + 2.06998 w_k1_MAT_s112
                                 + 0.921804 w_k1_MSFT_s112
                                 + 0.969815 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.885394 w_k1_ADM_s112
                                 + 0.672437 w_k1_AXP_s112
                                 + 0.849857 w_k1_DIS_s112
                                 + 0.936707 w_k1_FDX_s112
                                 + 0.66806 w_k1_IBM_s112
                                 + 0.879337 w_k1_INTC_s112
                                 + 0.850722 w_k1_KO_s112
                                 + 0.651008 w_k1_MAT_s112
                                 + 0.983139 w_k1_MSFT_s112
                                 + 0.926769 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.741963 w_k1_ADM_s112
                                 + 0.668777 w_k1_AXP_s112
                                 + 0.567678 w_k1_DIS_s112
                                 + 0.765445 w_k1_FDX_s112
                                 + 0.699498 w_k1_IBM_s112
                                 + 0.813892 w_k1_INTC_s112
                                 + 0.653925 w_k1_KO_s112
                                 + 0.83582 w_k1_MAT_s112
                                 + 0.742216 w_k1_MSFT_s112
                                 + 0.788534 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.973476 w_k1_ADM_s113
                                 + 1.02338 w_k1_AXP_s113
                                 + 1.41147 w_k1_DIS_s113
                                 + 1.17574 w_k1_FDX_s113
                                 + 0.795319 w_k1_IBM_s113
                                 + 0.990692 w_k1_INTC_s113
                                 + 1.20038 w_k1_KO_s113
                                 + 0.784619 w_k1_MAT_s113
                                 + 1.09278 w_k1_MSFT_s113
                                 + 1.07541 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.37713 w_k1_ADM_s113
                                 + 1.6719 w_k1_AXP_s113 + 1.49567 w_k1_DIS_s113
                                 + 1.55928 w_k1_FDX_s113
                                 + 0.908814 w_k1_IBM_s113
                                 + 0.930413 w_k1_INTC_s113
                                 + 1.32663 w_k1_KO_s113 + 1.15636 w_k1_MAT_s113
                                 + 1.18396 w_k1_MSFT_s113
                                 + 1.09898 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.895981 w_k1_ADM_s113
                                 + 0.887962 w_k1_AXP_s113
                                 + 0.807989 w_k1_DIS_s113
                                 + 0.912373 w_k1_FDX_s113
                                 + 0.808517 w_k1_IBM_s113
                                 + 0.904923 w_k1_INTC_s113
                                 + 0.952666 w_k1_KO_s113
                                 + 1.01639 w_k1_MAT_s113
                                 + 0.908177 w_k1_MSFT_s113
                                 + 0.979298 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.876778 w_k1_ADM_s113
                                 + 0.851467 w_k1_AXP_s113
                                 + 1.03854 w_k1_DIS_s113
                                 + 0.898749 w_k1_FDX_s113
                                 + 1.50782 w_k1_IBM_s113
                                 + 0.816439 w_k1_INTC_s113
                                 + 1.37672 w_k1_KO_s113 + 1.59262 w_k1_MAT_s113
                                 + 0.898702 w_k1_MSFT_s113
                                 + 1.07271 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.918462 w_k1_ADM_s113
                                 + 0.870069 w_k1_AXP_s113
                                 + 1.02975 w_k1_DIS_s113
                                 + 0.934415 w_k1_FDX_s113
                                 + 0.945597 w_k1_IBM_s113
                                 + 0.794461 w_k1_INTC_s113
                                 + 0.989149 w_k1_KO_s113
                                 + 0.908008 w_k1_MAT_s113
                                 + 0.913516 w_k1_MSFT_s113
                                 + 0.872793 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.682095 w_k1_ADM_s113
                                 + 0.641818 w_k1_AXP_s113
                                 + 0.560815 w_k1_DIS_s113
                                 + 0.718624 w_k1_FDX_s113
                                 + 0.810643 w_k1_IBM_s113
                                 + 0.698916 w_k1_INTC_s113
                                 + 0.706089 w_k1_KO_s113
                                 + 0.771227 w_k1_MAT_s113
                                 + 0.708557 w_k1_MSFT_s113
                                 + 0.810221 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.06667 w_k1_ADM_s113
                                 + 1.06024 w_k1_AXP_s113
                                 + 1.05433 w_k1_DIS_s113
                                 + 1.21797 w_k1_FDX_s113
                                 + 0.856094 w_k1_IBM_s113
                                 + 0.997762 w_k1_INTC_s113
                                 + 1.18132 w_k1_KO_s113
                                 + 0.865846 w_k1_MAT_s113
                                 + 1.07424 w_k1_MSFT_s113
                                 + 1.01398 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.11011 w_k1_ADM_s113
                                 + 1.1518 w_k1_AXP_s113 + 1.43283 w_k1_DIS_s113
                                 + 1.09472 w_k1_FDX_s113
                                 + 1.39167 w_k1_IBM_s113
                                 + 1.10105 w_k1_INTC_s113 + 1.5556 w_k1_KO_s113
                                 + 1.28999 w_k1_MAT_s113
                                 + 1.14946 w_k1_MSFT_s113 + 1.2122 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.09148 w_k1_ADM_s114
                                 + 1.30018 w_k1_AXP_s114 + 1.0359 w_k1_DIS_s114
                                 + 1.13413 w_k1_FDX_s114
                                 + 1.01676 w_k1_IBM_s114
                                 + 1.03291 w_k1_INTC_s114
                                 + 1.10771 w_k1_KO_s114 + 1.21653 w_k1_MAT_s114
                                 + 1.04592 w_k1_MSFT_s114
                                 + 1.18223 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.970784 w_k1_ADM_s114
                                 + 0.879084 w_k1_AXP_s114
                                 + 0.958332 w_k1_DIS_s114
                                 + 0.890399 w_k1_FDX_s114
                                 + 1.17471 w_k1_IBM_s114
                                 + 1.02971 w_k1_INTC_s114
                                 + 1.24486 w_k1_KO_s114 + 1.44805 w_k1_MAT_s114
                                 + 1.13694 w_k1_MSFT_s114
                                 + 0.997268 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.26265 w_k1_ADM_s114
                                 + 1.81555 w_k1_AXP_s114
                                 + 1.49224 w_k1_DIS_s114 + 1.2988 w_k1_FDX_s114
                                 + 1.61342 w_k1_IBM_s114
                                 + 1.10992 w_k1_INTC_s114
                                 + 1.47901 w_k1_KO_s114 + 2.04413 w_k1_MAT_s114
                                 + 1.1599 w_k1_MSFT_s114 + 1.3632 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.917044 w_k1_ADM_s114
                                 + 0.860221 w_k1_AXP_s114
                                 + 0.798217 w_k1_DIS_s114
                                 + 0.729956 w_k1_FDX_s114
                                 + 1.09313 w_k1_IBM_s114
                                 + 1.00911 w_k1_INTC_s114
                                 + 0.962493 w_k1_KO_s114
                                 + 1.02422 w_k1_MAT_s114
                                 + 0.782557 w_k1_MSFT_s114
                                 + 0.927972 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.972298 w_k1_ADM_s114
                                 + 1.05796 w_k1_AXP_s114 + 1.1446 w_k1_DIS_s114
                                 + 1.17192 w_k1_FDX_s114
                                 + 0.811377 w_k1_IBM_s114
                                 + 1.28158 w_k1_INTC_s114
                                 + 1.00216 w_k1_KO_s114
                                 + 0.753416 w_k1_MAT_s114
                                 + 1.18206 w_k1_MSFT_s114
                                 + 1.15671 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.0937 w_k1_ADM_s114
                                 + 1.22893 w_k1_AXP_s114
                                 + 1.20775 w_k1_DIS_s114
                                 + 0.985792 w_k1_FDX_s114
                                 + 1.55155 w_k1_IBM_s114
                                 + 0.80975 w_k1_INTC_s114
                                 + 1.24817 w_k1_KO_s114 + 2.11837 w_k1_MAT_s114
                                 + 0.903672 w_k1_MSFT_s114
                                 + 1.1208 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.927051 w_k1_ADM_s114
                                 + 0.844543 w_k1_AXP_s114
                                 + 0.684313 w_k1_DIS_s114
                                 + 0.893093 w_k1_FDX_s114
                                 + 0.756008 w_k1_IBM_s114
                                 + 0.828394 w_k1_INTC_s114
                                 + 0.65864 w_k1_KO_s114
                                 + 0.810225 w_k1_MAT_s114
                                 + 0.848549 w_k1_MSFT_s114
                                 + 0.799774 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.38501 w_k1_ADM_s114
                                 + 1.32681 w_k1_AXP_s114
                                 + 1.19025 w_k1_DIS_s114
                                 + 1.24859 w_k1_FDX_s114
                                 + 1.08457 w_k1_IBM_s114
                                 + 1.25666 w_k1_INTC_s114
                                 + 1.08025 w_k1_KO_s114 + 1.12243 w_k1_MAT_s114
                                 + 1.18794 w_k1_MSFT_s114
                                 + 1.26966 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.01501 w_k1_ADM_s115
                                 + 0.938993 w_k1_AXP_s115
                                 + 0.810635 w_k1_DIS_s115
                                 + 0.918769 w_k1_FDX_s115
                                 + 0.786626 w_k1_IBM_s115
                                 + 1.14475 w_k1_INTC_s115
                                 + 0.882858 w_k1_KO_s115
                                 + 0.711079 w_k1_MAT_s115
                                 + 0.964774 w_k1_MSFT_s115
                                 + 0.908058 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.766138 w_k1_ADM_s115
                                 + 0.765792 w_k1_AXP_s115
                                 + 0.702844 w_k1_DIS_s115
                                 + 0.801051 w_k1_FDX_s115
                                 + 0.680437 w_k1_IBM_s115
                                 + 0.900472 w_k1_INTC_s115
                                 + 0.706055 w_k1_KO_s115
                                 + 0.32246 w_k1_MAT_s115
                                 + 0.816127 w_k1_MSFT_s115
                                 + 0.761119 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.751977 w_k1_ADM_s115
                                 + 0.769355 w_k1_AXP_s115
                                 + 0.745517 w_k1_DIS_s115
                                 + 0.737853 w_k1_FDX_s115
                                 + 1.11633 w_k1_IBM_s115
                                 + 0.824018 w_k1_INTC_s115
                                 + 0.89326 w_k1_KO_s115
                                 + 0.827527 w_k1_MAT_s115
                                 + 0.809997 w_k1_MSFT_s115
                                 + 0.870033 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.688725 w_k1_ADM_s115
                                 + 0.932014 w_k1_AXP_s115
                                 + 0.808697 w_k1_DIS_s115
                                 + 0.893187 w_k1_FDX_s115
                                 + 1.01118 w_k1_IBM_s115
                                 + 1.12037 w_k1_INTC_s115
                                 + 0.952159 w_k1_KO_s115
                                 + 0.804577 w_k1_MAT_s115
                                 + 0.974843 w_k1_MSFT_s115
                                 + 0.982604 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.872001 w_k1_ADM_s115
                                 + 1.21131 w_k1_AXP_s115
                                 + 1.17071 w_k1_DIS_s115
                                 + 1.07656 w_k1_FDX_s115
                                 + 1.09287 w_k1_IBM_s115
                                 + 1.11253 w_k1_INTC_s115
                                 + 1.09921 w_k1_KO_s115 + 1.19753 w_k1_MAT_s115
                                 + 1.21763 w_k1_MSFT_s115
                                 + 1.01072 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.03934 w_k1_ADM_s115
                                 + 1.67779 w_k1_AXP_s115
                                 + 1.88265 w_k1_DIS_s115
                                 + 1.34993 w_k1_FDX_s115
                                 + 1.33982 w_k1_IBM_s115
                                 + 1.26554 w_k1_INTC_s115
                                 + 1.46323 w_k1_KO_s115 + 1.33003 w_k1_MAT_s115
                                 + 1.43499 w_k1_MSFT_s115
                                 + 1.19605 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.900336 w_k1_ADM_s115
                                 + 1.11038 w_k1_AXP_s115
                                 + 1.09984 w_k1_DIS_s115
                                 + 0.898874 w_k1_FDX_s115
                                 + 1.55013 w_k1_IBM_s115
                                 + 0.936194 w_k1_INTC_s115
                                 + 1.35266 w_k1_KO_s115 + 1.70474 w_k1_MAT_s115
                                 + 1.1352 w_k1_MSFT_s115 + 1.20429 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.23507 w_k1_ADM_s115
                                 + 1.33711 w_k1_AXP_s115
                                 + 1.15991 w_k1_DIS_s115
                                 + 1.22606 w_k1_FDX_s115
                                 + 0.653733 w_k1_IBM_s115
                                 + 1.10738 w_k1_INTC_s115
                                 + 0.914585 w_k1_KO_s115
                                 + 0.638015 w_k1_MAT_s115
                                 + 1.18995 w_k1_MSFT_s115
                                 + 1.08944 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.15414 w_k1_ADM_s116
                                 + 1.39769 w_k1_AXP_s116
                                 + 1.31274 w_k1_DIS_s116 + 1.0543 w_k1_FDX_s116
                                 + 1.50818 w_k1_IBM_s116
                                 + 0.961872 w_k1_INTC_s116
                                 + 1.18184 w_k1_KO_s116 + 2.28255 w_k1_MAT_s116
                                 + 1.0544 w_k1_MSFT_s116 + 1.22428 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.01768 w_k1_ADM_s116
                                 + 1.18712 w_k1_AXP_s116
                                 + 0.968606 w_k1_DIS_s116
                                 + 1.09328 w_k1_FDX_s116
                                 + 1.10432 w_k1_IBM_s116
                                 + 1.18479 w_k1_INTC_s116
                                 + 1.01732 w_k1_KO_s116 + 1.23008 w_k1_MAT_s116
                                 + 1.00413 w_k1_MSFT_s116
                                 + 1.00704 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.916881 w_k1_ADM_s116
                                 + 0.82527 w_k1_AXP_s116
                                 + 0.791767 w_k1_DIS_s116
                                 + 0.955424 w_k1_FDX_s116
                                 + 0.656993 w_k1_IBM_s116
                                 + 0.989572 w_k1_INTC_s116
                                 + 0.756751 w_k1_KO_s116
                                 + 0.562612 w_k1_MAT_s116
                                 + 0.913595 w_k1_MSFT_s116
                                 + 0.850972 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.11818 w_k1_ADM_s116
                                 + 0.968266 w_k1_AXP_s116
                                 + 0.776517 w_k1_DIS_s116
                                 + 0.867121 w_k1_FDX_s116
                                 + 1.09244 w_k1_IBM_s116
                                 + 0.873041 w_k1_INTC_s116
                                 + 1.03407 w_k1_KO_s116 + 1.6913 w_k1_MAT_s116
                                 + 0.905231 w_k1_MSFT_s116
                                 + 1.03075 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.792379 w_k1_ADM_s116
                                 + 0.920256 w_k1_AXP_s116
                                 + 0.86533 w_k1_DIS_s116
                                 + 0.837525 w_k1_FDX_s116
                                 + 1.44149 w_k1_IBM_s116
                                 + 0.974771 w_k1_INTC_s116
                                 + 1.18824 w_k1_KO_s116 + 1.18401 w_k1_MAT_s116
                                 + 0.866102 w_k1_MSFT_s116
                                 + 1.02567 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.876902 w_k1_ADM_s116
                                 + 1.27315 w_k1_AXP_s116
                                 + 1.05347 w_k1_DIS_s116
                                 + 0.997642 w_k1_FDX_s116
                                 + 0.939626 w_k1_IBM_s116
                                 + 1.23168 w_k1_INTC_s116
                                 + 0.978609 w_k1_KO_s116
                                 + 0.978519 w_k1_MAT_s116
                                 + 0.962648 w_k1_MSFT_s116
                                 + 1.03414 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.992344 w_k1_ADM_s116
                                 + 0.880484 w_k1_AXP_s116
                                 + 0.815377 w_k1_DIS_s116
                                 + 0.937206 w_k1_FDX_s116
                                 + 0.835337 w_k1_IBM_s116
                                 + 0.860653 w_k1_INTC_s116
                                 + 0.890638 w_k1_KO_s116
                                 + 1.18292 w_k1_MAT_s116
                                 + 0.865827 w_k1_MSFT_s116
                                 + 0.862196 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.19214 w_k1_ADM_s116
                                 + 1.58815 w_k1_AXP_s116
                                 + 1.45869 w_k1_DIS_s116
                                 + 1.36348 w_k1_FDX_s116
                                 + 1.12716 w_k1_IBM_s116
                                 + 1.31231 w_k1_INTC_s116
                                 + 1.24931 w_k1_KO_s116 + 1.75465 w_k1_MAT_s116
                                 + 1.34461 w_k1_MSFT_s116
                                 + 1.17618 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.985493 w_k1_ADM_s117
                                 + 1.40094 w_k1_AXP_s117 + 1.0663 w_k1_DIS_s117
                                 + 1.06049 w_k1_FDX_s117
                                 + 1.03537 w_k1_IBM_s117 + 1.317 w_k1_INTC_s117
                                 + 1.10607 w_k1_KO_s117 + 1.48482 w_k1_MAT_s117
                                 + 1.03869 w_k1_MSFT_s117
                                 + 1.09352 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.848337 w_k1_ADM_s117
                                 + 0.851805 w_k1_AXP_s117
                                 + 0.73242 w_k1_DIS_s117
                                 + 0.901251 w_k1_FDX_s117
                                 + 0.844098 w_k1_IBM_s117
                                 + 1.07952 w_k1_INTC_s117
                                 + 0.908607 w_k1_KO_s117
                                 + 0.929515 w_k1_MAT_s117
                                 + 0.985279 w_k1_MSFT_s117
                                 + 0.912947 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.00039 w_k1_ADM_s117
                                 + 1.37154 w_k1_AXP_s117
                                 + 1.39987 w_k1_DIS_s117
                                 + 1.13438 w_k1_FDX_s117
                                 + 1.32952 w_k1_IBM_s117
                                 + 1.18708 w_k1_INTC_s117
                                 + 1.34569 w_k1_KO_s117 + 1.81157 w_k1_MAT_s117
                                 + 1.35727 w_k1_MSFT_s117 + 1.2105 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.22208 w_k1_ADM_s117
                                 + 1.17593 w_k1_AXP_s117
                                 + 1.16414 w_k1_DIS_s117 + 1.209 w_k1_FDX_s117
                                 + 0.855819 w_k1_IBM_s117
                                 + 1.15358 w_k1_INTC_s117
                                 + 0.949811 w_k1_KO_s117
                                 + 1.17115 w_k1_MAT_s117
                                 + 1.14429 w_k1_MSFT_s117
                                 + 0.99777 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.02781 w_k1_ADM_s117
                                 + 0.738447 w_k1_AXP_s117
                                 + 0.715712 w_k1_DIS_s117
                                 + 0.82239 w_k1_FDX_s117
                                 + 0.666532 w_k1_IBM_s117
                                 + 0.838758 w_k1_INTC_s117
                                 + 0.727622 w_k1_KO_s117
                                 + 0.515632 w_k1_MAT_s117
                                 + 0.907331 w_k1_MSFT_s117
                                 + 0.909658 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.848233 w_k1_ADM_s117
                                 + 0.71408 w_k1_AXP_s117
                                 + 1.07778 w_k1_DIS_s117
                                 + 0.879885 w_k1_FDX_s117
                                 + 1.10849 w_k1_IBM_s117
                                 + 0.867072 w_k1_INTC_s117
                                 + 1.17651 w_k1_KO_s117 + 1.09598 w_k1_MAT_s117
                                 + 1.02268 w_k1_MSFT_s117
                                 + 1.03286 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.77942 w_k1_ADM_s117
                                 + 0.757775 w_k1_AXP_s117
                                 + 0.646433 w_k1_DIS_s117
                                 + 0.786471 w_k1_FDX_s117
                                 + 1.10923 w_k1_IBM_s117
                                 + 0.732821 w_k1_INTC_s117
                                 + 0.828361 w_k1_KO_s117
                                 + 1.30848 w_k1_MAT_s117
                                 + 0.721645 w_k1_MSFT_s117
                                 + 0.955344 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.07802 w_k1_ADM_s117
                                 + 1.74341 w_k1_AXP_s117
                                 + 1.30983 w_k1_DIS_s117
                                 + 1.05348 w_k1_FDX_s117
                                 + 1.66154 w_k1_IBM_s117
                                 + 1.01781 w_k1_INTC_s117
                                 + 1.36244 w_k1_KO_s117 + 2.76099 w_k1_MAT_s117
                                 + 0.996789 w_k1_MSFT_s117
                                 + 1.29364 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.026824003589784 w_k1_ADM_s000
                                 + 0.0556569039464902 w_k1_ADM_s001
                                 + 0.0522060691670496 w_k1_ADM_s002
                                 + 0.0477239274615687 w_k1_ADM_s003
                                 + 0.0325534049932921 w_k1_ADM_s004
                                 + 0.102017434016726 w_k1_ADM_s005
                                 + 0.0520171166028004 w_k1_ADM_s006
                                 + 0.0261712053359839 w_k1_ADM_s007
                                 + 0.0243607894725087 w_k1_ADM_s010
                                 + 0.0371922353300344 w_k1_ADM_s011
                                 + 0.029488893044774 w_k1_ADM_s012
                                 + 0.0483965675080212 w_k1_ADM_s013
                                 + 0.0155195871989074 w_k1_ADM_s014
                                 + 0.0261879940824526 w_k1_ADM_s015
                                 + 0.0244694426058798 w_k1_ADM_s016
                                 + 0.0256100530690191 w_k1_ADM_s017
                                 + 0.0416006616384466 w_k1_ADM_s100
                                 + 0.0249516227514481 w_k1_ADM_s101
                                 + 0.0463733998059794 w_k1_ADM_s102
                                 + 0.041912408098039 w_k1_ADM_s103
                                 + 0.0144158444385897 w_k1_ADM_s104
                                 + 0.0438545233812436 w_k1_ADM_s105
                                 + 0.0365105370525895 w_k1_ADM_s106
                                 + 0.0175039314281245 w_k1_ADM_s107
                                 + 0.00647804919499061 w_k1_ADM_s110
                                 + 0.00664726509412652 w_k1_ADM_s111
                                 + 0.0204467034176031 w_k1_ADM_s112
                                 + 0.0233941702712942 w_k1_ADM_s113
                                 + 0.00731654212598934 w_k1_ADM_s114
                                 + 0.00852141760853146 w_k1_ADM_s115
                                 + 0.0156444541039716 w_k1_ADM_s116
                                 + 0.0107651666210409 w_k1_ADM_s117
                                 + 0.0280130155354049 w_k1_AXP_s000
                                 + 0.0538124204791922 w_k1_AXP_s001
                                 + 0.0540155261041302 w_k1_AXP_s002
                                 + 0.0467092806158164 w_k1_AXP_s003
                                 + 0.035007088985251 w_k1_AXP_s004
                                 + 0.0972842192608002 w_k1_AXP_s005
                                 + 0.0535499340125905 w_k1_AXP_s006
                                 + 0.0270803925494698 w_k1_AXP_s007
                                 + 0.0264658457815253 w_k1_AXP_s010
                                 + 0.0460020276008773 w_k1_AXP_s011
                                 + 0.0343698881430028 w_k1_AXP_s012
                                 + 0.0544389714045423 w_k1_AXP_s013
                                 + 0.0156174744227219 w_k1_AXP_s014
                                 + 0.0278614158576078 w_k1_AXP_s015
                                 + 0.0236162736105347 w_k1_AXP_s016
                                 + 0.0249697760596712 w_k1_AXP_s017
                                 + 0.0424719892837845 w_k1_AXP_s100
                                 + 0.0271384698484482 w_k1_AXP_s101
                                 + 0.0447337005078931 w_k1_AXP_s102
                                 + 0.039882506198395 w_k1_AXP_s103
                                 + 0.0156727390379746 w_k1_AXP_s104
                                 + 0.0471999605953476 w_k1_AXP_s105
                                 + 0.0371907825899681 w_k1_AXP_s106
                                 + 0.0191267464016873 w_k1_AXP_s107
                                 + 0.00738805226829094 w_k1_AXP_s110
                                 + 0.00578220262722252 w_k1_AXP_s111
                                 + 0.0205146855978468 w_k1_AXP_s112
                                 + 0.0236102193573227 w_k1_AXP_s113
                                 + 0.00776000934576606 w_k1_AXP_s114
                                 + 0.0106263151886001 w_k1_AXP_s115
                                 + 0.0173372126540912 w_k1_AXP_s116
                                 + 0.0120841855250557 w_k1_AXP_s117
                                 + 0.0258375300700151 w_k1_DIS_s000
                                 + 0.0525292996092825 w_k1_DIS_s001
                                 + 0.0584332970806965 w_k1_DIS_s002
                                 + 0.0489454586103193 w_k1_DIS_s003
                                 + 0.0376182931947586 w_k1_DIS_s004
                                 + 0.101720109966742 w_k1_DIS_s005
                                 + 0.0524510656126046 w_k1_DIS_s006
                                 + 0.0285886246380807 w_k1_DIS_s007
                                 + 0.0253807644084271 w_k1_DIS_s010
                                 + 0.0463646853460665 w_k1_DIS_s011
                                 + 0.0359697601541463 w_k1_DIS_s012
                                 + 0.0581465075602657 w_k1_DIS_s013
                                 + 0.0161428672272079 w_k1_DIS_s014
                                 + 0.0271008006091412 w_k1_DIS_s015
                                 + 0.0237749404870588 w_k1_DIS_s016
                                 + 0.0245897850690813 w_k1_DIS_s017
                                 + 0.0392164234420428 w_k1_DIS_s100
                                 + 0.0259758554376993 w_k1_DIS_s101
                                 + 0.0482653248836179 w_k1_DIS_s102
                                 + 0.0432283248576603 w_k1_DIS_s103
                                 + 0.0138692099541765 w_k1_DIS_s104
                                 + 0.0477628080306556 w_k1_DIS_s105
                                 + 0.0375829581483475 w_k1_DIS_s106
                                 + 0.0166362714556381 w_k1_DIS_s107
                                 + 0.00852383734902879 w_k1_DIS_s110
                                 + 0.00589807096815579 w_k1_DIS_s111
                                 + 0.0215915858654377 w_k1_DIS_s112
                                 + 0.0256076789177181 w_k1_DIS_s113
                                 + 0.00699127490980652 w_k1_DIS_s114
                                 + 0.0101958552236614 w_k1_DIS_s115
                                 + 0.0154851550438142 w_k1_DIS_s116
                                 + 0.0112392201832428 w_k1_DIS_s117
                                 + 0.0272634673334087 w_k1_FDX_s000
                                 + 0.049256526171904 w_k1_FDX_s001
                                 + 0.054605166112194 w_k1_FDX_s002
                                 + 0.0487809159389829 w_k1_FDX_s003
                                 + 0.0366466258342335 w_k1_FDX_s004
                                 + 0.0994415733388903 w_k1_FDX_s005
                                 + 0.0514981788068483 w_k1_FDX_s006
                                 + 0.0277999456907283 w_k1_FDX_s007
                                 + 0.0261405660518818 w_k1_FDX_s010
                                 + 0.043765172645065 w_k1_FDX_s011
                                 + 0.0341193077493743 w_k1_FDX_s012
                                 + 0.0511384001807885 w_k1_FDX_s013
                                 + 0.0163969606832104 w_k1_FDX_s014
                                 + 0.0281603427156554 w_k1_FDX_s015
                                 + 0.0248315275253995 w_k1_FDX_s016
                                 + 0.0237833396741843 w_k1_FDX_s017
                                 + 0.0367846290726967 w_k1_FDX_s100
                                 + 0.0263708515248659 w_k1_FDX_s101
                                 + 0.0464904760712369 w_k1_FDX_s102
                                 + 0.0430229412619832 w_k1_FDX_s103
                                 + 0.0144542557547375 w_k1_FDX_s104
                                 + 0.048547516524931 w_k1_FDX_s105
                                 + 0.0357839541677814 w_k1_FDX_s106
                                 + 0.0175283077131523 w_k1_FDX_s107
                                 + 0.00748156906739319 w_k1_FDX_s110
                                 + 0.00563526217030976 w_k1_FDX_s111
                                 + 0.0204926090172045 w_k1_FDX_s112
                                 + 0.0244925265004705 w_k1_FDX_s113
                                 + 0.00721213360628733 w_k1_FDX_s114
                                 + 0.00957576003902059 w_k1_FDX_s115
                                 + 0.0158443986273237 w_k1_FDX_s116
                                 + 0.0111392401519379 w_k1_FDX_s117
                                 + 0.0263259138470418 w_k1_IBM_s000
                                 + 0.0512067703678399 w_k1_IBM_s001
                                 + 0.0568408785858859 w_k1_IBM_s002
                                 + 0.0486177970079138 w_k1_IBM_s003
                                 + 0.0407283078339949 w_k1_IBM_s004
                                 + 0.102510992052313 w_k1_IBM_s005
                                 + 0.051921506274959 w_k1_IBM_s006
                                 + 0.0269663856766851 w_k1_IBM_s007
                                 + 0.027093649932693 w_k1_IBM_s010
                                 + 0.0458605649663828 w_k1_IBM_s011
                                 + 0.036913898164332 w_k1_IBM_s012
                                 + 0.0529483999650922 w_k1_IBM_s013
                                 + 0.0165667009205038 w_k1_IBM_s014
                                 + 0.0283845352449619 w_k1_IBM_s015
                                 + 0.02505324380347 w_k1_IBM_s016
                                 + 0.0248401950583093 w_k1_IBM_s017
                                 + 0.0401039854843137 w_k1_IBM_s100
                                 + 0.0258626564622325 w_k1_IBM_s101
                                 + 0.0485473341695192 w_k1_IBM_s102
                                 + 0.0426515817753736 w_k1_IBM_s103
                                 + 0.0138725465102885 w_k1_IBM_s104
                                 + 0.0476455654489758 w_k1_IBM_s105
                                 + 0.0352770754632851 w_k1_IBM_s106
                                 + 0.0173210515838086 w_k1_IBM_s107
                                 + 0.00742515049212369 w_k1_IBM_s110
                                 + 0.00560403281538388 w_k1_IBM_s111
                                 + 0.0206737772293203 w_k1_IBM_s112
                                 + 0.0245172163364324 w_k1_IBM_s113
                                 + 0.00727199857141197 w_k1_IBM_s114
                                 + 0.00995559730957214 w_k1_IBM_s115
                                 + 0.015773567930146 w_k1_IBM_s116
                                 + 0.011358393829115 w_k1_IBM_s117
                                 + 0.0259528260784989 w_k1_INTC_s000
                                 + 0.054699389906781 w_k1_INTC_s001
                                 + 0.0539894139299658 w_k1_INTC_s002
                                 + 0.0436369443234878 w_k1_INTC_s003
                                 + 0.0299418679583021 w_k1_INTC_s004
                                 + 0.0901742513357199 w_k1_INTC_s005
                                 + 0.0494543690110491 w_k1_INTC_s006
                                 + 0.0313750524402993 w_k1_INTC_s007
                                 + 0.0304261783517345 w_k1_INTC_s010
                                 + 0.0460976840261715 w_k1_INTC_s011
                                 + 0.0312034083680712 w_k1_INTC_s012
                                 + 0.0443479679998485 w_k1_INTC_s013
                                 + 0.0153785946736394 w_k1_INTC_s014
                                 + 0.0305324837932545 w_k1_INTC_s015
                                 + 0.0211913831576846 w_k1_INTC_s016
                                 + 0.0249829625277912 w_k1_INTC_s017
                                 + 0.0382280089991704 w_k1_INTC_s100
                                 + 0.0271874097887544 w_k1_INTC_s101
                                 + 0.0425177224641261 w_k1_INTC_s102
                                 + 0.0432178855907156 w_k1_INTC_s103
                                 + 0.0158011370863082 w_k1_INTC_s104
                                 + 0.0453918903740794 w_k1_INTC_s105
                                 + 0.0357054440911697 w_k1_INTC_s106
                                 + 0.0181727602713291 w_k1_INTC_s107
                                 + 0.00729542414449651 w_k1_INTC_s110
                                 + 0.0053591423651917 w_k1_INTC_s111
                                 + 0.0203222377298233 w_k1_INTC_s112
                                 + 0.021885104197133 w_k1_INTC_s113
                                 + 0.00713658382158352 w_k1_INTC_s114
                                 + 0.0101755339241188 w_k1_INTC_s115
                                 + 0.0161376053301227 w_k1_INTC_s116
                                 + 0.011883739246663 w_k1_INTC_s117
                                 + 0.0251915732218971 w_k1_KO_s000
                                 + 0.0525384531994962 w_k1_KO_s001
                                 + 0.052464621394296 w_k1_KO_s002
                                 + 0.0485016227008998 w_k1_KO_s003
                                 + 0.0346431141150808 w_k1_KO_s004
                                 + 0.0951881995344449 w_k1_KO_s005
                                 + 0.0490987569806765 w_k1_KO_s006
                                 + 0.0280282755626666 w_k1_KO_s007
                                 + 0.0271373040161482 w_k1_KO_s010
                                 + 0.0461936252579522 w_k1_KO_s011
                                 + 0.033050528664533 w_k1_KO_s012
                                 + 0.0522893286439179 w_k1_KO_s013
                                 + 0.0151299445724472 w_k1_KO_s014
                                 + 0.027264661554345 w_k1_KO_s015
                                 + 0.026095402257383 w_k1_KO_s016
                                 + 0.0237099179788462 w_k1_KO_s017
                                 + 0.0391238802837463 w_k1_KO_s100
                                 + 0.0258997185695733 w_k1_KO_s101
                                 + 0.0424951511556115 w_k1_KO_s102
                                 + 0.0445591194763994 w_k1_KO_s103
                                 + 0.0139615189474459 w_k1_KO_s104
                                 + 0.0541356809849596 w_k1_KO_s105
                                 + 0.0372840262655747 w_k1_KO_s106
                                 + 0.0153637251174789 w_k1_KO_s107
                                 + 0.00829609012951169 w_k1_KO_s110
                                 + 0.00588236691108183 w_k1_KO_s111
                                 + 0.0230663193286297 w_k1_KO_s112
                                 + 0.0274288620023897 w_k1_KO_s113
                                 + 0.0071401376452743 w_k1_KO_s114
                                 + 0.00998391899124125 w_k1_KO_s115
                                 + 0.0155232495279564 w_k1_KO_s116
                                 + 0.0114827537381975 w_k1_KO_s117
                                 + 0.0262353912074197 w_k1_MAT_s000
                                 + 0.057834680166178 w_k1_MAT_s001
                                 + 0.0637798215711298 w_k1_MAT_s002
                                 + 0.0601817730718004 w_k1_MAT_s003
                                 + 0.0386991513406763 w_k1_MAT_s004
                                 + 0.100503434494741 w_k1_MAT_s005
                                 + 0.0416751424002586 w_k1_MAT_s006
                                 + 0.0232647028364029 w_k1_MAT_s007
                                 + 0.0367546713025256 w_k1_MAT_s010
                                 + 0.049215729537184 w_k1_MAT_s011
                                 + 0.0379661642306657 w_k1_MAT_s012
                                 + 0.0516340159170135 w_k1_MAT_s013
                                 + 0.014407373971167 w_k1_MAT_s014
                                 + 0.0341271420331837 w_k1_MAT_s015
                                 + 0.0259993461128545 w_k1_MAT_s016
                                 + 0.0221342917812891 w_k1_MAT_s017
                                 + 0.0420601741840015 w_k1_MAT_s100
                                 + 0.0221346464908685 w_k1_MAT_s101
                                 + 0.0575739077383432 w_k1_MAT_s102
                                 + 0.0417098248952126 w_k1_MAT_s103
                                 + 0.0169807442657993 w_k1_MAT_s104
                                 + 0.0604069727433562 w_k1_MAT_s105
                                 + 0.0477840776208628 w_k1_MAT_s106
                                 + 0.0159151984936886 w_k1_MAT_s107
                                 + 0.00695172538479386 w_k1_MAT_s110
                                 + 0.0059989695373429 w_k1_MAT_s111
                                 + 0.0309929946654975 w_k1_MAT_s112
                                 + 0.0252087280279219 w_k1_MAT_s113
                                 + 0.00830788078534119 w_k1_MAT_s114
                                 + 0.00957621389139437 w_k1_MAT_s115
                                 + 0.0202892690984928 w_k1_MAT_s116
                                 + 0.0145243168686034 w_k1_MAT_s117
                                 + 0.0257371468606749 w_k1_MSFT_s000
                                 + 0.0566952338025431 w_k1_MSFT_s001
                                 + 0.0545091809590026 w_k1_MSFT_s002
                                 + 0.0502034585221194 w_k1_MSFT_s003
                                 + 0.0346447436802221 w_k1_MSFT_s004
                                 + 0.0957548737804596 w_k1_MSFT_s005
                                 + 0.0520494891267798 w_k1_MSFT_s006
                                 + 0.0296650447135094 w_k1_MSFT_s007
                                 + 0.0298007013362868 w_k1_MSFT_s010
                                 + 0.0429536851545504 w_k1_MSFT_s011
                                 + 0.0343906331029217 w_k1_MSFT_s012
                                 + 0.0491325788736098 w_k1_MSFT_s013
                                 + 0.0145055496637661 w_k1_MSFT_s014
                                 + 0.0294550557572213 w_k1_MSFT_s015
                                 + 0.026073952110287 w_k1_MSFT_s016
                                 + 0.0259098701262018 w_k1_MSFT_s017
                                 + 0.0348963038807296 w_k1_MSFT_s100
                                 + 0.024037095959637 w_k1_MSFT_s101
                                 + 0.0469870158787858 w_k1_MSFT_s102
                                 + 0.0423978734698669 w_k1_MSFT_s103
                                 + 0.0147608207804593 w_k1_MSFT_s104
                                 + 0.0505535819479009 w_k1_MSFT_s105
                                 + 0.0331498477451911 w_k1_MSFT_s106
                                 + 0.0158600056666655 w_k1_MSFT_s107
                                 + 0.00796589029346338 w_k1_MSFT_s110
                                 + 0.00542905239991222 w_k1_MSFT_s111
                                 + 0.0206130463250789 w_k1_MSFT_s112
                                 + 0.0235968824525332 w_k1_MSFT_s113
                                 + 0.00710141014299517 w_k1_MSFT_s114
                                 + 0.010499423672689 w_k1_MSFT_s115
                                 + 0.0153270016771128 w_k1_MSFT_s116
                                 + 0.0116590462385541 w_k1_MSFT_s117
                                 + 0.0266686819045595 w_k1_PG_s000
                                 + 0.0508758142300501 w_k1_PG_s001
                                 + 0.0509694154732637 w_k1_PG_s002
                                 + 0.045608625783883 w_k1_PG_s003
                                 + 0.0323607973233443 w_k1_PG_s004
                                 + 0.100951365649487 w_k1_PG_s005
                                 + 0.0513335452552883 w_k1_PG_s006
                                 + 0.0271111192744246 w_k1_PG_s007
                                 + 0.0264584718952291 w_k1_PG_s010
                                 + 0.0423151859699037 w_k1_PG_s011
                                 + 0.0359243599411594 w_k1_PG_s012
                                 + 0.0508796962949858 w_k1_PG_s013
                                 + 0.0154150568786271 w_k1_PG_s014
                                 + 0.0289008325116415 w_k1_PG_s015
                                 + 0.0231423568146244 w_k1_PG_s016
                                 + 0.0235552518772119 w_k1_PG_s017
                                 + 0.037756602550066 w_k1_PG_s100
                                 + 0.0256688731551093 w_k1_PG_s101
                                 + 0.0448236772118587 w_k1_PG_s102
                                 + 0.0447796911314795 w_k1_PG_s103
                                 + 0.0139050819733727 w_k1_PG_s104
                                 + 0.046316451961768 w_k1_PG_s105
                                 + 0.0339005548855214 w_k1_PG_s106
                                 + 0.0149516841241338 w_k1_PG_s107
                                 + 0.00771189195044816 w_k1_PG_s110
                                 + 0.00562503694359932 w_k1_PG_s111
                                 + 0.0205073783728566 w_k1_PG_s112
                                 + 0.0243439483159324 w_k1_PG_s113
                                 + 0.00739375311701687 w_k1_PG_s114
                                 + 0.00955100293567924 w_k1_PG_s115
                                 + 0.0155005093561794 w_k1_PG_s116
                                 + 0.0114156812066704 w_k1_PG_s117 >= 1.07
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
