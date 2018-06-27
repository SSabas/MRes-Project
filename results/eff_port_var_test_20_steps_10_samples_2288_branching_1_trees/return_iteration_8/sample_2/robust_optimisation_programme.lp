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
 w_k1_ADM_s0_return_constraint:  1.14559 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.971179 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.0042 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.17573 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.949606 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.896642 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.684983 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.805344 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.704288 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.33797 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.06789 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.06995 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.01939 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.846821 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.753481 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.31974 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.895196 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.06122 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.02121 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.912774 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.910592 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.01093 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.13559 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.64386 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.20025 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.10825 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.08051 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.827079 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.843038 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.4685 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.938171 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.726916 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.12521 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.02897 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.986345 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.18379 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.1049 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.925328 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24665 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.870919 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.835979 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.28488 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.20928 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.849131 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.803572 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.829117 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.663147 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.54353 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.38184 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.08687 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.30572 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.12648 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.684756 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.32589 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.903698 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.12178 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.765869 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.2086 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.885947 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.48441 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.997373 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.69597 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.28945 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.856727 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.919358 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.64849 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.709684 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.41546 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.847149 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.668649 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.47688 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.00978 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.17553 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.28632 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.46351 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.859568 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.26201 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.864655 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.831573 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.44671 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.20927 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.814318 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.739495 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.835314 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.644061 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.70603 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.36295 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.13182 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.48813 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.12572 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.599 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.55163 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.792104 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.08986 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.719742 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.15564 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.78458 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.43666 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.2982 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 2.04126 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.15266 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.941796 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.939862 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.59728 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.615588 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.32633 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.981069 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.55357 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.59942 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.25385 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.790253 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.30451 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.35509 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.813648 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.12841 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.941802 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.918738 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.22631 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.06003 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.875107 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.708066 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.920809 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.693083 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.40963 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.42547 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.11115 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.24263 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.01672 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.710244 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.45166 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.759527 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.06668 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.858905 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.09885 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.891971 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.09922 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.07532 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.75229 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.12249 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.892559 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.06672 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.705648 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.664772 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.31687 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.90155 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.70146 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.56611 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.11487 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.982657 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.28946 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.198 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.973382 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.24374 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.87037 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.909779 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.34246 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17564 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.739551 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.29076 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.752976 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.859189 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.01913 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.705896 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.713179 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.55624 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.2126 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.941147 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.860631 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.45562 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.894011 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.611446 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.35735 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.986689 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.80307 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.921093 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.920823 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.60447 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.16623 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.762482 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.820828 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.34063 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.01555 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.40959 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.813495 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.920594 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.01948 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.18731 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.09606 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.74072 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.809064 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.04301 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.970314 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.948445 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.31197 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.997739 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.843444 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.833784 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.880779 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.81787 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.33749 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.33559 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.01708 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.12354 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.9945 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.69411 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.21866 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.89539 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.979304 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.887447 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.05705 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.737753 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.02974 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.18946 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.41086 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.20457 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.0106 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.06457 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.806309 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.856916 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.13801 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.08934 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.753229 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.39265 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.30697 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.0703 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.17942 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.11689 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.05222 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.14458 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.881348 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.880457 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.30664 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.05004 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.767618 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.970472 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.886793 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.912834 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.27755 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.934018 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.937656 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.3554 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.21295 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.861322 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.17784 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.09099 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.04934 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.720169 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.20528 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.947481 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.47543 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.22773 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.42127 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.33203 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.10106 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.836944 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.836851 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.06821 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.29824 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.23278 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.812066 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.22125 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.19775 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.07603 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.18287 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.37153 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.887815 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.32046 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.866345 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.814876 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.2784 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.13516 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.69448 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.12702 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.921859 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.910659 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.24648 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.00867 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.717197 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.54784 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.17438 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.05666 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.0409 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.4292 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.18177 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.61893 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.58177 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.32275 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.32742 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.01335 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.4378 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.83893 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.03549 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.784311 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.780475 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.18891 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.3547 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.30099 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.811809 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.27793 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.798241 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.38677 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.35476 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.88166 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.851133 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11621 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.949636 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.874892 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.17736 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.997353 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.884606 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.825609 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.976395 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.826445 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.39349 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.30894 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.1663 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.23718 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.09557 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.802353 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.30095 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.861715 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.1044 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.954181 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.18171 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.846205 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.05569 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.26148 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.63632 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.11041 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.03941 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.04418 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.810704 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.821191 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.26053 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.967247 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.689539 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.39721 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.08725 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.913622 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.10646 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.00144 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.01709 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12257 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.969212 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.908154 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.07749 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.01786 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.832824 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.841865 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.910771 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.816267 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.13704 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.12378 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.03793 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.21797 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.07208 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.907444 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.26143 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.03844 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.13166 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.919974 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.14769 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.05614 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.28735 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.07607 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.27987 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.16901 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.990388 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.978744 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.847539 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.851897 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.20587 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.996201 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.838819 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.32774 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.14907 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.961504 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.1522 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.28147 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.97977 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.25299989281845 z_k1_s0000
                                 - 0.526998770925932 z_k1_s0001
                                 - 0.290999704888835 z_k1_s0002
                                 - 0.365001218925623 z_k1_s0003
                                 - 0.121999906725899 z_k1_s0004
                                 - 0.254999941222146 z_k1_s0005
                                 - 0.472000901073783 z_k1_s0006
                                 - 0.483998976296286 z_k1_s0007
                                 - 0.274000295947257 z_k1_s0010
                                 - 0.978001671823717 z_k1_s0011
                                 - 0.211000053046892 z_k1_s0012
                                 - 1.0119986241208 z_k1_s0013
                                 - 0.258000264677338 z_k1_s0014
                                 - 0.630998957481361 z_k1_s0015
                                 - 0.637002083359074 z_k1_s0016
                                 - 0.790002741702852 z_k1_s0017
                                 - 0.473998471929584 z_k1_s0020
                                 - 0.405999867496489 z_k1_s0021
                                 - 1.10399811887254 z_k1_s0022
                                 - 0.228999701984171 z_k1_s0023
                                 - 0.312999939976321 z_k1_s0024
                                 - 0.404999711949094 z_k1_s0025
                                 - 0.707998096451461 z_k1_s0026
                                 - 0.636999452975815 z_k1_s0027
                                 - 0.550996979055618 z_k1_s0030
                                 - 0.351997413373312 z_k1_s0031
                                 - 0.273999220156099 z_k1_s0032
                                 - 0.411997528151138 z_k1_s0033
                                 - 0.122999746296054 z_k1_s0034
                                 - 0.521999368572121 z_k1_s0035
                                 - 0.548998605224659 z_k1_s0036
                                 - 0.476998467491267 z_k1_s0037
                                 - 0.175999373367136 z_k1_s0040
                                 - 0.132000034375227 z_k1_s0041
                                 - 0.0789999234147432 z_k1_s0042
                                 - 0.185999910887854 z_k1_s0043
                                 - 0.28499946834439 z_k1_s0044
                                 - 0.35500034899013 z_k1_s0045
                                 - 0.127999963466904 z_k1_s0046
                                 - 0.100000043508501 z_k1_s0047
                                 - 0.293000003088138 z_k1_s0050
                                 - 0.240999814768232 z_k1_s0051
                                 - 0.475999391884367 z_k1_s0052
                                 - 0.52699972772875 z_k1_s0053
                                 - 0.312999859595993 z_k1_s0054
                                 - 0.118999847971092 z_k1_s0055
                                 - 0.48099865426198 z_k1_s0056
                                 - 0.356999543913275 z_k1_s0057
                                 - 0.319999918020702 z_k1_s0060
                                 - 0.491000793290201 z_k1_s0061
                                 - 0.23999979846567 z_k1_s0062
                                 - 0.584999456241374 z_k1_s0063
                                 - 0.339999037585391 z_k1_s0064
                                 - 0.32399974193364 z_k1_s0065
                                 - 0.285998613763594 z_k1_s0066
                                 - 0.214999778610954 z_k1_s0067
                                 - 0.662998542879411 z_k1_s0070
                                 - 0.520999360590607 z_k1_s0071
                                 - 1.05000003804011 z_k1_s0072
                                 - 0.690999004692988 z_k1_s0073
                                 - 0.805999963371861 z_k1_s0074
                                 - 0.807997527791541 z_k1_s0075
                                 - 0.577999066427601 z_k1_s0076
                                 - 1.7949947987542 z_k1_s0077
                                 - 0.1550006050534 z_k1_s0100
                                 - 0.105000426645913 z_k1_s0101
                                 - 0.451001030411424 z_k1_s0102
                                 - 0.321001294414611 z_k1_s0103
                                 - 0.124000414722467 z_k1_s0104
                                 - 0.224001464739969 z_k1_s0105
                                 - 0.111000441122786 z_k1_s0106
                                 - 0.242000468366797 z_k1_s0107
                                 - 0.231000527100777 z_k1_s0110
                                 - 0.117000296973189 z_k1_s0111
                                 - 0.144000268582166 z_k1_s0112
                                 - 0.282001243475045 z_k1_s0113
                                 - 0.353999907762774 z_k1_s0114
                                 - 0.57400060976397 z_k1_s0115
                                 - 0.211999683744342 z_k1_s0116
                                 - 0.186000364418671 z_k1_s0117
                                 - 0.0990000583084048 z_k1_s0120
                                 - 0.132000704412462 z_k1_s0121
                                 - 0.0480000681496095 z_k1_s0122
                                 - 0.177000474552132 z_k1_s0123
                                 - 0.143999828448075 z_k1_s0124
                                 - 0.146000152454952 z_k1_s0125
                                 - 0.131000542409023 z_k1_s0126
                                 - 0.0630001481964801 z_k1_s0127
                                 - 0.820002918950764 z_k1_s0130
                                 - 0.241000796110944 z_k1_s0131
                                 - 0.203000761228177 z_k1_s0132
                                 - 0.215000572285262 z_k1_s0133
                                 - 0.310001134393679 z_k1_s0134
                                 - 0.771002582364171 z_k1_s0135
                                 - 0.188000522505322 z_k1_s0136
                                 - 0.418001966715437 z_k1_s0137
                                 - 0.156000219326437 z_k1_s0140
                                 - 0.267999897644843 z_k1_s0141
                                 - 0.239000872841148 z_k1_s0142
                                 - 0.259999212042678 z_k1_s0143
                                 - 0.25000066204274 z_k1_s0144
                                 - 0.550000995093475 z_k1_s0145
                                 - 0.442999899674207 z_k1_s0146
                                 - 0.141000087323762 z_k1_s0147
                                 - 0.149000260884453 z_k1_s0150
                                 - 0.0850001652992117 z_k1_s0151
                                 - 0.115000157004706 z_k1_s0152
                                 - 0.0590000969676589 z_k1_s0153
                                 - 0.114000459361686 z_k1_s0154
                                 - 0.151000600172045 z_k1_s0155
                                 - 0.115000157004706 z_k1_s0156
                                 - 0.156000032388696 z_k1_s0157
                                 - 0.283000943932137 z_k1_s0160
                                 - 0.303001682254993 z_k1_s0161
                                 - 0.236000638578883 z_k1_s0162
                                 - 0.160000692362945 z_k1_s0163
                                 - 0.175000993804123 z_k1_s0164
                                 - 0.216001582262446 z_k1_s0165
                                 - 0.160000692362945 z_k1_s0166
                                 - 0.149000538586337 z_k1_s0167
                                 - 0.138999859117749 z_k1_s0170
                                 - 0.100000412617778 z_k1_s0171
                                 - 0.0500000299086514 z_k1_s0172
                                 - 0.120999978886811 z_k1_s0173
                                 - 0.110000542079674 z_k1_s0174
                                 - 0.19400057115818 z_k1_s0175
                                 - 0.0870001561171936 z_k1_s0176
                                 - 0.0810000784400557 z_k1_s0177
                                 - 0.269000159977863 z_k1_s1000
                                 - 0.253000080831212 z_k1_s1001
                                 - 0.237000001684562 z_k1_s1002
                                 - 0.1959999762463 z_k1_s1003
                                 - 0.39900016449429 z_k1_s1004
                                 - 0.27699991575114 z_k1_s1005
                                 - 0.346999481567602 z_k1_s1006
                                 - 0.85999928763162 z_k1_s1007
                                 - 0.281999699075766 z_k1_s1010
                                 - 0.0980001531329131 z_k1_s1011
                                 - 0.322000880151293 z_k1_s1012
                                 - 0.161000034075331 z_k1_s1013
                                 - 0.196000103265668 z_k1_s1014
                                 - 0.263999675092172 z_k1_s1015
                                 - 0.224999284794125 z_k1_s1016
                                 - 0.481999514448671 z_k1_s1017
                                 - 0.648002726547099 z_k1_s1020
                                 - 0.316000702273814 z_k1_s1021
                                 - 0.578999578018486 z_k1_s1022
                                 - 0.65400072906545 z_k1_s1023
                                 - 0.818003119381152 z_k1_s1024
                                 - 0.797002987048494 z_k1_s1025
                                 - 0.533000645007601 z_k1_s1026
                                 - 0.404000849752474 z_k1_s1027
                                 - 0.546998022202295 z_k1_s1030
                                 - 0.289999130758798 z_k1_s1031
                                 - 0.828998574555845 z_k1_s1032
                                 - 0.943997313509577 z_k1_s1033
                                 - 0.411998732298118 z_k1_s1034
                                 - 0.222999207913871 z_k1_s1035
                                 - 0.257999199855151 z_k1_s1036
                                 - 0.817997342520969 z_k1_s1037
                                 - 0.451000578366675 z_k1_s1040
                                 - 1.54400295948224 z_k1_s1041
                                 - 1.23299859787376 z_k1_s1042
                                 - 0.642000178747627 z_k1_s1043
                                 - 0.585000170466301 z_k1_s1044
                                 - 0.918997953937571 z_k1_s1045
                                 - 1.42000234315047 z_k1_s1046
                                 - 1.32599703012101 z_k1_s1047
                                 - 1.06500410528331 z_k1_s1050
                                 - 1.6580032990828 z_k1_s1051
                                 - 1.44600392201518 z_k1_s1052
                                 - 0.62300101218832 z_k1_s1053
                                 - 0.428000921284239 z_k1_s1054
                                 - 1.42500066429906 z_k1_s1055
                                 - 1.79600408517635 z_k1_s1056
                                 - 0.581000812168621 z_k1_s1057
                                 - 0.208999897166993 z_k1_s1060
                                 - 0.274000629061115 z_k1_s1061
                                 - 0.348000986909757 z_k1_s1062
                                 - 0.609000439591939 z_k1_s1063
                                 - 0.153999951333577 z_k1_s1064
                                 - 0.623000716076839 z_k1_s1065
                                 - 0.155999954045702 z_k1_s1066
                                 - 0.20200001642457 z_k1_s1067
                                 - 0.384000496748528 z_k1_s1070
                                 - 0.567001316098493 z_k1_s1071
                                 - 0.960999057181284 z_k1_s1072
                                 - 0.783003124309814 z_k1_s1073
                                 - 0.576001043423334 z_k1_s1074
                                 - 0.629998810771294 z_k1_s1075
                                 - 1.53600318019628 z_k1_s1076
                                 - 0.529000810168383 z_k1_s1077
                                 - 0.300001436074839 z_k1_s1100
                                 - 0.187000803602938 z_k1_s1101
                                 - 0.0890003344851568 z_k1_s1102
                                 - 0.329000934042756 z_k1_s1103
                                 - 0.130000622299097 z_k1_s1104
                                 - 0.229000455351143 z_k1_s1105
                                 - 0.191001555153632 z_k1_s1106
                                 - 0.376001067477435 z_k1_s1107
                                 - 0.15600033182376 z_k1_s1110
                                 - 0.729999304339447 z_k1_s1111
                                 - 0.52000000307797 z_k1_s1112
                                 - 0.29600030719857 z_k1_s1113
                                 - 0.303000173607163 z_k1_s1114
                                 - 0.461999786376725 z_k1_s1115
                                 - 0.46100046726195 z_k1_s1116
                                 - 0.381001994020889 z_k1_s1117
                                 - 0.275000054527396 z_k1_s1120
                                 - 0.296000611195299 z_k1_s1121
                                 - 0.523001280774917 z_k1_s1122
                                 - 0.234000855135586 z_k1_s1123
                                 - 0.112000546685949 z_k1_s1124
                                 - 0.28300215402736 z_k1_s1125
                                 - 0.167000557591428 z_k1_s1126
                                 - 0.312000406177891 z_k1_s1127
                                 - 0.210000550409755 z_k1_s1130
                                 - 0.128000049011608 z_k1_s1131
                                 - 0.321000282711952 z_k1_s1132
                                 - 0.123000000097083 z_k1_s1133
                                 - 0.229000585884921 z_k1_s1134
                                 - 0.143000045355154 z_k1_s1135
                                 - 0.163000541813307 z_k1_s1136
                                 - 0.18699987420286 z_k1_s1137
                                 - 0.15199955977472 z_k1_s1140
                                 - 0.181999535256412 z_k1_s1141
                                 - 0.122999721725395 z_k1_s1142
                                 - 0.375999534704284 z_k1_s1143
                                 - 0.359998125784366 z_k1_s1144
                                 - 0.12199978179252 z_k1_s1145
                                 - 0.118999764494402 z_k1_s1146
                                 - 0.540998511105874 z_k1_s1147
                                 - 0.074000058335671 z_k1_s1150
                                 - 0.122000029088518 z_k1_s1151
                                 - 0.0749999715388292 z_k1_s1152
                                 - 0.0990000948152947 z_k1_s1153
                                 - 0.225000466216655 z_k1_s1154
                                 - 0.16799948393484 z_k1_s1155
                                 - 0.290999426226517 z_k1_s1156
                                 - 0.325000060534982 z_k1_s1157
                                 - 0.0720000491465781 z_k1_s1160
                                 - 0.0579999894513933 z_k1_s1161
                                 - 0.247000082836149 z_k1_s1162
                                 - 0.102999984542993 z_k1_s1163
                                 - 0.0769999958234047 z_k1_s1164
                                 - 0.171000057348103 z_k1_s1165
                                 - 0.0700000134758283 z_k1_s1166
                                 - 0.152000050976092 z_k1_s1167
                                 - 0.616001944525159 z_k1_s1170
                                 - 0.666998935457196 z_k1_s1171
                                 - 0.188999893751804 z_k1_s1172
                                 - 0.476998364373147 z_k1_s1173
                                 - 0.498001543346629 z_k1_s1174
                                 - 0.86299915253325 z_k1_s1175
                                 - 0.325000116506641 z_k1_s1176
                                 - 0.783002399972703 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.897301 w_k1_ADM_s000
                                 + 1.10003 w_k1_AXP_s000
                                 + 0.77806 w_k1_DIS_s000
                                 + 0.899448 w_k1_FDX_s000
                                 + 0.844637 w_k1_IBM_s000
                                 + 0.903566 w_k1_INTC_s000
                                 + 0.963299 w_k1_KO_s000
                                 + 1.10008 w_k1_MAT_s000
                                 + 0.936959 w_k1_MSFT_s000
                                 + 1.07409 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.4192 w_k1_ADM_s000
                                 + 1.50198 w_k1_AXP_s000
                                 + 1.31715 w_k1_DIS_s000
                                 + 1.37147 w_k1_FDX_s000
                                 + 1.02782 w_k1_IBM_s000
                                 + 1.32245 w_k1_INTC_s000
                                 + 1.18503 w_k1_KO_s000 + 1.37423 w_k1_MAT_s000
                                 + 1.19632 w_k1_MSFT_s000
                                 + 1.29846 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.861362 w_k1_ADM_s000
                                 + 0.770498 w_k1_AXP_s000
                                 + 1.01869 w_k1_DIS_s000
                                 + 1.05345 w_k1_FDX_s000
                                 + 0.864878 w_k1_IBM_s000
                                 + 1.13945 w_k1_INTC_s000
                                 + 1.03171 w_k1_KO_s000
                                 + 0.851841 w_k1_MAT_s000
                                 + 1.15584 w_k1_MSFT_s000
                                 + 1.10247 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.887496 w_k1_ADM_s000
                                 + 0.839483 w_k1_AXP_s000
                                 + 0.701797 w_k1_DIS_s000
                                 + 0.891881 w_k1_FDX_s000
                                 + 0.651994 w_k1_IBM_s000
                                 + 0.794746 w_k1_INTC_s000
                                 + 0.698188 w_k1_KO_s000
                                 + 0.697284 w_k1_MAT_s000
                                 + 0.751405 w_k1_MSFT_s000
                                 + 0.88133 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.22696 w_k1_ADM_s000
                                 + 1.16184 w_k1_AXP_s000
                                 + 1.28035 w_k1_DIS_s000
                                 + 1.29373 w_k1_FDX_s000
                                 + 0.677874 w_k1_IBM_s000
                                 + 1.23205 w_k1_INTC_s000
                                 + 1.05906 w_k1_KO_s000
                                 + 0.711945 w_k1_MAT_s000
                                 + 1.21412 w_k1_MSFT_s000
                                 + 1.14484 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.01051 w_k1_ADM_s000
                                 + 1.40812 w_k1_AXP_s000
                                 + 1.37515 w_k1_DIS_s000
                                 + 1.14731 w_k1_FDX_s000 + 1.6295 w_k1_IBM_s000
                                 + 0.953784 w_k1_INTC_s000
                                 + 1.16229 w_k1_KO_s000 + 1.66928 w_k1_MAT_s000
                                 + 1.01147 w_k1_MSFT_s000
                                 + 1.22018 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.950032 w_k1_ADM_s000
                                 + 1.13436 w_k1_AXP_s000
                                 + 1.01771 w_k1_DIS_s000
                                 + 0.952964 w_k1_FDX_s000
                                 + 1.30099 w_k1_IBM_s000
                                 + 1.08347 w_k1_INTC_s000
                                 + 1.24036 w_k1_KO_s000 + 1.62572 w_k1_MAT_s000
                                 + 0.996141 w_k1_MSFT_s000
                                 + 1.11731 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.970012 w_k1_ADM_s000
                                 + 0.668504 w_k1_AXP_s000
                                 + 0.80221 w_k1_DIS_s000
                                 + 0.856996 w_k1_FDX_s000
                                 + 0.956885 w_k1_IBM_s000
                                 + 0.958942 w_k1_INTC_s000
                                 + 0.936814 w_k1_KO_s000 + 1.0127 w_k1_MAT_s000
                                 + 0.907752 w_k1_MSFT_s000
                                 + 0.994531 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.896054 w_k1_ADM_s001
                                 + 0.995534 w_k1_AXP_s001
                                 + 0.852421 w_k1_DIS_s001
                                 + 0.999189 w_k1_FDX_s001
                                 + 0.580539 w_k1_IBM_s001
                                 + 1.11643 w_k1_INTC_s001
                                 + 0.680122 w_k1_KO_s001
                                 + 0.547956 w_k1_MAT_s001
                                 + 0.949357 w_k1_MSFT_s001
                                 + 0.864315 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.76552 w_k1_ADM_s001
                                 + 0.722164 w_k1_AXP_s001
                                 + 0.680616 w_k1_DIS_s001
                                 + 0.756637 w_k1_FDX_s001
                                 + 0.758811 w_k1_IBM_s001
                                 + 0.916436 w_k1_INTC_s001
                                 + 0.778884 w_k1_KO_s001
                                 + 0.593481 w_k1_MAT_s001
                                 + 0.771893 w_k1_MSFT_s001
                                 + 0.852858 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.15144 w_k1_ADM_s001
                                 + 0.870853 w_k1_AXP_s001
                                 + 0.70256 w_k1_DIS_s001
                                 + 0.94492 w_k1_FDX_s001
                                 + 0.67557 w_k1_IBM_s001
                                 + 0.856138 w_k1_INTC_s001
                                 + 0.675116 w_k1_KO_s001
                                 + 0.598234 w_k1_MAT_s001
                                 + 0.840861 w_k1_MSFT_s001
                                 + 0.822164 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.897768 w_k1_ADM_s001
                                 + 1.19215 w_k1_AXP_s001
                                 + 1.04345 w_k1_DIS_s001
                                 + 1.01366 w_k1_FDX_s001
                                 + 1.26568 w_k1_IBM_s001
                                 + 1.35842 w_k1_INTC_s001 + 1.0847 w_k1_KO_s001
                                 + 1.03556 w_k1_MAT_s001
                                 + 0.958635 w_k1_MSFT_s001
                                 + 1.03615 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.2227 w_k1_ADM_s001
                                 + 1.41467 w_k1_AXP_s001
                                 + 2.08123 w_k1_DIS_s001 + 1.2581 w_k1_FDX_s001
                                 + 1.47518 w_k1_IBM_s001
                                 + 1.27787 w_k1_INTC_s001
                                 + 1.38905 w_k1_KO_s001 + 1.30994 w_k1_MAT_s001
                                 + 1.54735 w_k1_MSFT_s001
                                 + 1.24182 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.11706 w_k1_ADM_s001
                                 + 1.42309 w_k1_AXP_s001
                                 + 1.56202 w_k1_DIS_s001
                                 + 1.25618 w_k1_FDX_s001
                                 + 1.31634 w_k1_IBM_s001
                                 + 1.34515 w_k1_INTC_s001
                                 + 1.34786 w_k1_KO_s001 + 1.01573 w_k1_MAT_s001
                                 + 1.25462 w_k1_MSFT_s001
                                 + 1.26192 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.16781 w_k1_ADM_s001
                                 + 1.09033 w_k1_AXP_s001
                                 + 1.02159 w_k1_DIS_s001
                                 + 1.08192 w_k1_FDX_s001
                                 + 0.867237 w_k1_IBM_s001
                                 + 0.984958 w_k1_INTC_s001
                                 + 0.889591 w_k1_KO_s001
                                 + 0.697501 w_k1_MAT_s001
                                 + 0.978156 w_k1_MSFT_s001
                                 + 1.05798 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.00798 w_k1_ADM_s001
                                 + 0.807195 w_k1_AXP_s001
                                 + 1.09952 w_k1_DIS_s001
                                 + 1.00811 w_k1_FDX_s001
                                 + 0.891918 w_k1_IBM_s001
                                 + 1.16855 w_k1_INTC_s001
                                 + 0.970612 w_k1_KO_s001
                                 + 0.429798 w_k1_MAT_s001
                                 + 1.09667 w_k1_MSFT_s001
                                 + 0.959704 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.944936 w_k1_ADM_s002
                                 + 1.01605 w_k1_AXP_s002
                                 + 0.863776 w_k1_DIS_s002
                                 + 1.04943 w_k1_FDX_s002
                                 + 1.11591 w_k1_IBM_s002
                                 + 0.885635 w_k1_INTC_s002
                                 + 0.913447 w_k1_KO_s002 + 1.0146 w_k1_MAT_s002
                                 + 0.957479 w_k1_MSFT_s002
                                 + 1.0229 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.954751 w_k1_ADM_s002
                                 + 0.943164 w_k1_AXP_s002
                                 + 1.18636 w_k1_DIS_s002
                                 + 0.933353 w_k1_FDX_s002
                                 + 1.51781 w_k1_IBM_s002
                                 + 0.988226 w_k1_INTC_s002
                                 + 1.18334 w_k1_KO_s002 + 1.28991 w_k1_MAT_s002
                                 + 0.93968 w_k1_MSFT_s002 + 1.0865 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.23432 w_k1_ADM_s002
                                 + 1.43204 w_k1_AXP_s002
                                 + 1.36391 w_k1_DIS_s002
                                 + 1.28888 w_k1_FDX_s002
                                 + 1.22602 w_k1_IBM_s002
                                 + 1.32558 w_k1_INTC_s002
                                 + 1.06089 w_k1_KO_s002 + 1.10591 w_k1_MAT_s002
                                 + 1.16521 w_k1_MSFT_s002
                                 + 1.24845 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.670386 w_k1_ADM_s002
                                 + 0.481032 w_k1_AXP_s002
                                 + 0.507099 w_k1_DIS_s002
                                 + 0.696948 w_k1_FDX_s002
                                 + 0.657902 w_k1_IBM_s002
                                 + 0.784852 w_k1_INTC_s002
                                 + 0.562212 w_k1_KO_s002
                                 + 0.464103 w_k1_MAT_s002
                                 + 0.697545 w_k1_MSFT_s002
                                 + 0.737786 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.995733 w_k1_ADM_s002
                                 + 0.833401 w_k1_AXP_s002
                                 + 0.738034 w_k1_DIS_s002
                                 + 0.908415 w_k1_FDX_s002
                                 + 0.541399 w_k1_IBM_s002
                                 + 0.769186 w_k1_INTC_s002
                                 + 0.668331 w_k1_KO_s002
                                 + 0.445816 w_k1_MAT_s002
                                 + 0.799749 w_k1_MSFT_s002
                                 + 0.917833 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.704391 w_k1_ADM_s002
                                 + 0.650901 w_k1_AXP_s002
                                 + 0.609698 w_k1_DIS_s002
                                 + 0.770763 w_k1_FDX_s002
                                 + 0.946257 w_k1_IBM_s002
                                 + 0.862169 w_k1_INTC_s002
                                 + 0.69227 w_k1_KO_s002
                                 + 0.794926 w_k1_MAT_s002
                                 + 0.760041 w_k1_MSFT_s002
                                 + 0.881752 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.04699 w_k1_ADM_s002
                                 + 0.900506 w_k1_AXP_s002
                                 + 0.835908 w_k1_DIS_s002
                                 + 0.935483 w_k1_FDX_s002
                                 + 0.857954 w_k1_IBM_s002
                                 + 1.0485 w_k1_INTC_s002
                                 + 0.872684 w_k1_KO_s002
                                 + 0.852149 w_k1_MAT_s002
                                 + 0.956076 w_k1_MSFT_s002
                                 + 0.97847 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.14889 w_k1_ADM_s002
                                 + 1.08896 w_k1_AXP_s002
                                 + 0.984978 w_k1_DIS_s002
                                 + 1.19321 w_k1_FDX_s002
                                 + 0.789091 w_k1_IBM_s002
                                 + 1.02722 w_k1_INTC_s002
                                 + 0.789089 w_k1_KO_s002
                                 + 0.882011 w_k1_MAT_s002
                                 + 1.04525 w_k1_MSFT_s002
                                 + 1.04844 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.06459 w_k1_ADM_s003
                                 + 1.46465 w_k1_AXP_s003
                                 + 1.44872 w_k1_DIS_s003
                                 + 1.22318 w_k1_FDX_s003
                                 + 1.60042 w_k1_IBM_s003
                                 + 1.24108 w_k1_INTC_s003
                                 + 1.40686 w_k1_KO_s003 + 1.35827 w_k1_MAT_s003
                                 + 1.12918 w_k1_MSFT_s003
                                 + 1.29205 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.772232 w_k1_ADM_s003
                                 + 0.703749 w_k1_AXP_s003
                                 + 0.60644 w_k1_DIS_s003
                                 + 0.759074 w_k1_FDX_s003
                                 + 0.856095 w_k1_IBM_s003
                                 + 0.704331 w_k1_INTC_s003
                                 + 0.797589 w_k1_KO_s003
                                 + 0.756398 w_k1_MAT_s003
                                 + 0.787319 w_k1_MSFT_s003
                                 + 0.913381 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.928719 w_k1_ADM_s003
                                 + 0.937496 w_k1_AXP_s003
                                 + 0.774434 w_k1_DIS_s003
                                 + 1.00083 w_k1_FDX_s003
                                 + 0.771879 w_k1_IBM_s003
                                 + 0.90707 w_k1_INTC_s003
                                 + 0.806953 w_k1_KO_s003
                                 + 1.03933 w_k1_MAT_s003
                                 + 1.05203 w_k1_MSFT_s003
                                 + 1.02468 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.10242 w_k1_ADM_s003
                                 + 1.15896 w_k1_AXP_s003
                                 + 1.47339 w_k1_DIS_s003
                                 + 1.32193 w_k1_FDX_s003
                                 + 0.980237 w_k1_IBM_s003
                                 + 1.13347 w_k1_INTC_s003
                                 + 1.18112 w_k1_KO_s003 + 1.07931 w_k1_MAT_s003
                                 + 1.30951 w_k1_MSFT_s003
                                 + 1.10296 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.593703 w_k1_ADM_s003
                                 + 0.484669 w_k1_AXP_s003
                                 + 0.615173 w_k1_DIS_s003
                                 + 0.715136 w_k1_FDX_s003
                                 + 0.570486 w_k1_IBM_s003
                                 + 0.701882 w_k1_INTC_s003
                                 + 0.758397 w_k1_KO_s003
                                 + 0.558655 w_k1_MAT_s003
                                 + 0.911386 w_k1_MSFT_s003
                                 + 0.854355 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.851952 w_k1_ADM_s003
                                 + 1.13715 w_k1_AXP_s003
                                 + 0.890363 w_k1_DIS_s003
                                 + 0.968157 w_k1_FDX_s003
                                 + 1.10412 w_k1_IBM_s003
                                 + 0.982885 w_k1_INTC_s003
                                 + 0.905728 w_k1_KO_s003
                                 + 1.18746 w_k1_MAT_s003
                                 + 0.913964 w_k1_MSFT_s003
                                 + 1.01263 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.960676 w_k1_ADM_s003
                                 + 0.839631 w_k1_AXP_s003
                                 + 1.10699 w_k1_DIS_s003
                                 + 1.06467 w_k1_FDX_s003
                                 + 1.02674 w_k1_IBM_s003
                                 + 1.0746 w_k1_INTC_s003 + 1.14495 w_k1_KO_s003
                                 + 0.853734 w_k1_MAT_s003
                                 + 1.2124 w_k1_MSFT_s003 + 1.18832 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.722833 w_k1_ADM_s003
                                 + 0.71522 w_k1_AXP_s003
                                 + 0.776415 w_k1_DIS_s003
                                 + 0.817238 w_k1_FDX_s003
                                 + 1.08902 w_k1_IBM_s003
                                 + 0.763797 w_k1_INTC_s003
                                 + 0.964683 w_k1_KO_s003
                                 + 1.09364 w_k1_MAT_s003
                                 + 0.850247 w_k1_MSFT_s003
                                 + 1.00426 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.22435 w_k1_ADM_s004
                                 + 0.858809 w_k1_AXP_s004
                                 + 1.32407 w_k1_DIS_s004
                                 + 1.05277 w_k1_FDX_s004
                                 + 0.904229 w_k1_IBM_s004
                                 + 1.08868 w_k1_INTC_s004
                                 + 1.30862 w_k1_KO_s004
                                 + 0.732582 w_k1_MAT_s004
                                 + 1.19837 w_k1_MSFT_s004
                                 + 0.97002 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.0078 w_k1_ADM_s004
                                 + 1.03181 w_k1_AXP_s004
                                 + 0.669591 w_k1_DIS_s004
                                 + 0.76137 w_k1_FDX_s004 + 1.0639 w_k1_IBM_s004
                                 + 0.805733 w_k1_INTC_s004
                                 + 0.992846 w_k1_KO_s004
                                 + 1.21986 w_k1_MAT_s004
                                 + 0.774391 w_k1_MSFT_s004
                                 + 0.975653 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.24891 w_k1_ADM_s004
                                 + 1.5684 w_k1_AXP_s004 + 1.77079 w_k1_DIS_s004
                                 + 1.47031 w_k1_FDX_s004
                                 + 1.17535 w_k1_IBM_s004
                                 + 1.17541 w_k1_INTC_s004
                                 + 1.51583 w_k1_KO_s004 + 1.37035 w_k1_MAT_s004
                                 + 1.31063 w_k1_MSFT_s004
                                 + 1.09369 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.048 w_k1_ADM_s004
                                 + 1.44513 w_k1_AXP_s004
                                 + 1.29056 w_k1_DIS_s004
                                 + 1.13503 w_k1_FDX_s004
                                 + 1.46445 w_k1_IBM_s004
                                 + 1.23481 w_k1_INTC_s004
                                 + 1.46188 w_k1_KO_s004 + 1.41848 w_k1_MAT_s004
                                 + 1.16109 w_k1_MSFT_s004 + 1.1829 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.00861 w_k1_ADM_s004
                                 + 0.568376 w_k1_AXP_s004
                                 + 0.881763 w_k1_DIS_s004
                                 + 0.839059 w_k1_FDX_s004
                                 + 0.81951 w_k1_IBM_s004
                                 + 0.899205 w_k1_INTC_s004
                                 + 1.07164 w_k1_KO_s004
                                 + 0.493804 w_k1_MAT_s004
                                 + 1.08288 w_k1_MSFT_s004
                                 + 0.905107 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.1427 w_k1_ADM_s004
                                 + 1.06253 w_k1_AXP_s004 + 1.0161 w_k1_DIS_s004
                                 + 0.971725 w_k1_FDX_s004
                                 + 1.12958 w_k1_IBM_s004
                                 + 1.04358 w_k1_INTC_s004
                                 + 1.27156 w_k1_KO_s004 + 1.21509 w_k1_MAT_s004
                                 + 1.0351 w_k1_MSFT_s004
                                 + 0.993251 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.45935 w_k1_ADM_s004
                                 + 1.35061 w_k1_AXP_s004
                                 + 1.19324 w_k1_DIS_s004
                                 + 1.26253 w_k1_FDX_s004
                                 + 0.995066 w_k1_IBM_s004
                                 + 1.21653 w_k1_INTC_s004
                                 + 1.22665 w_k1_KO_s004
                                 + 0.832824 w_k1_MAT_s004
                                 + 1.22422 w_k1_MSFT_s004
                                 + 1.11671 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.692315 w_k1_ADM_s004
                                 + 0.598353 w_k1_AXP_s004
                                 + 0.683327 w_k1_DIS_s004
                                 + 0.70376 w_k1_FDX_s004
                                 + 0.733172 w_k1_IBM_s004
                                 + 0.742309 w_k1_INTC_s004
                                 + 0.884156 w_k1_KO_s004
                                 + 0.858114 w_k1_MAT_s004
                                 + 0.756225 w_k1_MSFT_s004
                                 + 0.800286 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.01596 w_k1_ADM_s005
                                 + 1.54826 w_k1_AXP_s005
                                 + 1.20236 w_k1_DIS_s005
                                 + 1.26021 w_k1_FDX_s005
                                 + 1.33039 w_k1_IBM_s005
                                 + 1.32858 w_k1_INTC_s005
                                 + 1.25404 w_k1_KO_s005 + 1.2123 w_k1_MAT_s005
                                 + 0.988879 w_k1_MSFT_s005
                                 + 1.17056 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.10807 w_k1_ADM_s005
                                 + 1.21571 w_k1_AXP_s005
                                 + 1.03909 w_k1_DIS_s005
                                 + 1.28168 w_k1_FDX_s005
                                 + 0.785873 w_k1_IBM_s005
                                 + 1.04411 w_k1_INTC_s005
                                 + 0.930177 w_k1_KO_s005
                                 + 1.28311 w_k1_MAT_s005
                                 + 1.07933 w_k1_MSFT_s005
                                 + 0.968268 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.961744 w_k1_ADM_s005
                                 + 1.06833 w_k1_AXP_s005
                                 + 0.758613 w_k1_DIS_s005
                                 + 0.931793 w_k1_FDX_s005
                                 + 1.28622 w_k1_IBM_s005
                                 + 1.11441 w_k1_INTC_s005
                                 + 0.916586 w_k1_KO_s005 + 1.4114 w_k1_MAT_s005
                                 + 0.956616 w_k1_MSFT_s005
                                 + 0.957478 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.850478 w_k1_ADM_s005
                                 + 0.82511 w_k1_AXP_s005
                                 + 0.797117 w_k1_DIS_s005
                                 + 0.920285 w_k1_FDX_s005
                                 + 0.792735 w_k1_IBM_s005
                                 + 1.00613 w_k1_INTC_s005
                                 + 0.912734 w_k1_KO_s005
                                 + 0.729108 w_k1_MAT_s005
                                 + 0.831228 w_k1_MSFT_s005
                                 + 0.883911 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.04109 w_k1_ADM_s005
                                 + 1.09485 w_k1_AXP_s005
                                 + 1.39112 w_k1_DIS_s005
                                 + 1.20817 w_k1_FDX_s005
                                 + 1.14756 w_k1_IBM_s005
                                 + 1.38555 w_k1_INTC_s005
                                 + 1.34492 w_k1_KO_s005 + 1.20822 w_k1_MAT_s005
                                 + 1.16778 w_k1_MSFT_s005
                                 + 1.13366 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.18017 w_k1_ADM_s005
                                 + 1.65069 w_k1_AXP_s005
                                 + 1.42094 w_k1_DIS_s005
                                 + 1.33816 w_k1_FDX_s005
                                 + 1.59459 w_k1_IBM_s005
                                 + 1.20859 w_k1_INTC_s005
                                 + 1.24029 w_k1_KO_s005 + 2.23931 w_k1_MAT_s005
                                 + 1.21568 w_k1_MSFT_s005
                                 + 1.03963 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.935035 w_k1_ADM_s005
                                 + 0.764178 w_k1_AXP_s005
                                 + 0.59601 w_k1_DIS_s005
                                 + 0.750398 w_k1_FDX_s005
                                 + 0.791415 w_k1_IBM_s005
                                 + 0.911757 w_k1_INTC_s005
                                 + 0.90784 w_k1_KO_s005
                                 + 0.773195 w_k1_MAT_s005
                                 + 0.761927 w_k1_MSFT_s005
                                 + 0.817152 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.02388 w_k1_ADM_s005
                                 + 1.03786 w_k1_AXP_s005
                                 + 1.02621 w_k1_DIS_s005
                                 + 1.08136 w_k1_FDX_s005
                                 + 1.07468 w_k1_IBM_s005
                                 + 1.02437 w_k1_INTC_s005
                                 + 1.02592 w_k1_KO_s005 + 1.20008 w_k1_MAT_s005
                                 + 0.91171 w_k1_MSFT_s005
                                 + 0.943637 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.14324 w_k1_ADM_s006
                                 + 0.826184 w_k1_AXP_s006
                                 + 1.33519 w_k1_DIS_s006
                                 + 1.04725 w_k1_FDX_s006
                                 + 0.77707 w_k1_IBM_s006 + 1.205 w_k1_INTC_s006
                                 + 1.26914 w_k1_KO_s006
                                 + 0.622695 w_k1_MAT_s006
                                 + 1.24654 w_k1_MSFT_s006
                                 + 1.06902 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.14148 w_k1_ADM_s006
                                 + 1.41573 w_k1_AXP_s006
                                 + 1.41343 w_k1_DIS_s006
                                 + 1.01862 w_k1_FDX_s006
                                 + 1.64689 w_k1_IBM_s006
                                 + 1.1643 w_k1_INTC_s006 + 1.70882 w_k1_KO_s006
                                 + 2.35392 w_k1_MAT_s006
                                 + 1.07544 w_k1_MSFT_s006
                                 + 1.16409 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.958855 w_k1_ADM_s006
                                 + 0.947773 w_k1_AXP_s006
                                 + 0.761025 w_k1_DIS_s006
                                 + 0.718248 w_k1_FDX_s006
                                 + 1.17357 w_k1_IBM_s006
                                 + 0.897393 w_k1_INTC_s006
                                 + 1.05543 w_k1_KO_s006 + 1.27756 w_k1_MAT_s006
                                 + 0.787541 w_k1_MSFT_s006
                                 + 0.946491 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.00847 w_k1_ADM_s006
                                 + 1.12008 w_k1_AXP_s006
                                 + 1.10647 w_k1_DIS_s006
                                 + 1.04253 w_k1_FDX_s006
                                 + 1.00031 w_k1_IBM_s006
                                 + 1.26905 w_k1_INTC_s006
                                 + 1.15353 w_k1_KO_s006
                                 + 0.968086 w_k1_MAT_s006
                                 + 0.969919 w_k1_MSFT_s006
                                 + 0.940366 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.02423 w_k1_ADM_s006
                                 + 0.826206 w_k1_AXP_s006
                                 + 0.962563 w_k1_DIS_s006
                                 + 0.915288 w_k1_FDX_s006
                                 + 0.990168 w_k1_IBM_s006
                                 + 0.960577 w_k1_INTC_s006
                                 + 0.956533 w_k1_KO_s006
                                 + 0.83086 w_k1_MAT_s006
                                 + 1.1191 w_k1_MSFT_s006
                                 + 0.914764 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.774153 w_k1_ADM_s006
                                 + 0.890791 w_k1_AXP_s006
                                 + 0.750839 w_k1_DIS_s006
                                 + 0.855775 w_k1_FDX_s006
                                 + 0.686822 w_k1_IBM_s006
                                 + 0.885923 w_k1_INTC_s006
                                 + 0.834154 w_k1_KO_s006
                                 + 0.78115 w_k1_MAT_s006
                                 + 0.863974 w_k1_MSFT_s006
                                 + 0.943165 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.16395 w_k1_ADM_s006
                                 + 1.57252 w_k1_AXP_s006 + 1.2995 w_k1_DIS_s006
                                 + 1.27121 w_k1_FDX_s006
                                 + 1.01696 w_k1_IBM_s006
                                 + 1.50958 w_k1_INTC_s006
                                 + 1.21916 w_k1_KO_s006 + 1.33603 w_k1_MAT_s006
                                 + 1.4402 w_k1_MSFT_s006 + 1.25791 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.711799 w_k1_ADM_s006
                                 + 0.5927 w_k1_AXP_s006
                                 + 0.681223 w_k1_DIS_s006
                                 + 0.677711 w_k1_FDX_s006
                                 + 0.84077 w_k1_IBM_s006
                                 + 0.750419 w_k1_INTC_s006
                                 + 0.847505 w_k1_KO_s006
                                 + 0.598073 w_k1_MAT_s006
                                 + 0.711627 w_k1_MSFT_s006
                                 + 0.851527 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.2298 w_k1_ADM_s007
                                 + 1.43997 w_k1_AXP_s007
                                 + 1.62844 w_k1_DIS_s007
                                 + 1.36944 w_k1_FDX_s007
                                 + 1.07501 w_k1_IBM_s007
                                 + 1.56277 w_k1_INTC_s007
                                 + 1.13805 w_k1_KO_s007 + 1.05049 w_k1_MAT_s007
                                 + 1.19625 w_k1_MSFT_s007
                                 + 1.09484 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.858318 w_k1_ADM_s007
                                 + 0.775449 w_k1_AXP_s007
                                 + 0.808701 w_k1_DIS_s007
                                 + 0.764747 w_k1_FDX_s007
                                 + 1.40506 w_k1_IBM_s007
                                 + 0.935101 w_k1_INTC_s007
                                 + 1.17474 w_k1_KO_s007 + 1.48225 w_k1_MAT_s007
                                 + 0.865914 w_k1_MSFT_s007
                                 + 1.01752 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.894568 w_k1_ADM_s007
                                 + 0.702078 w_k1_AXP_s007
                                 + 0.817236 w_k1_DIS_s007
                                 + 0.844074 w_k1_FDX_s007
                                 + 0.976623 w_k1_IBM_s007
                                 + 1.01335 w_k1_INTC_s007
                                 + 0.919187 w_k1_KO_s007 + 0.8114 w_k1_MAT_s007
                                 + 0.859627 w_k1_MSFT_s007
                                 + 0.950484 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.30503 w_k1_ADM_s007
                                 + 1.08815 w_k1_AXP_s007
                                 + 1.15758 w_k1_DIS_s007
                                 + 1.03728 w_k1_FDX_s007
                                 + 1.27179 w_k1_IBM_s007
                                 + 1.14438 w_k1_INTC_s007
                                 + 1.18083 w_k1_KO_s007 + 1.12211 w_k1_MAT_s007
                                 + 1.03968 w_k1_MSFT_s007
                                 + 1.08464 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.17787 w_k1_ADM_s007
                                 + 1.15055 w_k1_AXP_s007
                                 + 0.921885 w_k1_DIS_s007
                                 + 1.02205 w_k1_FDX_s007
                                 + 1.00039 w_k1_IBM_s007
                                 + 1.25609 w_k1_INTC_s007
                                 + 0.994496 w_k1_KO_s007
                                 + 1.19762 w_k1_MAT_s007
                                 + 1.01805 w_k1_MSFT_s007
                                 + 1.12277 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.28717 w_k1_ADM_s007
                                 + 1.16786 w_k1_AXP_s007
                                 + 1.14033 w_k1_DIS_s007
                                 + 1.13469 w_k1_FDX_s007
                                 + 0.902253 w_k1_IBM_s007
                                 + 1.17268 w_k1_INTC_s007
                                 + 0.984617 w_k1_KO_s007 + 1.1707 w_k1_MAT_s007
                                 + 1.12389 w_k1_MSFT_s007
                                 + 0.916375 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.31159 w_k1_ADM_s007
                                 + 1.21405 w_k1_AXP_s007
                                 + 1.27416 w_k1_DIS_s007
                                 + 1.05564 w_k1_FDX_s007 + 1.6764 w_k1_IBM_s007
                                 + 1.1535 w_k1_INTC_s007 + 1.34288 w_k1_KO_s007
                                 + 2.1568 w_k1_MAT_s007
                                 + 1.11773 w_k1_MSFT_s007
                                 + 1.07939 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.984013 w_k1_ADM_s007
                                 + 0.83056 w_k1_AXP_s007
                                 + 0.785619 w_k1_DIS_s007
                                 + 0.913281 w_k1_FDX_s007
                                 + 0.72143 w_k1_IBM_s007
                                 + 0.94562 w_k1_INTC_s007
                                 + 0.766194 w_k1_KO_s007
                                 + 0.64053 w_k1_MAT_s007
                                 + 0.863961 w_k1_MSFT_s007
                                 + 0.950536 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.19814 w_k1_ADM_s010
                                 + 1.41661 w_k1_AXP_s010
                                 + 1.36362 w_k1_DIS_s010
                                 + 1.22913 w_k1_FDX_s010 + 1.6721 w_k1_IBM_s010
                                 + 1.32983 w_k1_INTC_s010
                                 + 1.31313 w_k1_KO_s010 + 1.56396 w_k1_MAT_s010
                                 + 1.15684 w_k1_MSFT_s010
                                 + 1.22771 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.758443 w_k1_ADM_s010
                                 + 1.12878 w_k1_AXP_s010
                                 + 1.55512 w_k1_DIS_s010 + 1.122 w_k1_FDX_s010
                                 + 0.997589 w_k1_IBM_s010
                                 + 1.22081 w_k1_INTC_s010
                                 + 1.12714 w_k1_KO_s010
                                 + 0.661167 w_k1_MAT_s010
                                 + 1.1719 w_k1_MSFT_s010 + 1.00681 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.07464 w_k1_ADM_s010
                                 + 1.0491 w_k1_AXP_s010
                                 + 0.841547 w_k1_DIS_s010
                                 + 0.885303 w_k1_FDX_s010
                                 + 1.03779 w_k1_IBM_s010
                                 + 1.06711 w_k1_INTC_s010
                                 + 1.00578 w_k1_KO_s010
                                 + 0.929594 w_k1_MAT_s010
                                 + 0.855499 w_k1_MSFT_s010
                                 + 0.898961 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.869993 w_k1_ADM_s010
                                 + 0.813217 w_k1_AXP_s010
                                 + 0.813694 w_k1_DIS_s010
                                 + 0.783083 w_k1_FDX_s010
                                 + 1.33359 w_k1_IBM_s010
                                 + 0.990373 w_k1_INTC_s010
                                 + 1.17322 w_k1_KO_s010 + 1.03329 w_k1_MAT_s010
                                 + 0.87199 w_k1_MSFT_s010
                                 + 0.906093 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.903733 w_k1_ADM_s010
                                 + 1.4884 w_k1_AXP_s010 + 1.65724 w_k1_DIS_s010
                                 + 1.29728 w_k1_FDX_s010
                                 + 1.21961 w_k1_IBM_s010
                                 + 1.37907 w_k1_INTC_s010
                                 + 1.35275 w_k1_KO_s010 + 1.17982 w_k1_MAT_s010
                                 + 1.15884 w_k1_MSFT_s010
                                 + 1.07096 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.01313 w_k1_ADM_s010
                                 + 0.95846 w_k1_AXP_s010
                                 + 1.04299 w_k1_DIS_s010
                                 + 0.995009 w_k1_FDX_s010
                                 + 0.80383 w_k1_IBM_s010
                                 + 1.1987 w_k1_INTC_s010
                                 + 0.801783 w_k1_KO_s010
                                 + 0.609921 w_k1_MAT_s010
                                 + 1.0591 w_k1_MSFT_s010
                                 + 0.919388 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.2185 w_k1_ADM_s010
                                 + 1.31032 w_k1_AXP_s010
                                 + 1.47408 w_k1_DIS_s010
                                 + 1.29415 w_k1_FDX_s010
                                 + 0.865247 w_k1_IBM_s010
                                 + 1.05415 w_k1_INTC_s010
                                 + 0.961112 w_k1_KO_s010
                                 + 0.823026 w_k1_MAT_s010
                                 + 1.04843 w_k1_MSFT_s010
                                 + 1.00296 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.879901 w_k1_ADM_s010
                                 + 0.881839 w_k1_AXP_s010
                                 + 0.836723 w_k1_DIS_s010
                                 + 0.900852 w_k1_FDX_s010
                                 + 0.73481 w_k1_IBM_s010
                                 + 1.02569 w_k1_INTC_s010
                                 + 0.845752 w_k1_KO_s010
                                 + 0.740507 w_k1_MAT_s010
                                 + 0.942677 w_k1_MSFT_s010
                                 + 0.889539 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.63345 w_k1_ADM_s011
                                 + 1.76132 w_k1_AXP_s011 + 1.4567 w_k1_DIS_s011
                                 + 1.26816 w_k1_FDX_s011
                                 + 1.49628 w_k1_IBM_s011
                                 + 1.20051 w_k1_INTC_s011
                                 + 1.59039 w_k1_KO_s011 + 1.71578 w_k1_MAT_s011
                                 + 1.23901 w_k1_MSFT_s011
                                 + 1.44939 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.23173 w_k1_ADM_s011
                                 + 1.22473 w_k1_AXP_s011
                                 + 1.78574 w_k1_DIS_s011
                                 + 1.22015 w_k1_FDX_s011
                                 + 1.83897 w_k1_IBM_s011
                                 + 1.20049 w_k1_INTC_s011
                                 + 1.71256 w_k1_KO_s011 + 1.82141 w_k1_MAT_s011
                                 + 1.5096 w_k1_MSFT_s011 + 1.4714 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.855281 w_k1_ADM_s011
                                 + 0.83624 w_k1_AXP_s011
                                 + 0.991094 w_k1_DIS_s011
                                 + 0.854102 w_k1_FDX_s011
                                 + 1.42586 w_k1_IBM_s011
                                 + 0.982538 w_k1_INTC_s011
                                 + 1.04482 w_k1_KO_s011
                                 + 0.782297 w_k1_MAT_s011
                                 + 1.04731 w_k1_MSFT_s011
                                 + 0.949835 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.01153 w_k1_ADM_s011
                                 + 1.06662 w_k1_AXP_s011
                                 + 0.879968 w_k1_DIS_s011
                                 + 0.894739 w_k1_FDX_s011
                                 + 1.08886 w_k1_IBM_s011
                                 + 0.860962 w_k1_INTC_s011
                                 + 0.907925 w_k1_KO_s011
                                 + 1.00496 w_k1_MAT_s011
                                 + 1.0169 w_k1_MSFT_s011
                                 + 0.919058 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.758781 w_k1_ADM_s011
                                 + 1.04692 w_k1_AXP_s011
                                 + 1.13531 w_k1_DIS_s011
                                 + 1.03449 w_k1_FDX_s011
                                 + 1.05042 w_k1_IBM_s011
                                 + 0.949635 w_k1_INTC_s011
                                 + 0.92825 w_k1_KO_s011 + 1.11236 w_k1_MAT_s011
                                 + 1.01759 w_k1_MSFT_s011
                                 + 0.935765 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.77017 w_k1_ADM_s011
                                 + 0.889609 w_k1_AXP_s011
                                 + 0.703803 w_k1_DIS_s011
                                 + 0.911171 w_k1_FDX_s011
                                 + 0.729843 w_k1_IBM_s011
                                 + 0.768237 w_k1_INTC_s011
                                 + 0.719317 w_k1_KO_s011
                                 + 0.919593 w_k1_MAT_s011
                                 + 0.855421 w_k1_MSFT_s011
                                 + 0.913703 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.949382 w_k1_ADM_s011
                                 + 1.35343 w_k1_AXP_s011
                                 + 1.46898 w_k1_DIS_s011
                                 + 1.27426 w_k1_FDX_s011
                                 + 0.739433 w_k1_IBM_s011
                                 + 1.02383 w_k1_INTC_s011
                                 + 1.04008 w_k1_KO_s011
                                 + 0.880909 w_k1_MAT_s011
                                 + 1.29591 w_k1_MSFT_s011
                                 + 1.06677 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.01887 w_k1_ADM_s011
                                 + 1.08228 w_k1_AXP_s011
                                 + 0.804957 w_k1_DIS_s011
                                 + 0.799303 w_k1_FDX_s011
                                 + 1.38571 w_k1_IBM_s011
                                 + 0.999058 w_k1_INTC_s011
                                 + 1.06075 w_k1_KO_s011 + 1.38048 w_k1_MAT_s011
                                 + 0.94594 w_k1_MSFT_s011
                                 + 0.891166 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.06812 w_k1_ADM_s012
                                 + 1.09632 w_k1_AXP_s012
                                 + 1.65794 w_k1_DIS_s012 + 1.3389 w_k1_FDX_s012
                                 + 1.08724 w_k1_IBM_s012
                                 + 1.46506 w_k1_INTC_s012
                                 + 1.20918 w_k1_KO_s012
                                 + 0.899829 w_k1_MAT_s012
                                 + 1.50571 w_k1_MSFT_s012
                                 + 1.15269 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.732262 w_k1_ADM_s012
                                 + 0.727515 w_k1_AXP_s012
                                 + 1.13163 w_k1_DIS_s012
                                 + 1.15876 w_k1_FDX_s012
                                 + 0.708844 w_k1_IBM_s012
                                 + 1.10951 w_k1_INTC_s012
                                 + 1.00763 w_k1_KO_s012
                                 + 0.788651 w_k1_MAT_s012
                                 + 1.28617 w_k1_MSFT_s012
                                 + 0.968006 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.871477 w_k1_ADM_s012
                                 + 0.809829 w_k1_AXP_s012
                                 + 1.13963 w_k1_DIS_s012
                                 + 0.929061 w_k1_FDX_s012
                                 + 0.922991 w_k1_IBM_s012
                                 + 0.959937 w_k1_INTC_s012
                                 + 1.04996 w_k1_KO_s012
                                 + 0.687778 w_k1_MAT_s012
                                 + 1.06469 w_k1_MSFT_s012
                                 + 0.910183 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.649232 w_k1_ADM_s012
                                 + 0.795841 w_k1_AXP_s012
                                 + 0.802492 w_k1_DIS_s012
                                 + 0.778004 w_k1_FDX_s012
                                 + 0.929985 w_k1_IBM_s012
                                 + 0.701057 w_k1_INTC_s012
                                 + 0.868108 w_k1_KO_s012
                                 + 0.739871 w_k1_MAT_s012
                                 + 0.84673 w_k1_MSFT_s012
                                 + 0.916365 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.648962 w_k1_ADM_s012
                                 + 0.938184 w_k1_AXP_s012
                                 + 0.843463 w_k1_DIS_s012
                                 + 0.98324 w_k1_FDX_s012
                                 + 0.821743 w_k1_IBM_s012
                                 + 0.894762 w_k1_INTC_s012
                                 + 0.839026 w_k1_KO_s012
                                 + 0.977612 w_k1_MAT_s012
                                 + 0.915743 w_k1_MSFT_s012
                                 + 0.792215 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.14028 w_k1_ADM_s012
                                 + 0.929177 w_k1_AXP_s012
                                 + 1.14076 w_k1_DIS_s012
                                 + 1.10197 w_k1_FDX_s012
                                 + 1.51448 w_k1_IBM_s012
                                 + 1.07608 w_k1_INTC_s012 + 1.3012 w_k1_KO_s012
                                 + 1.27153 w_k1_MAT_s012
                                 + 1.08059 w_k1_MSFT_s012
                                 + 0.977456 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.817225 w_k1_ADM_s012
                                 + 1.018 w_k1_AXP_s012 + 1.02765 w_k1_DIS_s012
                                 + 0.950052 w_k1_FDX_s012
                                 + 1.15372 w_k1_IBM_s012
                                 + 0.853338 w_k1_INTC_s012
                                 + 1.12472 w_k1_KO_s012 + 1.04394 w_k1_MAT_s012
                                 + 0.917334 w_k1_MSFT_s012
                                 + 1.05306 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.988375 w_k1_ADM_s012
                                 + 1.05238 w_k1_AXP_s012 + 1.1309 w_k1_DIS_s012
                                 + 1.21005 w_k1_FDX_s012
                                 + 0.91986 w_k1_IBM_s012
                                 + 1.0325 w_k1_INTC_s012
                                 + 0.823081 w_k1_KO_s012
                                 + 0.70066 w_k1_MAT_s012
                                 + 1.05345 w_k1_MSFT_s012
                                 + 0.844747 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.856499 w_k1_ADM_s013
                                 + 0.768135 w_k1_AXP_s013
                                 + 0.880038 w_k1_DIS_s013
                                 + 0.92931 w_k1_FDX_s013
                                 + 0.774404 w_k1_IBM_s013
                                 + 1.03947 w_k1_INTC_s013
                                 + 0.836937 w_k1_KO_s013
                                 + 0.540249 w_k1_MAT_s013
                                 + 0.983882 w_k1_MSFT_s013
                                 + 0.94556 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.973225 w_k1_ADM_s013
                                 + 1.05498 w_k1_AXP_s013
                                 + 1.13953 w_k1_DIS_s013
                                 + 1.01068 w_k1_FDX_s013
                                 + 1.55074 w_k1_IBM_s013
                                 + 1.22903 w_k1_INTC_s013
                                 + 1.26435 w_k1_KO_s013
                                 + 0.790018 w_k1_MAT_s013
                                 + 0.944278 w_k1_MSFT_s013
                                 + 1.07707 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.18164 w_k1_ADM_s013
                                 + 1.05199 w_k1_AXP_s013
                                 + 1.18077 w_k1_DIS_s013 + 1.3612 w_k1_FDX_s013
                                 + 0.731556 w_k1_IBM_s013
                                 + 1.04827 w_k1_INTC_s013
                                 + 0.953071 w_k1_KO_s013
                                 + 0.541355 w_k1_MAT_s013
                                 + 1.13636 w_k1_MSFT_s013
                                 + 1.04386 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.901934 w_k1_ADM_s013
                                 + 1.06721 w_k1_AXP_s013
                                 + 1.19787 w_k1_DIS_s013 + 1.0219 w_k1_FDX_s013
                                 + 1.85178 w_k1_IBM_s013
                                 + 1.01295 w_k1_INTC_s013
                                 + 1.32697 w_k1_KO_s013 + 1.79341 w_k1_MAT_s013
                                 + 0.955305 w_k1_MSFT_s013
                                 + 0.995656 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.963142 w_k1_ADM_s013
                                 + 0.777858 w_k1_AXP_s013
                                 + 0.616148 w_k1_DIS_s013
                                 + 0.868792 w_k1_FDX_s013
                                 + 0.760453 w_k1_IBM_s013
                                 + 0.865043 w_k1_INTC_s013
                                 + 0.720003 w_k1_KO_s013
                                 + 0.618085 w_k1_MAT_s013
                                 + 0.691465 w_k1_MSFT_s013
                                 + 0.85554 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.996769 w_k1_ADM_s013
                                 + 1.12745 w_k1_AXP_s013
                                 + 1.13784 w_k1_DIS_s013 + 1.0904 w_k1_FDX_s013
                                 + 1.05553 w_k1_IBM_s013
                                 + 1.06414 w_k1_INTC_s013
                                 + 1.16477 w_k1_KO_s013
                                 + 0.857156 w_k1_MAT_s013
                                 + 1.06354 w_k1_MSFT_s013
                                 + 0.948035 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.642456 w_k1_ADM_s013
                                 + 0.570106 w_k1_AXP_s013
                                 + 0.571077 w_k1_DIS_s013
                                 + 0.677911 w_k1_FDX_s013
                                 + 1.01803 w_k1_IBM_s013
                                 + 0.740805 w_k1_INTC_s013
                                 + 0.785997 w_k1_KO_s013
                                 + 0.57068 w_k1_MAT_s013
                                 + 0.664081 w_k1_MSFT_s013
                                 + 0.861523 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.06358 w_k1_ADM_s013
                                 + 1.18729 w_k1_AXP_s013
                                 + 1.28065 w_k1_DIS_s013
                                 + 1.15999 w_k1_FDX_s013
                                 + 1.21069 w_k1_IBM_s013
                                 + 1.09095 w_k1_INTC_s013 + 1.2399 w_k1_KO_s013
                                 + 1.0779 w_k1_MAT_s013
                                 + 1.10534 w_k1_MSFT_s013
                                 + 0.980596 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.901634 w_k1_ADM_s014
                                 + 0.848239 w_k1_AXP_s014
                                 + 0.851258 w_k1_DIS_s014 + 1.04 w_k1_FDX_s014
                                 + 0.640979 w_k1_IBM_s014
                                 + 0.754138 w_k1_INTC_s014
                                 + 0.817201 w_k1_KO_s014
                                 + 0.509128 w_k1_MAT_s014
                                 + 0.843779 w_k1_MSFT_s014
                                 + 0.897829 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.655596 w_k1_ADM_s014
                                 + 0.682333 w_k1_AXP_s014
                                 + 0.697769 w_k1_DIS_s014
                                 + 0.888643 w_k1_FDX_s014
                                 + 0.757279 w_k1_IBM_s014
                                 + 0.883552 w_k1_INTC_s014
                                 + 0.785625 w_k1_KO_s014
                                 + 0.682878 w_k1_MAT_s014
                                 + 0.844496 w_k1_MSFT_s014
                                 + 0.937369 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.895331 w_k1_ADM_s014
                                 + 1.22806 w_k1_AXP_s014 + 1.2281 w_k1_DIS_s014
                                 + 1.06016 w_k1_FDX_s014
                                 + 1.50057 w_k1_IBM_s014
                                 + 1.00936 w_k1_INTC_s014
                                 + 1.40463 w_k1_KO_s014 + 2.01566 w_k1_MAT_s014
                                 + 0.976179 w_k1_MSFT_s014
                                 + 1.07413 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.957672 w_k1_ADM_s014
                                 + 1.13842 w_k1_AXP_s014
                                 + 1.02187 w_k1_DIS_s014
                                 + 0.945381 w_k1_FDX_s014
                                 + 1.13702 w_k1_IBM_s014
                                 + 0.748706 w_k1_INTC_s014
                                 + 1.01857 w_k1_KO_s014 + 1.17707 w_k1_MAT_s014
                                 + 0.822983 w_k1_MSFT_s014
                                 + 0.965404 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.736511 w_k1_ADM_s014
                                 + 0.726224 w_k1_AXP_s014
                                 + 0.872522 w_k1_DIS_s014
                                 + 0.807433 w_k1_FDX_s014
                                 + 1.08592 w_k1_IBM_s014
                                 + 0.724523 w_k1_INTC_s014
                                 + 1.00701 w_k1_KO_s014 + 1.15244 w_k1_MAT_s014
                                 + 0.86043 w_k1_MSFT_s014
                                 + 0.93225 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.10364 w_k1_ADM_s014
                                 + 1.28649 w_k1_AXP_s014
                                 + 1.41127 w_k1_DIS_s014
                                 + 1.40644 w_k1_FDX_s014
                                 + 0.969483 w_k1_IBM_s014
                                 + 1.11207 w_k1_INTC_s014 + 1.335 w_k1_KO_s014
                                 + 1.0072 w_k1_MAT_s014
                                 + 1.21152 w_k1_MSFT_s014
                                 + 1.25043 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.896656 w_k1_ADM_s014
                                 + 1.15265 w_k1_AXP_s014
                                 + 1.10555 w_k1_DIS_s014
                                 + 1.11819 w_k1_FDX_s014
                                 + 0.937982 w_k1_IBM_s014
                                 + 0.911626 w_k1_INTC_s014
                                 + 1.16872 w_k1_KO_s014
                                 + 0.960311 w_k1_MAT_s014
                                 + 0.920999 w_k1_MSFT_s014
                                 + 1.05279 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.563199 w_k1_ADM_s014
                                 + 0.548544 w_k1_AXP_s014
                                 + 0.461434 w_k1_DIS_s014
                                 + 0.582083 w_k1_FDX_s014
                                 + 0.876195 w_k1_IBM_s014
                                 + 0.847624 w_k1_INTC_s014
                                 + 0.76809 w_k1_KO_s014
                                 + 0.599169 w_k1_MAT_s014
                                 + 0.661278 w_k1_MSFT_s014
                                 + 0.699014 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.29923 w_k1_ADM_s015
                                 + 1.08271 w_k1_AXP_s015 + 1.3508 w_k1_DIS_s015
                                 + 1.11689 w_k1_FDX_s015
                                 + 1.13813 w_k1_IBM_s015
                                 + 1.0595 w_k1_INTC_s015 + 1.40267 w_k1_KO_s015
                                 + 0.994612 w_k1_MAT_s015
                                 + 0.981097 w_k1_MSFT_s015
                                 + 1.13198 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.86619 w_k1_ADM_s015
                                 + 1.65576 w_k1_AXP_s015
                                 + 1.42195 w_k1_DIS_s015
                                 + 1.35172 w_k1_FDX_s015
                                 + 1.28513 w_k1_IBM_s015
                                 + 1.3592 w_k1_INTC_s015 + 1.34998 w_k1_KO_s015
                                 + 1.30119 w_k1_MAT_s015
                                 + 1.16707 w_k1_MSFT_s015
                                 + 1.27443 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.06851 w_k1_ADM_s015
                                 + 1.20981 w_k1_AXP_s015
                                 + 1.00146 w_k1_DIS_s015
                                 + 1.03778 w_k1_FDX_s015
                                 + 1.00075 w_k1_IBM_s015
                                 + 0.946289 w_k1_INTC_s015
                                 + 0.944119 w_k1_KO_s015
                                 + 1.04716 w_k1_MAT_s015
                                 + 0.774105 w_k1_MSFT_s015
                                 + 0.981809 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.04075 w_k1_ADM_s015
                                 + 0.633114 w_k1_AXP_s015
                                 + 0.943288 w_k1_DIS_s015
                                 + 0.892249 w_k1_FDX_s015
                                 + 0.872255 w_k1_IBM_s015
                                 + 0.877474 w_k1_INTC_s015
                                 + 0.902363 w_k1_KO_s015
                                 + 0.59329 w_k1_MAT_s015
                                 + 0.895257 w_k1_MSFT_s015
                                 + 0.895515 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.16615 w_k1_ADM_s015
                                 + 1.08375 w_k1_AXP_s015
                                 + 1.12256 w_k1_DIS_s015
                                 + 0.919339 w_k1_FDX_s015 + 1.366 w_k1_IBM_s015
                                 + 0.921469 w_k1_INTC_s015
                                 + 1.11229 w_k1_KO_s015 + 1.34357 w_k1_MAT_s015
                                 + 0.923936 w_k1_MSFT_s015
                                 + 0.975315 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.865038 w_k1_ADM_s015
                                 + 0.706131 w_k1_AXP_s015
                                 + 0.607478 w_k1_DIS_s015
                                 + 0.781363 w_k1_FDX_s015
                                 + 0.777269 w_k1_IBM_s015
                                 + 0.808798 w_k1_INTC_s015
                                 + 0.779029 w_k1_KO_s015
                                 + 0.684852 w_k1_MAT_s015
                                 + 0.725069 w_k1_MSFT_s015
                                 + 0.816418 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.920627 w_k1_ADM_s015
                                 + 0.74413 w_k1_AXP_s015
                                 + 0.702689 w_k1_DIS_s015
                                 + 0.701447 w_k1_FDX_s015
                                 + 1.1514 w_k1_IBM_s015
                                 + 0.897071 w_k1_INTC_s015
                                 + 1.02068 w_k1_KO_s015 + 1.25886 w_k1_MAT_s015
                                 + 0.776502 w_k1_MSFT_s015
                                 + 0.993211 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.53201 w_k1_ADM_s015
                                 + 1.04793 w_k1_AXP_s015
                                 + 0.955882 w_k1_DIS_s015
                                 + 1.04646 w_k1_FDX_s015
                                 + 0.826351 w_k1_IBM_s015
                                 + 0.767927 w_k1_INTC_s015
                                 + 0.87139 w_k1_KO_s015
                                 + 0.775941 w_k1_MAT_s015
                                 + 0.850905 w_k1_MSFT_s015
                                 + 0.97562 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.07105 w_k1_ADM_s016
                                 + 1.04791 w_k1_AXP_s016
                                 + 1.23085 w_k1_DIS_s016
                                 + 0.966286 w_k1_FDX_s016
                                 + 1.34158 w_k1_IBM_s016
                                 + 1.17275 w_k1_INTC_s016
                                 + 1.19074 w_k1_KO_s016
                                 + 0.980832 w_k1_MAT_s016
                                 + 1.03891 w_k1_MSFT_s016
                                 + 1.12149 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.18818 w_k1_ADM_s016
                                 + 1.45806 w_k1_AXP_s016
                                 + 1.44763 w_k1_DIS_s016 + 1.238 w_k1_FDX_s016
                                 + 1.29313 w_k1_IBM_s016
                                 + 1.45173 w_k1_INTC_s016
                                 + 1.22206 w_k1_KO_s016 + 1.15593 w_k1_MAT_s016
                                 + 1.15929 w_k1_MSFT_s016
                                 + 1.14858 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.914985 w_k1_ADM_s016
                                 + 0.750155 w_k1_AXP_s016
                                 + 0.831457 w_k1_DIS_s016
                                 + 0.775554 w_k1_FDX_s016
                                 + 1.01019 w_k1_IBM_s016
                                 + 0.882928 w_k1_INTC_s016
                                 + 0.873259 w_k1_KO_s016
                                 + 0.854661 w_k1_MAT_s016
                                 + 0.808602 w_k1_MSFT_s016
                                 + 0.836486 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.08522 w_k1_ADM_s016
                                 + 1.22223 w_k1_AXP_s016 + 1.2697 w_k1_DIS_s016
                                 + 1.18432 w_k1_FDX_s016
                                 + 0.841858 w_k1_IBM_s016
                                 + 1.24059 w_k1_INTC_s016
                                 + 1.06798 w_k1_KO_s016
                                 + 0.760319 w_k1_MAT_s016
                                 + 1.30552 w_k1_MSFT_s016
                                 + 1.02588 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.896812 w_k1_ADM_s016
                                 + 0.960688 w_k1_AXP_s016
                                 + 1.07398 w_k1_DIS_s016
                                 + 0.938056 w_k1_FDX_s016
                                 + 1.03686 w_k1_IBM_s016
                                 + 1.16012 w_k1_INTC_s016
                                 + 0.976293 w_k1_KO_s016
                                 + 0.769935 w_k1_MAT_s016
                                 + 0.984385 w_k1_MSFT_s016
                                 + 0.950396 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.769655 w_k1_ADM_s016
                                 + 0.727836 w_k1_AXP_s016
                                 + 0.835887 w_k1_DIS_s016
                                 + 0.909389 w_k1_FDX_s016
                                 + 0.693268 w_k1_IBM_s016
                                 + 1.25625 w_k1_INTC_s016
                                 + 0.827552 w_k1_KO_s016
                                 + 0.385075 w_k1_MAT_s016
                                 + 1.01446 w_k1_MSFT_s016
                                 + 0.810754 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.791263 w_k1_ADM_s016
                                 + 0.948083 w_k1_AXP_s016
                                 + 0.748395 w_k1_DIS_s016
                                 + 0.891382 w_k1_FDX_s016
                                 + 0.859858 w_k1_IBM_s016
                                 + 0.851706 w_k1_INTC_s016
                                 + 0.846724 w_k1_KO_s016
                                 + 0.758187 w_k1_MAT_s016
                                 + 0.90928 w_k1_MSFT_s016
                                 + 0.941538 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.876825 w_k1_ADM_s016
                                 + 1.40814 w_k1_AXP_s016 + 0.8417 w_k1_DIS_s016
                                 + 1.07763 w_k1_FDX_s016
                                 + 1.16591 w_k1_IBM_s016
                                 + 1.16922 w_k1_INTC_s016 + 1.0425 w_k1_KO_s016
                                 + 1.26306 w_k1_MAT_s016
                                 + 0.985854 w_k1_MSFT_s016
                                 + 0.985996 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.860141 w_k1_ADM_s017
                                 + 0.840908 w_k1_AXP_s017
                                 + 0.58709 w_k1_DIS_s017
                                 + 0.829389 w_k1_FDX_s017
                                 + 0.846143 w_k1_IBM_s017
                                 + 0.899843 w_k1_INTC_s017
                                 + 0.800197 w_k1_KO_s017
                                 + 0.79718 w_k1_MAT_s017
                                 + 0.985855 w_k1_MSFT_s017
                                 + 0.798571 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.981679 w_k1_ADM_s017
                                 + 0.692242 w_k1_AXP_s017
                                 + 0.767512 w_k1_DIS_s017
                                 + 0.817927 w_k1_FDX_s017
                                 + 1.48958 w_k1_IBM_s017
                                 + 1.0626 w_k1_INTC_s017 + 1.18551 w_k1_KO_s017
                                 + 1.19926 w_k1_MAT_s017
                                 + 1.13756 w_k1_MSFT_s017
                                 + 1.15782 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.04133 w_k1_ADM_s017
                                 + 1.09472 w_k1_AXP_s017
                                 + 1.34592 w_k1_DIS_s017
                                 + 1.27816 w_k1_FDX_s017
                                 + 0.768886 w_k1_IBM_s017
                                 + 1.77619 w_k1_INTC_s017
                                 + 1.03374 w_k1_KO_s017
                                 + 0.601281 w_k1_MAT_s017
                                 + 1.38914 w_k1_MSFT_s017
                                 + 0.913626 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.04943 w_k1_ADM_s017
                                 + 0.768869 w_k1_AXP_s017
                                 + 0.779466 w_k1_DIS_s017
                                 + 0.966049 w_k1_FDX_s017
                                 + 0.684774 w_k1_IBM_s017
                                 + 0.898255 w_k1_INTC_s017
                                 + 0.907088 w_k1_KO_s017
                                 + 0.619621 w_k1_MAT_s017
                                 + 0.991077 w_k1_MSFT_s017
                                 + 0.963412 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.29955 w_k1_ADM_s017
                                 + 1.34383 w_k1_AXP_s017
                                 + 1.44371 w_k1_DIS_s017
                                 + 1.33895 w_k1_FDX_s017
                                 + 1.14031 w_k1_IBM_s017
                                 + 1.0625 w_k1_INTC_s017 + 1.28618 w_k1_KO_s017
                                 + 1.04362 w_k1_MAT_s017
                                 + 1.2828 w_k1_MSFT_s017 + 1.19932 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.2342 w_k1_ADM_s017
                                 + 0.777523 w_k1_AXP_s017
                                 + 0.79538 w_k1_DIS_s017
                                 + 0.88908 w_k1_FDX_s017
                                 + 1.10661 w_k1_IBM_s017
                                 + 0.975152 w_k1_INTC_s017
                                 + 1.03218 w_k1_KO_s017
                                 + 0.949533 w_k1_MAT_s017
                                 + 1.1386 w_k1_MSFT_s017
                                 + 0.950704 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.37602 w_k1_ADM_s017
                                 + 1.2517 w_k1_AXP_s017 + 1.38632 w_k1_DIS_s017
                                 + 1.30174 w_k1_FDX_s017
                                 + 1.60404 w_k1_IBM_s017
                                 + 1.31835 w_k1_INTC_s017
                                 + 1.47998 w_k1_KO_s017 + 1.76686 w_k1_MAT_s017
                                 + 1.33452 w_k1_MSFT_s017
                                 + 1.27612 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.2953 w_k1_ADM_s017
                                 + 0.930594 w_k1_AXP_s017
                                 + 1.09639 w_k1_DIS_s017
                                 + 1.11578 w_k1_FDX_s017
                                 + 0.951683 w_k1_IBM_s017
                                 + 1.35344 w_k1_INTC_s017
                                 + 1.07282 w_k1_KO_s017
                                 + 0.799842 w_k1_MAT_s017
                                 + 1.24827 w_k1_MSFT_s017
                                 + 1.05153 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.97077 w_k1_ADM_s100
                                 + 0.951312 w_k1_AXP_s100
                                 + 1.29713 w_k1_DIS_s100
                                 + 1.07893 w_k1_FDX_s100
                                 + 1.15235 w_k1_IBM_s100
                                 + 0.994332 w_k1_INTC_s100
                                 + 1.0863 w_k1_KO_s100 + 0.8952 w_k1_MAT_s100
                                 + 1.09416 w_k1_MSFT_s100
                                 + 0.96793 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.2724 w_k1_ADM_s100
                                 + 1.37524 w_k1_AXP_s100
                                 + 1.54664 w_k1_DIS_s100
                                 + 1.37483 w_k1_FDX_s100
                                 + 0.994201 w_k1_IBM_s100
                                 + 1.40193 w_k1_INTC_s100
                                 + 1.32763 w_k1_KO_s100 + 1.03636 w_k1_MAT_s100
                                 + 1.47423 w_k1_MSFT_s100
                                 + 1.16501 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.970824 w_k1_ADM_s100
                                 + 1.19358 w_k1_AXP_s100
                                 + 1.51697 w_k1_DIS_s100
                                 + 1.25868 w_k1_FDX_s100
                                 + 1.65992 w_k1_IBM_s100
                                 + 1.09997 w_k1_INTC_s100
                                 + 1.45438 w_k1_KO_s100 + 1.39785 w_k1_MAT_s100
                                 + 1.21706 w_k1_MSFT_s100
                                 + 1.10596 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.813385 w_k1_ADM_s100
                                 + 0.708357 w_k1_AXP_s100
                                 + 0.676324 w_k1_DIS_s100
                                 + 0.734397 w_k1_FDX_s100
                                 + 1.07115 w_k1_IBM_s100
                                 + 0.894921 w_k1_INTC_s100
                                 + 0.82994 w_k1_KO_s100
                                 + 0.747043 w_k1_MAT_s100
                                 + 0.681433 w_k1_MSFT_s100
                                 + 0.862732 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.965011 w_k1_ADM_s100
                                 + 1.00554 w_k1_AXP_s100 + 1.036 w_k1_DIS_s100
                                 + 1.16824 w_k1_FDX_s100
                                 + 0.655249 w_k1_IBM_s100
                                 + 0.980787 w_k1_INTC_s100
                                 + 0.906427 w_k1_KO_s100
                                 + 0.800269 w_k1_MAT_s100
                                 + 1.10019 w_k1_MSFT_s100
                                 + 0.977184 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.08236 w_k1_ADM_s100
                                 + 1.1333 w_k1_AXP_s100 + 1.05231 w_k1_DIS_s100
                                 + 1.09727 w_k1_FDX_s100
                                 + 1.14583 w_k1_IBM_s100
                                 + 1.03091 w_k1_INTC_s100
                                 + 1.02093 w_k1_KO_s100
                                 + 0.957276 w_k1_MAT_s100
                                 + 0.997741 w_k1_MSFT_s100
                                 + 1.01868 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.12318 w_k1_ADM_s100
                                 + 0.937313 w_k1_AXP_s100
                                 + 1.04692 w_k1_DIS_s100
                                 + 0.951007 w_k1_FDX_s100
                                 + 1.32305 w_k1_IBM_s100
                                 + 0.91259 w_k1_INTC_s100
                                 + 1.16716 w_k1_KO_s100 + 1.01726 w_k1_MAT_s100
                                 + 0.965616 w_k1_MSFT_s100
                                 + 1.13575 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.922667 w_k1_ADM_s100
                                 + 0.762725 w_k1_AXP_s100
                                 + 0.821462 w_k1_DIS_s100
                                 + 0.921644 w_k1_FDX_s100
                                 + 0.853542 w_k1_IBM_s100
                                 + 0.802635 w_k1_INTC_s100
                                 + 0.950338 w_k1_KO_s100
                                 + 0.898144 w_k1_MAT_s100
                                 + 0.879329 w_k1_MSFT_s100
                                 + 0.947747 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.16143 w_k1_ADM_s101
                                 + 1.10061 w_k1_AXP_s101
                                 + 0.851949 w_k1_DIS_s101
                                 + 0.962139 w_k1_FDX_s101
                                 + 1.16393 w_k1_IBM_s101
                                 + 0.944215 w_k1_INTC_s101
                                 + 1.05424 w_k1_KO_s101
                                 + 0.986421 w_k1_MAT_s101
                                 + 0.871798 w_k1_MSFT_s101
                                 + 1.07955 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.22944 w_k1_ADM_s101
                                 + 1.26309 w_k1_AXP_s101
                                 + 1.29176 w_k1_DIS_s101
                                 + 1.10477 w_k1_FDX_s101
                                 + 2.06777 w_k1_IBM_s101
                                 + 1.35111 w_k1_INTC_s101
                                 + 1.54555 w_k1_KO_s101 + 1.82708 w_k1_MAT_s101
                                 + 1.26563 w_k1_MSFT_s101
                                 + 1.38903 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.06624 w_k1_ADM_s101
                                 + 0.867846 w_k1_AXP_s101
                                 + 0.927275 w_k1_DIS_s101
                                 + 1.06143 w_k1_FDX_s101
                                 + 0.710797 w_k1_IBM_s101
                                 + 1.10431 w_k1_INTC_s101
                                 + 0.911601 w_k1_KO_s101
                                 + 0.838713 w_k1_MAT_s101
                                 + 1.10819 w_k1_MSFT_s101
                                 + 1.05315 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.24866 w_k1_ADM_s101
                                 + 1.29684 w_k1_AXP_s101
                                 + 1.96347 w_k1_DIS_s101
                                 + 1.36752 w_k1_FDX_s101
                                 + 1.97891 w_k1_IBM_s101
                                 + 1.49682 w_k1_INTC_s101 + 1.8486 w_k1_KO_s101
                                 + 1.19317 w_k1_MAT_s101
                                 + 1.63827 w_k1_MSFT_s101
                                 + 1.57657 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.90821 w_k1_ADM_s101
                                 + 0.947481 w_k1_AXP_s101
                                 + 1.28922 w_k1_DIS_s101
                                 + 1.06206 w_k1_FDX_s101
                                 + 1.25458 w_k1_IBM_s101
                                 + 0.981605 w_k1_INTC_s101
                                 + 1.29926 w_k1_KO_s101
                                 + 0.945558 w_k1_MAT_s101
                                 + 1.13394 w_k1_MSFT_s101
                                 + 1.14873 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.34385 w_k1_ADM_s101
                                 + 1.30908 w_k1_AXP_s101
                                 + 1.33268 w_k1_DIS_s101
                                 + 1.26286 w_k1_FDX_s101
                                 + 1.13912 w_k1_IBM_s101
                                 + 1.41266 w_k1_INTC_s101
                                 + 1.22048 w_k1_KO_s101
                                 + 0.817591 w_k1_MAT_s101
                                 + 1.22971 w_k1_MSFT_s101
                                 + 1.20555 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.737491 w_k1_ADM_s101
                                 + 0.706068 w_k1_AXP_s101
                                 + 1.01147 w_k1_DIS_s101
                                 + 0.847254 w_k1_FDX_s101
                                 + 1.20279 w_k1_IBM_s101
                                 + 1.11983 w_k1_INTC_s101
                                 + 1.05252 w_k1_KO_s101
                                 + 0.632285 w_k1_MAT_s101
                                 + 0.925406 w_k1_MSFT_s101
                                 + 0.927424 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.888642 w_k1_ADM_s101
                                 + 0.643546 w_k1_AXP_s101
                                 + 0.652349 w_k1_DIS_s101
                                 + 0.772767 w_k1_FDX_s101
                                 + 0.841412 w_k1_IBM_s101
                                 + 0.792725 w_k1_INTC_s101
                                 + 0.683979 w_k1_KO_s101
                                 + 0.469617 w_k1_MAT_s101
                                 + 0.723625 w_k1_MSFT_s101
                                 + 0.839841 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.02294 w_k1_ADM_s102
                                 + 0.792971 w_k1_AXP_s102
                                 + 0.773717 w_k1_DIS_s102
                                 + 0.815087 w_k1_FDX_s102
                                 + 1.21792 w_k1_IBM_s102
                                 + 0.664059 w_k1_INTC_s102
                                 + 1.0498 w_k1_KO_s102 + 1.36621 w_k1_MAT_s102
                                 + 0.79515 w_k1_MSFT_s102
                                 + 1.02896 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.05699 w_k1_ADM_s102
                                 + 1.23841 w_k1_AXP_s102
                                 + 1.19464 w_k1_DIS_s102
                                 + 0.974891 w_k1_FDX_s102
                                 + 1.77674 w_k1_IBM_s102
                                 + 0.927087 w_k1_INTC_s102
                                 + 1.48274 w_k1_KO_s102 + 1.71494 w_k1_MAT_s102
                                 + 1.0399 w_k1_MSFT_s102 + 1.1295 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.07447 w_k1_ADM_s102
                                 + 0.877846 w_k1_AXP_s102
                                 + 1.30683 w_k1_DIS_s102
                                 + 1.12521 w_k1_FDX_s102
                                 + 0.849769 w_k1_IBM_s102
                                 + 1.05235 w_k1_INTC_s102
                                 + 1.13901 w_k1_KO_s102
                                 + 0.626727 w_k1_MAT_s102
                                 + 1.22915 w_k1_MSFT_s102
                                 + 1.05075 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.20965 w_k1_ADM_s102
                                 + 1.05719 w_k1_AXP_s102
                                 + 0.895004 w_k1_DIS_s102
                                 + 0.984836 w_k1_FDX_s102
                                 + 0.884349 w_k1_IBM_s102
                                 + 1.0086 w_k1_INTC_s102
                                 + 0.928657 w_k1_KO_s102
                                 + 0.802261 w_k1_MAT_s102
                                 + 1.03874 w_k1_MSFT_s102
                                 + 0.948035 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.23835 w_k1_ADM_s102
                                 + 1.29719 w_k1_AXP_s102
                                 + 1.76227 w_k1_DIS_s102
                                 + 1.26298 w_k1_FDX_s102
                                 + 1.38896 w_k1_IBM_s102
                                 + 1.01947 w_k1_INTC_s102
                                 + 1.42948 w_k1_KO_s102 + 1.4238 w_k1_MAT_s102
                                 + 1.18953 w_k1_MSFT_s102
                                 + 1.16323 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.816682 w_k1_ADM_s102
                                 + 0.712604 w_k1_AXP_s102
                                 + 0.774047 w_k1_DIS_s102
                                 + 0.760613 w_k1_FDX_s102
                                 + 0.668008 w_k1_IBM_s102
                                 + 0.790079 w_k1_INTC_s102
                                 + 0.870447 w_k1_KO_s102
                                 + 0.711305 w_k1_MAT_s102
                                 + 0.843698 w_k1_MSFT_s102
                                 + 0.856849 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.935102 w_k1_ADM_s102
                                 + 1.0414 w_k1_AXP_s102 + 1.21606 w_k1_DIS_s102
                                 + 1.10908 w_k1_FDX_s102
                                 + 0.964783 w_k1_IBM_s102
                                 + 0.915132 w_k1_INTC_s102
                                 + 1.27476 w_k1_KO_s102 + 1.12032 w_k1_MAT_s102
                                 + 1.16121 w_k1_MSFT_s102
                                 + 1.06255 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.836426 w_k1_ADM_s102
                                 + 0.623833 w_k1_AXP_s102
                                 + 0.935181 w_k1_DIS_s102
                                 + 0.791755 w_k1_FDX_s102
                                 + 0.98516 w_k1_IBM_s102
                                 + 0.807283 w_k1_INTC_s102
                                 + 1.15167 w_k1_KO_s102
                                 + 0.777296 w_k1_MAT_s102
                                 + 0.977799 w_k1_MSFT_s102
                                 + 1.01063 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.824102 w_k1_ADM_s103
                                 + 0.970185 w_k1_AXP_s103
                                 + 0.835621 w_k1_DIS_s103
                                 + 0.947077 w_k1_FDX_s103
                                 + 0.972406 w_k1_IBM_s103
                                 + 0.836402 w_k1_INTC_s103
                                 + 0.953113 w_k1_KO_s103 + 1.0578 w_k1_MAT_s103
                                 + 0.769816 w_k1_MSFT_s103
                                 + 0.964862 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.618624 w_k1_ADM_s103
                                 + 0.730758 w_k1_AXP_s103
                                 + 0.632211 w_k1_DIS_s103
                                 + 0.780996 w_k1_FDX_s103
                                 + 0.982268 w_k1_IBM_s103
                                 + 0.767196 w_k1_INTC_s103
                                 + 0.846883 w_k1_KO_s103
                                 + 1.10527 w_k1_MAT_s103
                                 + 0.795456 w_k1_MSFT_s103
                                 + 0.847601 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.867636 w_k1_ADM_s103
                                 + 1.10142 w_k1_AXP_s103
                                 + 1.01718 w_k1_DIS_s103
                                 + 0.960291 w_k1_FDX_s103
                                 + 1.26105 w_k1_IBM_s103
                                 + 0.961175 w_k1_INTC_s103
                                 + 1.05131 w_k1_KO_s103 + 1.18438 w_k1_MAT_s103
                                 + 0.895355 w_k1_MSFT_s103
                                 + 1.0681 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.891751 w_k1_ADM_s103
                                 + 1.09356 w_k1_AXP_s103 + 1.0236 w_k1_DIS_s103
                                 + 1.0709 w_k1_FDX_s103
                                 + 0.898155 w_k1_IBM_s103
                                 + 1.05754 w_k1_INTC_s103
                                 + 0.967292 w_k1_KO_s103
                                 + 0.795956 w_k1_MAT_s103
                                 + 0.879609 w_k1_MSFT_s103
                                 + 0.999721 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.773662 w_k1_ADM_s103
                                 + 0.790375 w_k1_AXP_s103
                                 + 0.77275 w_k1_DIS_s103
                                 + 0.871733 w_k1_FDX_s103
                                 + 0.790417 w_k1_IBM_s103
                                 + 0.802482 w_k1_INTC_s103
                                 + 0.894403 w_k1_KO_s103
                                 + 0.805015 w_k1_MAT_s103
                                 + 0.936531 w_k1_MSFT_s103
                                 + 0.932424 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.10359 w_k1_ADM_s103
                                 + 1.04142 w_k1_AXP_s103
                                 + 1.01963 w_k1_DIS_s103
                                 + 1.18718 w_k1_FDX_s103
                                 + 0.644795 w_k1_IBM_s103
                                 + 1.13999 w_k1_INTC_s103
                                 + 0.896819 w_k1_KO_s103
                                 + 0.825036 w_k1_MAT_s103
                                 + 1.04634 w_k1_MSFT_s103
                                 + 0.968144 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.879117 w_k1_ADM_s103
                                 + 0.94403 w_k1_AXP_s103
                                 + 0.696641 w_k1_DIS_s103
                                 + 0.859118 w_k1_FDX_s103
                                 + 0.616319 w_k1_IBM_s103
                                 + 0.930557 w_k1_INTC_s103
                                 + 0.664161 w_k1_KO_s103
                                 + 0.765509 w_k1_MAT_s103
                                 + 0.805852 w_k1_MSFT_s103
                                 + 0.770856 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.32059 w_k1_ADM_s103
                                 + 1.37606 w_k1_AXP_s103
                                 + 1.35871 w_k1_DIS_s103
                                 + 1.27984 w_k1_FDX_s103
                                 + 1.35867 w_k1_IBM_s103
                                 + 1.17751 w_k1_INTC_s103
                                 + 1.45067 w_k1_KO_s103 + 1.63848 w_k1_MAT_s103
                                 + 1.11515 w_k1_MSFT_s103
                                 + 1.26151 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.858449 w_k1_ADM_s104
                                 + 0.684177 w_k1_AXP_s104
                                 + 0.57282 w_k1_DIS_s104
                                 + 0.812557 w_k1_FDX_s104
                                 + 0.567511 w_k1_IBM_s104
                                 + 0.797158 w_k1_INTC_s104
                                 + 0.753126 w_k1_KO_s104
                                 + 0.460375 w_k1_MAT_s104
                                 + 0.804407 w_k1_MSFT_s104
                                 + 0.758037 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.998583 w_k1_ADM_s104
                                 + 1.06525 w_k1_AXP_s104
                                 + 0.879096 w_k1_DIS_s104
                                 + 0.855953 w_k1_FDX_s104
                                 + 1.20997 w_k1_IBM_s104
                                 + 0.833325 w_k1_INTC_s104
                                 + 1.23333 w_k1_KO_s104 + 1.49753 w_k1_MAT_s104
                                 + 0.980475 w_k1_MSFT_s104
                                 + 1.01519 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.06873 w_k1_ADM_s104
                                 + 0.847629 w_k1_AXP_s104
                                 + 0.863995 w_k1_DIS_s104
                                 + 1.06308 w_k1_FDX_s104
                                 + 0.751638 w_k1_IBM_s104
                                 + 0.942113 w_k1_INTC_s104
                                 + 0.981671 w_k1_KO_s104
                                 + 0.729087 w_k1_MAT_s104
                                 + 1.04729 w_k1_MSFT_s104
                                 + 0.950975 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.47335 w_k1_ADM_s104
                                 + 1.5077 w_k1_AXP_s104 + 1.32577 w_k1_DIS_s104
                                 + 1.1778 w_k1_FDX_s104 + 1.61581 w_k1_IBM_s104
                                 + 0.96828 w_k1_INTC_s104
                                 + 1.67725 w_k1_KO_s104 + 1.63859 w_k1_MAT_s104
                                 + 1.19323 w_k1_MSFT_s104
                                 + 1.20225 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.11625 w_k1_ADM_s104
                                 + 0.995445 w_k1_AXP_s104
                                 + 1.31429 w_k1_DIS_s104
                                 + 1.01696 w_k1_FDX_s104
                                 + 1.43989 w_k1_IBM_s104
                                 + 1.0693 w_k1_INTC_s104 + 1.51961 w_k1_KO_s104
                                 + 1.14148 w_k1_MAT_s104
                                 + 1.16818 w_k1_MSFT_s104
                                 + 1.16169 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.09286 w_k1_ADM_s104
                                 + 0.977691 w_k1_AXP_s104
                                 + 1.02249 w_k1_DIS_s104
                                 + 1.00039 w_k1_FDX_s104
                                 + 0.968852 w_k1_IBM_s104
                                 + 1.13666 w_k1_INTC_s104 + 1.2704 w_k1_KO_s104
                                 + 0.890509 w_k1_MAT_s104
                                 + 1.08888 w_k1_MSFT_s104
                                 + 0.999285 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.852642 w_k1_ADM_s104
                                 + 0.699435 w_k1_AXP_s104
                                 + 0.698875 w_k1_DIS_s104
                                 + 0.765016 w_k1_FDX_s104
                                 + 0.858526 w_k1_IBM_s104
                                 + 0.821092 w_k1_INTC_s104
                                 + 0.991077 w_k1_KO_s104
                                 + 0.730173 w_k1_MAT_s104
                                 + 0.901141 w_k1_MSFT_s104
                                 + 0.902668 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.23887 w_k1_ADM_s104
                                 + 1.2592 w_k1_AXP_s104 + 1.3395 w_k1_DIS_s104
                                 + 1.24021 w_k1_FDX_s104
                                 + 0.944997 w_k1_IBM_s104
                                 + 1.10588 w_k1_INTC_s104
                                 + 1.17735 w_k1_KO_s104 + 1.23365 w_k1_MAT_s104
                                 + 1.21296 w_k1_MSFT_s104
                                 + 1.14779 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.23455 w_k1_ADM_s105
                                 + 1.42336 w_k1_AXP_s105
                                 + 1.23159 w_k1_DIS_s105
                                 + 1.21981 w_k1_FDX_s105
                                 + 1.41335 w_k1_IBM_s105
                                 + 1.19618 w_k1_INTC_s105
                                 + 1.28889 w_k1_KO_s105 + 1.47811 w_k1_MAT_s105
                                 + 1.23503 w_k1_MSFT_s105
                                 + 1.11199 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.960068 w_k1_ADM_s105
                                 + 0.935837 w_k1_AXP_s105
                                 + 0.877694 w_k1_DIS_s105
                                 + 0.969081 w_k1_FDX_s105
                                 + 0.950689 w_k1_IBM_s105
                                 + 1.07469 w_k1_INTC_s105
                                 + 0.930701 w_k1_KO_s105
                                 + 0.94319 w_k1_MAT_s105
                                 + 0.946157 w_k1_MSFT_s105
                                 + 0.919583 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.818964 w_k1_ADM_s105
                                 + 0.702278 w_k1_AXP_s105
                                 + 0.77737 w_k1_DIS_s105
                                 + 0.793611 w_k1_FDX_s105
                                 + 1.13689 w_k1_IBM_s105
                                 + 0.849559 w_k1_INTC_s105
                                 + 0.914414 w_k1_KO_s105
                                 + 0.971319 w_k1_MAT_s105
                                 + 0.804944 w_k1_MSFT_s105
                                 + 0.854849 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.09664 w_k1_ADM_s105
                                 + 0.832124 w_k1_AXP_s105
                                 + 1.21447 w_k1_DIS_s105
                                 + 1.10527 w_k1_FDX_s105
                                 + 0.963791 w_k1_IBM_s105
                                 + 1.13964 w_k1_INTC_s105
                                 + 1.12238 w_k1_KO_s105
                                 + 0.954718 w_k1_MAT_s105
                                 + 1.20075 w_k1_MSFT_s105 + 1.0793 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.24897 w_k1_ADM_s105
                                 + 1.52324 w_k1_AXP_s105
                                 + 1.64784 w_k1_DIS_s105
                                 + 1.35061 w_k1_FDX_s105
                                 + 1.79786 w_k1_IBM_s105
                                 + 1.49249 w_k1_INTC_s105
                                 + 1.43848 w_k1_KO_s105 + 2.27924 w_k1_MAT_s105
                                 + 1.30645 w_k1_MSFT_s105
                                 + 1.24918 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.06395 w_k1_ADM_s105
                                 + 1.38808 w_k1_AXP_s105 + 1.2549 w_k1_DIS_s105
                                 + 1.3277 w_k1_FDX_s105 + 0.8724 w_k1_IBM_s105
                                 + 1.36878 w_k1_INTC_s105
                                 + 0.973035 w_k1_KO_s105
                                 + 1.24069 w_k1_MAT_s105
                                 + 1.26328 w_k1_MSFT_s105
                                 + 1.08698 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.01743 w_k1_ADM_s105
                                 + 0.809238 w_k1_AXP_s105
                                 + 0.71689 w_k1_DIS_s105
                                 + 0.90293 w_k1_FDX_s105
                                 + 0.647732 w_k1_IBM_s105
                                 + 0.88738 w_k1_INTC_s105
                                 + 0.781578 w_k1_KO_s105
                                 + 0.617554 w_k1_MAT_s105
                                 + 0.840418 w_k1_MSFT_s105
                                 + 0.905995 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.45182 w_k1_ADM_s105
                                 + 1.14667 w_k1_AXP_s105
                                 + 1.31423 w_k1_DIS_s105
                                 + 1.23061 w_k1_FDX_s105 + 1.0792 w_k1_IBM_s105
                                 + 1.09125 w_k1_INTC_s105
                                 + 1.06371 w_k1_KO_s105 + 1.14568 w_k1_MAT_s105
                                 + 1.19503 w_k1_MSFT_s105
                                 + 1.12572 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.25841 w_k1_ADM_s106
                                 + 1.46341 w_k1_AXP_s106
                                 + 1.64254 w_k1_DIS_s106
                                 + 1.38723 w_k1_FDX_s106
                                 + 1.24283 w_k1_IBM_s106
                                 + 1.1507 w_k1_INTC_s106 + 1.2125 w_k1_KO_s106
                                 + 1.02705 w_k1_MAT_s106
                                 + 1.08939 w_k1_MSFT_s106
                                 + 1.30465 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.891696 w_k1_ADM_s106
                                 + 0.921112 w_k1_AXP_s106
                                 + 1.24827 w_k1_DIS_s106
                                 + 1.02831 w_k1_FDX_s106
                                 + 0.91616 w_k1_IBM_s106
                                 + 0.95144 w_k1_INTC_s106
                                 + 0.966999 w_k1_KO_s106 + 0.9121 w_k1_MAT_s106
                                 + 1.00959 w_k1_MSFT_s106
                                 + 1.04345 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.09146 w_k1_ADM_s106
                                 + 1.2182 w_k1_AXP_s106 + 1.15015 w_k1_DIS_s106
                                 + 1.05098 w_k1_FDX_s106
                                 + 1.13542 w_k1_IBM_s106
                                 + 0.916718 w_k1_INTC_s106
                                 + 1.09414 w_k1_KO_s106 + 1.14994 w_k1_MAT_s106
                                 + 1.09927 w_k1_MSFT_s106
                                 + 1.08432 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.15351 w_k1_ADM_s106
                                 + 1.11451 w_k1_AXP_s106 + 1.3107 w_k1_DIS_s106
                                 + 1.27733 w_k1_FDX_s106
                                 + 0.815806 w_k1_IBM_s106
                                 + 1.05463 w_k1_INTC_s106
                                 + 0.991046 w_k1_KO_s106
                                 + 1.03883 w_k1_MAT_s106
                                 + 1.05706 w_k1_MSFT_s106
                                 + 1.11072 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.05119 w_k1_ADM_s106
                                 + 1.21247 w_k1_AXP_s106
                                 + 0.998989 w_k1_DIS_s106
                                 + 1.02738 w_k1_FDX_s106
                                 + 1.43159 w_k1_IBM_s106
                                 + 1.09243 w_k1_INTC_s106
                                 + 1.14409 w_k1_KO_s106 + 1.3426 w_k1_MAT_s106
                                 + 0.944537 w_k1_MSFT_s106
                                 + 1.07166 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.780638 w_k1_ADM_s106
                                 + 0.91909 w_k1_AXP_s106
                                 + 0.722317 w_k1_DIS_s106
                                 + 0.829568 w_k1_FDX_s106
                                 + 0.797921 w_k1_IBM_s106
                                 + 0.707416 w_k1_INTC_s106
                                 + 0.681544 w_k1_KO_s106
                                 + 0.943036 w_k1_MAT_s106
                                 + 0.763658 w_k1_MSFT_s106
                                 + 0.852864 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.636648 w_k1_ADM_s106
                                 + 0.812227 w_k1_AXP_s106
                                 + 0.866453 w_k1_DIS_s106
                                 + 0.820497 w_k1_FDX_s106
                                 + 1.21983 w_k1_IBM_s106
                                 + 0.781696 w_k1_INTC_s106
                                 + 0.954804 w_k1_KO_s106
                                 + 1.03388 w_k1_MAT_s106
                                 + 0.844016 w_k1_MSFT_s106
                                 + 1.0309 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.46101 w_k1_ADM_s106
                                 + 1.5465 w_k1_AXP_s106 + 1.86808 w_k1_DIS_s106
                                 + 1.2312 w_k1_FDX_s106 + 1.71109 w_k1_IBM_s106
                                 + 0.943387 w_k1_INTC_s106
                                 + 1.40993 w_k1_KO_s106 + 1.95241 w_k1_MAT_s106
                                 + 1.18736 w_k1_MSFT_s106
                                 + 1.28178 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.862622 w_k1_ADM_s107
                                 + 0.700831 w_k1_AXP_s107
                                 + 0.635106 w_k1_DIS_s107
                                 + 0.792208 w_k1_FDX_s107
                                 + 0.569819 w_k1_IBM_s107
                                 + 0.878054 w_k1_INTC_s107
                                 + 0.558789 w_k1_KO_s107
                                 + 0.510415 w_k1_MAT_s107
                                 + 0.760933 w_k1_MSFT_s107
                                 + 0.702859 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.80435 w_k1_ADM_s107
                                 + 0.790126 w_k1_AXP_s107
                                 + 0.857329 w_k1_DIS_s107
                                 + 1.03495 w_k1_FDX_s107
                                 + 0.660687 w_k1_IBM_s107
                                 + 0.92633 w_k1_INTC_s107
                                 + 0.74709 w_k1_KO_s107
                                 + 0.641551 w_k1_MAT_s107
                                 + 0.905533 w_k1_MSFT_s107
                                 + 0.919112 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.21469 w_k1_ADM_s107
                                 + 1.29477 w_k1_AXP_s107 + 1.4784 w_k1_DIS_s107
                                 + 1.2116 w_k1_FDX_s107 + 1.55857 w_k1_IBM_s107
                                 + 1.18957 w_k1_INTC_s107
                                 + 1.22733 w_k1_KO_s107 + 1.27579 w_k1_MAT_s107
                                 + 1.2201 w_k1_MSFT_s107 + 1.12886 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.23272 w_k1_ADM_s107
                                 + 1.24327 w_k1_AXP_s107
                                 + 1.35108 w_k1_DIS_s107
                                 + 1.37742 w_k1_FDX_s107
                                 + 0.965234 w_k1_IBM_s107
                                 + 1.37993 w_k1_INTC_s107
                                 + 1.00333 w_k1_KO_s107 + 0.90018 w_k1_MAT_s107
                                 + 1.25486 w_k1_MSFT_s107
                                 + 1.09098 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.839237 w_k1_ADM_s107
                                 + 0.888431 w_k1_AXP_s107
                                 + 0.994513 w_k1_DIS_s107
                                 + 0.915629 w_k1_FDX_s107
                                 + 1.34523 w_k1_IBM_s107
                                 + 1.17261 w_k1_INTC_s107 + 1.1744 w_k1_KO_s107
                                 + 1.07273 w_k1_MAT_s107
                                 + 1.01284 w_k1_MSFT_s107
                                 + 0.973508 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.846384 w_k1_ADM_s107
                                 + 0.651623 w_k1_AXP_s107
                                 + 0.773097 w_k1_DIS_s107
                                 + 0.793104 w_k1_FDX_s107
                                 + 0.810695 w_k1_IBM_s107
                                 + 0.939357 w_k1_INTC_s107
                                 + 0.833058 w_k1_KO_s107
                                 + 0.612047 w_k1_MAT_s107
                                 + 0.945593 w_k1_MSFT_s107
                                 + 0.82336 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.01984 w_k1_ADM_s107
                                 + 0.969332 w_k1_AXP_s107
                                 + 0.988836 w_k1_DIS_s107
                                 + 1.00277 w_k1_FDX_s107
                                 + 0.979285 w_k1_IBM_s107
                                 + 0.930953 w_k1_INTC_s107
                                 + 0.968383 w_k1_KO_s107
                                 + 0.827719 w_k1_MAT_s107 + w_k1_MSFT_s107
                                 + 0.996461 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.985707 w_k1_ADM_s107
                                 + 1.29049 w_k1_AXP_s107
                                 + 1.17242 w_k1_DIS_s107
                                 + 1.07003 w_k1_FDX_s107
                                 + 1.14887 w_k1_IBM_s107
                                 + 1.1203 w_k1_INTC_s107
                                 + 0.998556 w_k1_KO_s107
                                 + 1.31247 w_k1_MAT_s107
                                 + 1.04035 w_k1_MSFT_s107
                                 + 1.04987 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.691511 w_k1_ADM_s110
                                 + 1.01055 w_k1_AXP_s110
                                 + 0.748829 w_k1_DIS_s110
                                 + 0.824566 w_k1_FDX_s110
                                 + 0.822022 w_k1_IBM_s110
                                 + 0.861522 w_k1_INTC_s110
                                 + 0.812846 w_k1_KO_s110
                                 + 0.835795 w_k1_MAT_s110
                                 + 0.845375 w_k1_MSFT_s110
                                 + 0.887811 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.646523 w_k1_ADM_s110
                                 + 0.668697 w_k1_AXP_s110
                                 + 0.663492 w_k1_DIS_s110
                                 + 0.741841 w_k1_FDX_s110
                                 + 0.703228 w_k1_IBM_s110
                                 + 0.701874 w_k1_INTC_s110
                                 + 0.644703 w_k1_KO_s110
                                 + 0.673979 w_k1_MAT_s110
                                 + 0.733431 w_k1_MSFT_s110
                                 + 0.749676 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.838439 w_k1_ADM_s110
                                 + 0.974044 w_k1_AXP_s110
                                 + 0.535272 w_k1_DIS_s110
                                 + 0.674407 w_k1_FDX_s110
                                 + 1.32799 w_k1_IBM_s110
                                 + 0.677029 w_k1_INTC_s110
                                 + 0.85769 w_k1_KO_s110 + 1.35542 w_k1_MAT_s110
                                 + 0.654072 w_k1_MSFT_s110
                                 + 0.89905 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.928822 w_k1_ADM_s110
                                 + 0.957379 w_k1_AXP_s110
                                 + 1.10037 w_k1_DIS_s110
                                 + 0.99997 w_k1_FDX_s110
                                 + 0.836651 w_k1_IBM_s110
                                 + 1.12864 w_k1_INTC_s110
                                 + 1.09173 w_k1_KO_s110
                                 + 0.625438 w_k1_MAT_s110
                                 + 1.11169 w_k1_MSFT_s110 + 1.0493 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.2572 w_k1_ADM_s110
                                 + 1.5208 w_k1_AXP_s110 + 1.66558 w_k1_DIS_s110
                                 + 1.45659 w_k1_FDX_s110
                                 + 0.953887 w_k1_IBM_s110
                                 + 1.12125 w_k1_INTC_s110
                                 + 1.26566 w_k1_KO_s110 + 1.1722 w_k1_MAT_s110
                                 + 1.31158 w_k1_MSFT_s110
                                 + 1.42582 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.939515 w_k1_ADM_s110
                                 + 1.01847 w_k1_AXP_s110
                                 + 0.839443 w_k1_DIS_s110
                                 + 0.851183 w_k1_FDX_s110
                                 + 1.09026 w_k1_IBM_s110
                                 + 0.723007 w_k1_INTC_s110
                                 + 0.850724 w_k1_KO_s110
                                 + 0.818146 w_k1_MAT_s110
                                 + 0.744111 w_k1_MSFT_s110
                                 + 0.978601 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.09281 w_k1_ADM_s110
                                 + 1.15654 w_k1_AXP_s110
                                 + 0.944728 w_k1_DIS_s110
                                 + 0.910026 w_k1_FDX_s110
                                 + 1.43508 w_k1_IBM_s110
                                 + 0.886531 w_k1_INTC_s110
                                 + 1.04445 w_k1_KO_s110 + 1.30376 w_k1_MAT_s110
                                 + 0.964558 w_k1_MSFT_s110
                                 + 1.07105 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.13849 w_k1_ADM_s110
                                 + 1.36048 w_k1_AXP_s110
                                 + 1.37191 w_k1_DIS_s110
                                 + 1.13849 w_k1_FDX_s110
                                 + 1.14201 w_k1_IBM_s110
                                 + 0.895835 w_k1_INTC_s110
                                 + 1.15272 w_k1_KO_s110 + 1.57601 w_k1_MAT_s110
                                 + 1.15554 w_k1_MSFT_s110
                                 + 1.12862 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.12971 w_k1_ADM_s111
                                 + 1.38132 w_k1_AXP_s111
                                 + 1.34744 w_k1_DIS_s111
                                 + 1.11574 w_k1_FDX_s111
                                 + 1.89649 w_k1_IBM_s111
                                 + 0.956592 w_k1_INTC_s111
                                 + 1.26432 w_k1_KO_s111 + 2.1555 w_k1_MAT_s111
                                 + 1.00721 w_k1_MSFT_s111
                                 + 1.22602 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.00613 w_k1_ADM_s111
                                 + 1.11872 w_k1_AXP_s111
                                 + 1.25775 w_k1_DIS_s111
                                 + 1.06718 w_k1_FDX_s111
                                 + 1.19028 w_k1_IBM_s111
                                 + 1.17594 w_k1_INTC_s111
                                 + 1.19694 w_k1_KO_s111
                                 + 0.939087 w_k1_MAT_s111
                                 + 1.04218 w_k1_MSFT_s111
                                 + 1.07981 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.887909 w_k1_ADM_s111
                                 + 0.921817 w_k1_AXP_s111
                                 + 0.840258 w_k1_DIS_s111
                                 + 0.998831 w_k1_FDX_s111
                                 + 0.540406 w_k1_IBM_s111
                                 + 1.10151 w_k1_INTC_s111
                                 + 0.698381 w_k1_KO_s111
                                 + 0.517562 w_k1_MAT_s111
                                 + 1.01396 w_k1_MSFT_s111
                                 + 0.913498 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.741788 w_k1_ADM_s111
                                 + 0.688569 w_k1_AXP_s111
                                 + 0.754733 w_k1_DIS_s111
                                 + 0.717975 w_k1_FDX_s111
                                 + 1.00856 w_k1_IBM_s111
                                 + 0.872876 w_k1_INTC_s111
                                 + 0.909679 w_k1_KO_s111
                                 + 1.08809 w_k1_MAT_s111
                                 + 0.811475 w_k1_MSFT_s111
                                 + 0.909714 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.657216 w_k1_ADM_s111
                                 + 0.865547 w_k1_AXP_s111
                                 + 0.880724 w_k1_DIS_s111
                                 + 0.754569 w_k1_FDX_s111
                                 + 1.45416 w_k1_IBM_s111
                                 + 0.943074 w_k1_INTC_s111
                                 + 0.980105 w_k1_KO_s111
                                 + 1.43808 w_k1_MAT_s111
                                 + 0.926854 w_k1_MSFT_s111
                                 + 0.951194 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.12003 w_k1_ADM_s111
                                 + 1.04361 w_k1_AXP_s111
                                 + 0.973584 w_k1_DIS_s111
                                 + 1.07082 w_k1_FDX_s111
                                 + 0.743879 w_k1_IBM_s111
                                 + 1.14286 w_k1_INTC_s111
                                 + 0.71932 w_k1_KO_s111 + 0.74795 w_k1_MAT_s111
                                 + 1.0118 w_k1_MSFT_s111
                                 + 0.919673 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.76641 w_k1_ADM_s111
                                 + 0.713863 w_k1_AXP_s111
                                 + 0.893471 w_k1_DIS_s111
                                 + 0.840657 w_k1_FDX_s111
                                 + 0.871652 w_k1_IBM_s111
                                 + 1.04315 w_k1_INTC_s111
                                 + 0.847437 w_k1_KO_s111
                                 + 0.637255 w_k1_MAT_s111
                                 + 0.946789 w_k1_MSFT_s111
                                 + 0.875173 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.22344 w_k1_ADM_s111
                                 + 1.23808 w_k1_AXP_s111
                                 + 1.60978 w_k1_DIS_s111
                                 + 1.37498 w_k1_FDX_s111
                                 + 1.03496 w_k1_IBM_s111
                                 + 1.27601 w_k1_INTC_s111
                                 + 1.19106 w_k1_KO_s111 + 1.34349 w_k1_MAT_s111
                                 + 1.3217 w_k1_MSFT_s111 + 1.17471 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.19781 w_k1_ADM_s112
                                 + 1.00237 w_k1_AXP_s112
                                 + 1.10285 w_k1_DIS_s112
                                 + 0.984346 w_k1_FDX_s112
                                 + 0.95747 w_k1_IBM_s112
                                 + 1.10656 w_k1_INTC_s112
                                 + 0.991798 w_k1_KO_s112
                                 + 0.922376 w_k1_MAT_s112
                                 + 0.958837 w_k1_MSFT_s112
                                 + 0.915655 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.63456 w_k1_ADM_s112
                                 + 1.78334 w_k1_AXP_s112
                                 + 1.80662 w_k1_DIS_s112
                                 + 1.37492 w_k1_FDX_s112
                                 + 1.30141 w_k1_IBM_s112
                                 + 1.15779 w_k1_INTC_s112
                                 + 1.57168 w_k1_KO_s112 + 1.73829 w_k1_MAT_s112
                                 + 1.25056 w_k1_MSFT_s112
                                 + 1.32937 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.913238 w_k1_ADM_s112
                                 + 0.710562 w_k1_AXP_s112
                                 + 0.652604 w_k1_DIS_s112
                                 + 0.828738 w_k1_FDX_s112
                                 + 0.632546 w_k1_IBM_s112
                                 + 0.913982 w_k1_INTC_s112
                                 + 0.707862 w_k1_KO_s112
                                 + 0.528718 w_k1_MAT_s112
                                 + 0.85312 w_k1_MSFT_s112
                                 + 0.794639 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.14155 w_k1_ADM_s112
                                 + 0.804116 w_k1_AXP_s112
                                 + 0.790513 w_k1_DIS_s112
                                 + 0.789759 w_k1_FDX_s112
                                 + 1.05354 w_k1_IBM_s112
                                 + 0.952126 w_k1_INTC_s112
                                 + 1.10303 w_k1_KO_s112
                                 + 0.772496 w_k1_MAT_s112
                                 + 0.902379 w_k1_MSFT_s112
                                 + 0.945326 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.20776 w_k1_ADM_s112
                                 + 1.11998 w_k1_AXP_s112
                                 + 0.918327 w_k1_DIS_s112
                                 + 0.936013 w_k1_FDX_s112
                                 + 1.39757 w_k1_IBM_s112
                                 + 0.900968 w_k1_INTC_s112
                                 + 1.17571 w_k1_KO_s112 + 1.34916 w_k1_MAT_s112
                                 + 0.923397 w_k1_MSFT_s112
                                 + 0.981549 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.58276 w_k1_ADM_s112
                                 + 1.17805 w_k1_AXP_s112
                                 + 1.24684 w_k1_DIS_s112
                                 + 1.22179 w_k1_FDX_s112
                                 + 0.797094 w_k1_IBM_s112
                                 + 1.13789 w_k1_INTC_s112 + 1.0097 w_k1_KO_s112
                                 + 0.956588 w_k1_MAT_s112
                                 + 1.15514 w_k1_MSFT_s112
                                 + 1.00322 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.21789 w_k1_ADM_s112
                                 + 1.21454 w_k1_AXP_s112
                                 + 0.765646 w_k1_DIS_s112
                                 + 0.96958 w_k1_FDX_s112
                                 + 1.03558 w_k1_IBM_s112
                                 + 1.01993 w_k1_INTC_s112
                                 + 0.91398 w_k1_KO_s112
                                 + 0.866133 w_k1_MAT_s112
                                 + 0.8512 w_k1_MSFT_s112
                                 + 0.817072 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.30052 w_k1_ADM_s112
                                 + 1.04539 w_k1_AXP_s112
                                 + 1.16825 w_k1_DIS_s112
                                 + 0.994833 w_k1_FDX_s112
                                 + 1.16413 w_k1_IBM_s112
                                 + 1.01156 w_k1_INTC_s112 + 1.4416 w_k1_KO_s112
                                 + 1.3697 w_k1_MAT_s112
                                 + 1.08851 w_k1_MSFT_s112
                                 + 1.03849 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.02098 w_k1_ADM_s113
                                 + 0.697431 w_k1_AXP_s113
                                 + 0.775432 w_k1_DIS_s113
                                 + 0.985144 w_k1_FDX_s113
                                 + 0.65755 w_k1_IBM_s113
                                 + 0.807731 w_k1_INTC_s113
                                 + 0.672996 w_k1_KO_s113
                                 + 0.584337 w_k1_MAT_s113
                                 + 0.939869 w_k1_MSFT_s113
                                 + 0.854964 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.18259 w_k1_ADM_s113
                                 + 1.2009 w_k1_AXP_s113 + 1.40932 w_k1_DIS_s113
                                 + 1.12744 w_k1_FDX_s113
                                 + 1.64292 w_k1_IBM_s113
                                 + 0.963745 w_k1_INTC_s113
                                 + 1.4086 w_k1_KO_s113 + 1.80967 w_k1_MAT_s113
                                 + 1.11017 w_k1_MSFT_s113
                                 + 1.06018 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.19705 w_k1_ADM_s113
                                 + 0.901284 w_k1_AXP_s113
                                 + 0.913343 w_k1_DIS_s113
                                 + 0.915885 w_k1_FDX_s113
                                 + 0.942909 w_k1_IBM_s113
                                 + 0.943253 w_k1_INTC_s113
                                 + 0.915107 w_k1_KO_s113
                                 + 0.75647 w_k1_MAT_s113
                                 + 1.00376 w_k1_MSFT_s113
                                 + 0.866423 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.772466 w_k1_ADM_s113
                                 + 0.629527 w_k1_AXP_s113
                                 + 0.628188 w_k1_DIS_s113
                                 + 0.6968 w_k1_FDX_s113
                                 + 0.771769 w_k1_IBM_s113
                                 + 0.594001 w_k1_INTC_s113
                                 + 0.744298 w_k1_KO_s113
                                 + 1.12131 w_k1_MAT_s113
                                 + 0.778354 w_k1_MSFT_s113
                                 + 0.785446 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.84376 w_k1_ADM_s113
                                 + 1.01943 w_k1_AXP_s113
                                 + 0.822829 w_k1_DIS_s113
                                 + 0.875162 w_k1_FDX_s113
                                 + 1.2406 w_k1_IBM_s113
                                 + 1.00942 w_k1_INTC_s113
                                 + 1.02611 w_k1_KO_s113 + 1.50809 w_k1_MAT_s113
                                 + 1.00673 w_k1_MSFT_s113
                                 + 0.935856 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.18715 w_k1_ADM_s113
                                 + 1.37378 w_k1_AXP_s113
                                 + 1.45151 w_k1_DIS_s113
                                 + 1.24908 w_k1_FDX_s113
                                 + 1.19208 w_k1_IBM_s113
                                 + 0.93156 w_k1_INTC_s113 + 1.2106 w_k1_KO_s113
                                 + 1.59667 w_k1_MAT_s113
                                 + 1.09404 w_k1_MSFT_s113
                                 + 1.17641 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.918076 w_k1_ADM_s113
                                 + 1.0319 w_k1_AXP_s113 + 1.12799 w_k1_DIS_s113
                                 + 1.05279 w_k1_FDX_s113
                                 + 1.18518 w_k1_IBM_s113
                                 + 0.949237 w_k1_INTC_s113
                                 + 1.19351 w_k1_KO_s113 + 1.15378 w_k1_MAT_s113
                                 + 1.26424 w_k1_MSFT_s113
                                 + 1.08256 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.47683 w_k1_ADM_s113
                                 + 1.13213 w_k1_AXP_s113
                                 + 1.45326 w_k1_DIS_s113
                                 + 1.27138 w_k1_FDX_s113
                                 + 0.924955 w_k1_IBM_s113
                                 + 0.840441 w_k1_INTC_s113
                                 + 1.14551 w_k1_KO_s113 + 1.4175 w_k1_MAT_s113
                                 + 1.18911 w_k1_MSFT_s113
                                 + 1.06852 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.826988 w_k1_ADM_s114
                                 + 0.692696 w_k1_AXP_s114
                                 + 0.677869 w_k1_DIS_s114
                                 + 0.785349 w_k1_FDX_s114
                                 + 0.652116 w_k1_IBM_s114
                                 + 0.762421 w_k1_INTC_s114
                                 + 0.679929 w_k1_KO_s114
                                 + 0.614444 w_k1_MAT_s114
                                 + 0.838039 w_k1_MSFT_s114
                                 + 0.883541 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.11559 w_k1_ADM_s114
                                 + 0.79476 w_k1_AXP_s114
                                 + 1.20504 w_k1_DIS_s114
                                 + 0.985408 w_k1_FDX_s114
                                 + 0.707742 w_k1_IBM_s114
                                 + 0.836401 w_k1_INTC_s114
                                 + 1.02072 w_k1_KO_s114
                                 + 0.795259 w_k1_MAT_s114
                                 + 1.18333 w_k1_MSFT_s114
                                 + 1.11587 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.959409 w_k1_ADM_s114
                                 + 0.942786 w_k1_AXP_s114
                                 + 0.715477 w_k1_DIS_s114
                                 + 0.796157 w_k1_FDX_s114
                                 + 0.857046 w_k1_IBM_s114
                                 + 0.737855 w_k1_INTC_s114
                                 + 0.678354 w_k1_KO_s114
                                 + 0.873192 w_k1_MAT_s114
                                 + 0.607517 w_k1_MSFT_s114
                                 + 0.950611 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.22449 w_k1_ADM_s114
                                 + 1.34939 w_k1_AXP_s114
                                 + 1.64216 w_k1_DIS_s114
                                 + 1.24736 w_k1_FDX_s114
                                 + 1.02223 w_k1_IBM_s114
                                 + 1.21427 w_k1_INTC_s114
                                 + 1.09788 w_k1_KO_s114
                                 + 0.997428 w_k1_MAT_s114
                                 + 1.15866 w_k1_MSFT_s114
                                 + 1.13843 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.992901 w_k1_ADM_s114
                                 + 0.788921 w_k1_AXP_s114
                                 + 0.849659 w_k1_DIS_s114
                                 + 0.896378 w_k1_FDX_s114
                                 + 0.873696 w_k1_IBM_s114
                                 + 0.826753 w_k1_INTC_s114
                                 + 1.03491 w_k1_KO_s114
                                 + 0.701791 w_k1_MAT_s114
                                 + 0.97744 w_k1_MSFT_s114
                                 + 1.03107 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.27918 w_k1_ADM_s114
                                 + 1.53356 w_k1_AXP_s114
                                 + 2.05078 w_k1_DIS_s114
                                 + 1.38331 w_k1_FDX_s114
                                 + 1.43545 w_k1_IBM_s114
                                 + 1.20539 w_k1_INTC_s114
                                 + 1.55803 w_k1_KO_s114 + 1.81496 w_k1_MAT_s114
                                 + 1.38382 w_k1_MSFT_s114
                                 + 1.64262 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.902544 w_k1_ADM_s114
                                 + 1.18914 w_k1_AXP_s114 + 1.1726 w_k1_DIS_s114
                                 + 0.943094 w_k1_FDX_s114
                                 + 1.42102 w_k1_IBM_s114
                                 + 1.1117 w_k1_INTC_s114 + 1.21863 w_k1_KO_s114
                                 + 1.71919 w_k1_MAT_s114
                                 + 0.994986 w_k1_MSFT_s114
                                 + 1.17513 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.821026 w_k1_ADM_s114
                                 + 0.954855 w_k1_AXP_s114
                                 + 1.10818 w_k1_DIS_s114
                                 + 0.901443 w_k1_FDX_s114
                                 + 1.19994 w_k1_IBM_s114
                                 + 0.839329 w_k1_INTC_s114
                                 + 1.06215 w_k1_KO_s114 + 1.14453 w_k1_MAT_s114
                                 + 1.0211 w_k1_MSFT_s114 + 1.02521 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.959376 w_k1_ADM_s115
                                 + 1.24362 w_k1_AXP_s115
                                 + 1.23389 w_k1_DIS_s115
                                 + 0.987857 w_k1_FDX_s115
                                 + 1.57588 w_k1_IBM_s115
                                 + 1.20652 w_k1_INTC_s115
                                 + 1.50516 w_k1_KO_s115 + 1.87013 w_k1_MAT_s115
                                 + 1.29197 w_k1_MSFT_s115
                                 + 1.20307 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.03545 w_k1_ADM_s115
                                 + 0.888617 w_k1_AXP_s115
                                 + 0.599857 w_k1_DIS_s115
                                 + 0.847854 w_k1_FDX_s115
                                 + 0.6022 w_k1_IBM_s115
                                 + 0.829293 w_k1_INTC_s115
                                 + 0.671138 w_k1_KO_s115
                                 + 0.505547 w_k1_MAT_s115
                                 + 0.829452 w_k1_MSFT_s115
                                 + 0.936874 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.37071 w_k1_ADM_s115
                                 + 1.75877 w_k1_AXP_s115
                                 + 1.58384 w_k1_DIS_s115
                                 + 1.26288 w_k1_FDX_s115
                                 + 1.41064 w_k1_IBM_s115
                                 + 1.34763 w_k1_INTC_s115
                                 + 1.38652 w_k1_KO_s115 + 1.64601 w_k1_MAT_s115
                                 + 1.3859 w_k1_MSFT_s115 + 1.14945 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.00907 w_k1_ADM_s115
                                 + 0.726808 w_k1_AXP_s115
                                 + 0.657057 w_k1_DIS_s115
                                 + 0.710813 w_k1_FDX_s115
                                 + 0.838973 w_k1_IBM_s115
                                 + 0.809946 w_k1_INTC_s115
                                 + 0.809824 w_k1_KO_s115
                                 + 0.579341 w_k1_MAT_s115
                                 + 0.829186 w_k1_MSFT_s115
                                 + 0.855208 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04051 w_k1_ADM_s115
                                 + 1.38828 w_k1_AXP_s115 + 1.0692 w_k1_DIS_s115
                                 + 1.03014 w_k1_FDX_s115
                                 + 1.14716 w_k1_IBM_s115
                                 + 1.24059 w_k1_INTC_s115
                                 + 1.07835 w_k1_KO_s115 + 1.33587 w_k1_MAT_s115
                                 + 1.17831 w_k1_MSFT_s115
                                 + 1.10224 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.0311 w_k1_ADM_s115
                                 + 0.86134 w_k1_AXP_s115
                                 + 0.895465 w_k1_DIS_s115
                                 + 0.867757 w_k1_FDX_s115
                                 + 1.12032 w_k1_IBM_s115
                                 + 0.778962 w_k1_INTC_s115
                                 + 1.11235 w_k1_KO_s115 + 1.04378 w_k1_MAT_s115
                                 + 0.876948 w_k1_MSFT_s115
                                 + 1.00384 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.11442 w_k1_ADM_s115
                                 + 1.27931 w_k1_AXP_s115 + 1.3684 w_k1_DIS_s115
                                 + 1.22226 w_k1_FDX_s115
                                 + 1.07411 w_k1_IBM_s115
                                 + 1.15032 w_k1_INTC_s115 + 1.2764 w_k1_KO_s115
                                 + 1.08999 w_k1_MAT_s115
                                 + 1.21219 w_k1_MSFT_s115
                                 + 1.02883 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.971995 w_k1_ADM_s115
                                 + 0.854033 w_k1_AXP_s115
                                 + 0.912097 w_k1_DIS_s115
                                 + 0.972712 w_k1_FDX_s115
                                 + 0.709129 w_k1_IBM_s115
                                 + 0.906885 w_k1_INTC_s115
                                 + 0.913372 w_k1_KO_s115
                                 + 0.515596 w_k1_MAT_s115
                                 + 0.988145 w_k1_MSFT_s115
                                 + 0.87634 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.915251 w_k1_ADM_s116
                                 + 0.61216 w_k1_AXP_s116
                                 + 0.592241 w_k1_DIS_s116
                                 + 0.692957 w_k1_FDX_s116
                                 + 0.813979 w_k1_IBM_s116
                                 + 0.830993 w_k1_INTC_s116
                                 + 0.922104 w_k1_KO_s116
                                 + 0.920072 w_k1_MAT_s116
                                 + 0.885876 w_k1_MSFT_s116
                                 + 0.783375 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.800642 w_k1_ADM_s116
                                 + 0.649397 w_k1_AXP_s116
                                 + 0.617484 w_k1_DIS_s116
                                 + 0.742783 w_k1_FDX_s116
                                 + 0.658811 w_k1_IBM_s116
                                 + 0.868269 w_k1_INTC_s116
                                 + 0.813471 w_k1_KO_s116
                                 + 0.734993 w_k1_MAT_s116
                                 + 0.995785 w_k1_MSFT_s116
                                 + 0.862138 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.846125 w_k1_ADM_s116
                                 + 0.896843 w_k1_AXP_s116
                                 + 0.708086 w_k1_DIS_s116
                                 + 0.751504 w_k1_FDX_s116
                                 + 1.1456 w_k1_IBM_s116
                                 + 0.934614 w_k1_INTC_s116
                                 + 0.878451 w_k1_KO_s116
                                 + 1.18292 w_k1_MAT_s116
                                 + 0.83498 w_k1_MSFT_s116
                                 + 0.905245 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.34805 w_k1_ADM_s116
                                 + 1.37581 w_k1_AXP_s116
                                 + 1.46011 w_k1_DIS_s116
                                 + 1.28005 w_k1_FDX_s116
                                 + 1.34608 w_k1_IBM_s116
                                 + 1.41811 w_k1_INTC_s116 + 1.4969 w_k1_KO_s116
                                 + 1.30498 w_k1_MAT_s116
                                 + 1.39456 w_k1_MSFT_s116
                                 + 1.12773 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.13525 w_k1_ADM_s116
                                 + 1.17579 w_k1_AXP_s116
                                 + 0.903444 w_k1_DIS_s116
                                 + 1.01898 w_k1_FDX_s116
                                 + 1.28607 w_k1_IBM_s116
                                 + 1.22562 w_k1_INTC_s116
                                 + 1.16757 w_k1_KO_s116 + 1.31382 w_k1_MAT_s116
                                 + 1.08295 w_k1_MSFT_s116
                                 + 1.03805 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.18071 w_k1_ADM_s116
                                 + 1.18346 w_k1_AXP_s116
                                 + 1.19172 w_k1_DIS_s116
                                 + 1.13574 w_k1_FDX_s116
                                 + 0.854968 w_k1_IBM_s116
                                 + 1.21565 w_k1_INTC_s116
                                 + 1.11504 w_k1_KO_s116
                                 + 0.999583 w_k1_MAT_s116
                                 + 1.20254 w_k1_MSFT_s116
                                 + 1.02952 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.62392 w_k1_ADM_s116
                                 + 1.17338 w_k1_AXP_s116
                                 + 1.61052 w_k1_DIS_s116
                                 + 1.31882 w_k1_FDX_s116
                                 + 0.957312 w_k1_IBM_s116
                                 + 1.15357 w_k1_INTC_s116 + 1.2959 w_k1_KO_s116
                                 + 0.927523 w_k1_MAT_s116
                                 + 1.43512 w_k1_MSFT_s116
                                 + 1.15432 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.989784 w_k1_ADM_s116
                                 + 0.888676 w_k1_AXP_s116
                                 + 0.933849 w_k1_DIS_s116
                                 + 0.964056 w_k1_FDX_s116
                                 + 0.774026 w_k1_IBM_s116
                                 + 1.16602 w_k1_INTC_s116 + 1.0094 w_k1_KO_s116
                                 + 0.884083 w_k1_MAT_s116
                                 + 1.16233 w_k1_MSFT_s116
                                 + 0.970102 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.812178 w_k1_ADM_s117
                                 + 0.71768 w_k1_AXP_s117 + 0.9405 w_k1_DIS_s117
                                 + 0.876671 w_k1_FDX_s117
                                 + 0.958302 w_k1_IBM_s117
                                 + 0.962598 w_k1_INTC_s117
                                 + 1.01551 w_k1_KO_s117
                                 + 0.676056 w_k1_MAT_s117
                                 + 0.924642 w_k1_MSFT_s117
                                 + 0.92758 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.05395 w_k1_ADM_s117
                                 + 1.15785 w_k1_AXP_s117
                                 + 1.07295 w_k1_DIS_s117
                                 + 1.00073 w_k1_FDX_s117
                                 + 1.32234 w_k1_IBM_s117
                                 + 0.87991 w_k1_INTC_s117
                                 + 1.14248 w_k1_KO_s117 + 1.19101 w_k1_MAT_s117
                                 + 0.898475 w_k1_MSFT_s117
                                 + 1.0627 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.778698 w_k1_ADM_s117
                                 + 0.699706 w_k1_AXP_s117
                                 + 0.673416 w_k1_DIS_s117
                                 + 0.819291 w_k1_FDX_s117
                                 + 0.540275 w_k1_IBM_s117
                                 + 0.708905 w_k1_INTC_s117
                                 + 0.640592 w_k1_KO_s117
                                 + 0.389446 w_k1_MAT_s117
                                 + 0.76453 w_k1_MSFT_s117
                                 + 0.773618 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.27537 w_k1_ADM_s117
                                 + 1.46856 w_k1_AXP_s117
                                 + 1.59068 w_k1_DIS_s117
                                 + 1.38456 w_k1_FDX_s117
                                 + 0.968435 w_k1_IBM_s117
                                 + 1.23257 w_k1_INTC_s117
                                 + 1.20287 w_k1_KO_s117
                                 + 0.976881 w_k1_MAT_s117
                                 + 1.23005 w_k1_MSFT_s117
                                 + 1.09531 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.06708 w_k1_ADM_s117
                                 + 1.10411 w_k1_AXP_s117
                                 + 1.00423 w_k1_DIS_s117
                                 + 1.11292 w_k1_FDX_s117
                                 + 0.63266 w_k1_IBM_s117
                                 + 0.922074 w_k1_INTC_s117
                                 + 0.825443 w_k1_KO_s117
                                 + 0.768779 w_k1_MAT_s117
                                 + 0.87166 w_k1_MSFT_s117
                                 + 0.899932 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.968172 w_k1_ADM_s117
                                 + 0.848064 w_k1_AXP_s117
                                 + 0.759617 w_k1_DIS_s117
                                 + 0.744915 w_k1_FDX_s117
                                 + 0.971485 w_k1_IBM_s117
                                 + 0.708996 w_k1_INTC_s117
                                 + 0.99321 w_k1_KO_s117
                                 + 0.862085 w_k1_MAT_s117
                                 + 0.769113 w_k1_MSFT_s117
                                 + 0.850112 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.33499 w_k1_ADM_s117
                                 + 1.46153 w_k1_AXP_s117
                                 + 1.74167 w_k1_DIS_s117 + 1.2033 w_k1_FDX_s117
                                 + 1.56865 w_k1_IBM_s117
                                 + 1.14602 w_k1_INTC_s117
                                 + 1.50665 w_k1_KO_s117 + 1.70802 w_k1_MAT_s117
                                 + 1.15334 w_k1_MSFT_s117
                                 + 1.30409 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.17658 w_k1_ADM_s117
                                 + 1.10542 w_k1_AXP_s117
                                 + 1.26355 w_k1_DIS_s117
                                 + 1.05418 w_k1_FDX_s117
                                 + 1.04607 w_k1_IBM_s117
                                 + 0.974982 w_k1_INTC_s117
                                 + 1.24659 w_k1_KO_s117
                                 + 0.858154 w_k1_MAT_s117
                                 + 1.04598 w_k1_MSFT_s117
                                 + 1.05616 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0287479669332288 w_k1_ADM_s000
                                 + 0.0470621776847353 w_k1_ADM_s001
                                 + 0.0442178390172958 w_k1_ADM_s002
                                 + 0.0295701587893933 w_k1_ADM_s003
                                 + 0.0159124665775266 w_k1_ADM_s004
                                 + 0.02752288525592 w_k1_ADM_s005
                                 + 0.028313710005853 w_k1_ADM_s006
                                 + 0.0761739896948702 w_k1_ADM_s007
                                 + 0.0171647825237318 w_k1_ADM_s010
                                 + 0.0203131932699097 w_k1_ADM_s011
                                 + 0.00788404529834662 w_k1_ADM_s012
                                 + 0.0300311700658332 w_k1_ADM_s013
                                 + 0.0204709458057527 w_k1_ADM_s014
                                 + 0.0114492398812767 w_k1_ADM_s015
                                 + 0.0163314140961528 w_k1_ADM_s016
                                 + 0.0100379538767442 w_k1_ADM_s017
                                 + 0.0284065233544536 w_k1_ADM_s100
                                 + 0.0211941791698664 w_k1_ADM_s101
                                 + 0.0491030609015539 w_k1_ADM_s102
                                 + 0.0406316019414003 w_k1_ADM_s103
                                 + 0.0870344378044813 w_k1_ADM_s104
                                 + 0.0949553651845776 w_k1_ADM_s105
                                 + 0.026323124243503 w_k1_ADM_s106
                                 + 0.060243939272921 w_k1_ADM_s107
                                 + 0.0172394728057767 w_k1_ADM_s110
                                 + 0.0312803012406092 w_k1_ADM_s111
                                 + 0.0275032224108246 w_k1_ADM_s112
                                 + 0.0163384259424776 w_k1_ADM_s113
                                 + 0.0197223193073439 w_k1_ADM_s114
                                 + 0.0144755356415217 w_k1_ADM_s115
                                 + 0.010136147820977 w_k1_ADM_s116
                                 + 0.0468088597023682 w_k1_ADM_s117
                                 + 0.0296026517759336 w_k1_AXP_s000
                                 + 0.049644406843691 w_k1_AXP_s001
                                 + 0.0441134889586584 w_k1_AXP_s002
                                 + 0.0324441821899521 w_k1_AXP_s003
                                 + 0.0145194506773948 w_k1_AXP_s004
                                 + 0.0296718713685974 w_k1_AXP_s005
                                 + 0.0298890909315512 w_k1_AXP_s006
                                 + 0.0691134083593834 w_k1_AXP_s007
                                 + 0.0183039688956237 w_k1_AXP_s010
                                 + 0.0234084347127928 w_k1_AXP_s011
                                 + 0.00854721285094638 w_k1_AXP_s012
                                 + 0.0304100299841103 w_k1_AXP_s013
                                 + 0.0238178220630539 w_k1_AXP_s014
                                 + 0.00957772456901788 w_k1_AXP_s015
                                 + 0.0179779149443031 w_k1_AXP_s016
                                 + 0.00816817961629954 w_k1_AXP_s017
                                 + 0.027218800401875 w_k1_AXP_s100
                                 + 0.0192274988529979 w_k1_AXP_s101
                                 + 0.0454101021092891 w_k1_AXP_s102
                                 + 0.0461504351404866 w_k1_AXP_s103
                                 + 0.081101087357914 w_k1_AXP_s104
                                 + 0.0935098991391755 w_k1_AXP_s105
                                 + 0.0285932413833973 w_k1_AXP_s106
                                 + 0.0602870516590876 w_k1_AXP_s107
                                 + 0.0199326006566151 w_k1_AXP_s110
                                 + 0.0326052329211433 w_k1_AXP_s111
                                 + 0.0235113022398451 w_k1_AXP_s112
                                 + 0.0147672948885084 w_k1_AXP_s113
                                 + 0.0200245444550985 w_k1_AXP_s114
                                 + 0.0151120858864823 w_k1_AXP_s115
                                 + 0.00955092239332335 w_k1_AXP_s116
                                 + 0.0466939300118144 w_k1_AXP_s117
                                 + 0.0281907538309755 w_k1_DIS_s000
                                 + 0.0514538700906448 w_k1_DIS_s001
                                 + 0.0421015393315868 w_k1_DIS_s002
                                 + 0.0334945278279158 w_k1_DIS_s003
                                 + 0.01534443091643 w_k1_DIS_s004
                                 + 0.0264144026842726 w_k1_DIS_s005
                                 + 0.0303984626539646 w_k1_DIS_s006
                                 + 0.0697003891639726 w_k1_DIS_s007
                                 + 0.018206262030929 w_k1_DIS_s010
                                 + 0.0220332902775702 w_k1_DIS_s011
                                 + 0.0100413513251458 w_k1_DIS_s012
                                 + 0.0320446246067518 w_k1_DIS_s013
                                 + 0.0242815257740079 w_k1_DIS_s014
                                 + 0.00942586544689916 w_k1_DIS_s015
                                 + 0.0180000125257323 w_k1_DIS_s016
                                 + 0.00842499143099095 w_k1_DIS_s017
                                 + 0.0300690098683847 w_k1_DIS_s100
                                 + 0.0212807041046284 w_k1_DIS_s101
                                 + 0.0530529990294669 w_k1_DIS_s102
                                 + 0.0428684278753352 w_k1_DIS_s103
                                 + 0.0800922142889734 w_k1_DIS_s104
                                 + 0.091921804994967 w_k1_DIS_s105
                                 + 0.0300015475208063 w_k1_DIS_s106
                                 + 0.064075856450058 w_k1_DIS_s107
                                 + 0.0186342886480218 w_k1_DIS_s110
                                 + 0.0353056813792448 w_k1_DIS_s111
                                 + 0.0231240719536673 w_k1_DIS_s112
                                 + 0.0156530139841295 w_k1_DIS_s113
                                 + 0.0232294059130495 w_k1_DIS_s114
                                 + 0.0143397051507257 w_k1_DIS_s115
                                 + 0.00931774944555834 w_k1_DIS_s116
                                 + 0.0489209619984302 w_k1_DIS_s117
                                 + 0.0289739851848227 w_k1_FDX_s000
                                 + 0.048418043894877 w_k1_FDX_s001
                                 + 0.04477778324478 w_k1_FDX_s002
                                 + 0.0332768160914387 w_k1_FDX_s003
                                 + 0.0142913149025038 w_k1_FDX_s004
                                 + 0.0289103235417484 w_k1_FDX_s005
                                 + 0.0271526193160348 w_k1_FDX_s006
                                 + 0.0689949915892965 w_k1_FDX_s007
                                 + 0.0170437415249524 w_k1_FDX_s010
                                 + 0.021190474092519 w_k1_FDX_s011
                                 + 0.00970975218798707 w_k1_FDX_s012
                                 + 0.0322399876317004 w_k1_FDX_s013
                                 + 0.0245240793731449 w_k1_FDX_s014
                                 + 0.00920006624558427 w_k1_FDX_s015
                                 + 0.0168487857466443 w_k1_FDX_s016
                                 + 0.00901275042845435 w_k1_FDX_s017
                                 + 0.0297299716955195 w_k1_FDX_s100
                                 + 0.0204620606064555 w_k1_FDX_s101
                                 + 0.0468216687707033 w_k1_FDX_s102
                                 + 0.0444399456618242 w_k1_FDX_s103
                                 + 0.0800009902786118 w_k1_FDX_s104
                                 + 0.0954867818441655 w_k1_FDX_s105
                                 + 0.0276706296549528 w_k1_FDX_s106
                                 + 0.0626725938090848 w_k1_FDX_s107
                                 + 0.0176127809744238 w_k1_FDX_s110
                                 + 0.0331977595226166 w_k1_FDX_s111
                                 + 0.0221881955097925 w_k1_FDX_s112
                                 + 0.015192832493659 w_k1_FDX_s113
                                 + 0.0195701557225457 w_k1_FDX_s114
                                 + 0.0139099933277502 w_k1_FDX_s115
                                 + 0.00921968064479761 w_k1_FDX_s116
                                 + 0.044363906542435 w_k1_FDX_s117
                                 + 0.0282043357641528 w_k1_IBM_s000
                                 + 0.0479285344809919 w_k1_IBM_s001
                                 + 0.043121254304548 w_k1_IBM_s002
                                 + 0.0352817855096819 w_k1_IBM_s003
                                 + 0.0150006555549899 w_k1_IBM_s004
                                 + 0.0292248249480001 w_k1_IBM_s005
                                 + 0.0295492483423417 w_k1_IBM_s006
                                 + 0.0714827846592614 w_k1_IBM_s007
                                 + 0.0186521066393848 w_k1_IBM_s010
                                 + 0.0227846663901881 w_k1_IBM_s011
                                 + 0.00958650571968154 w_k1_IBM_s012
                                 + 0.0330240192106041 w_k1_IBM_s013
                                 + 0.0230097223996875 w_k1_IBM_s014
                                 + 0.00979782142577223 w_k1_IBM_s015
                                 + 0.0178708992833684 w_k1_IBM_s016
                                 + 0.00944630153215758 w_k1_IBM_s017
                                 + 0.0293684413336696 w_k1_IBM_s100
                                 + 0.0230116370808466 w_k1_IBM_s101
                                 + 0.0500186077413817 w_k1_IBM_s102
                                 + 0.0444986341898087 w_k1_IBM_s103
                                 + 0.0829314500729493 w_k1_IBM_s104
                                 + 0.0912886092850814 w_k1_IBM_s105
                                 + 0.0265623702844538 w_k1_IBM_s106
                                 + 0.0624451552046628 w_k1_IBM_s107
                                 + 0.0184873792715588 w_k1_IBM_s110
                                 + 0.0332473830732554 w_k1_IBM_s111
                                 + 0.0214413347996118 w_k1_IBM_s112
                                 + 0.0156669841679606 w_k1_IBM_s113
                                 + 0.0202562479190771 w_k1_IBM_s114
                                 + 0.0136829707414438 w_k1_IBM_s115
                                 + 0.0094831787916226 w_k1_IBM_s116
                                 + 0.0451871223776332 w_k1_IBM_s117
                                 + 0.0291624510987143 w_k1_INTC_s000
                                 + 0.0548659947263482 w_k1_INTC_s001
                                 + 0.0445078591295792 w_k1_INTC_s002
                                 + 0.0320095788874307 w_k1_INTC_s003
                                 + 0.0147718657042383 w_k1_INTC_s004
                                 + 0.0308334547360508 w_k1_INTC_s005
                                 + 0.0312175160906079 w_k1_INTC_s006
                                 + 0.0770212017508634 w_k1_INTC_s007
                                 + 0.0193823313311122 w_k1_INTC_s010
                                 + 0.0198207021691495 w_k1_INTC_s011
                                 + 0.00924450402512087 w_k1_INTC_s012
                                 + 0.0326306005223409 w_k1_INTC_s013
                                 + 0.0210647455086577 w_k1_INTC_s014
                                 + 0.0088412925116222 w_k1_INTC_s015
                                 + 0.0196349491999901 w_k1_INTC_s016
                                 + 0.00959217900740871 w_k1_INTC_s017
                                 + 0.0274209119957895 w_k1_INTC_s100
                                 + 0.0219464467229088 w_k1_INTC_s101
                                 + 0.0426974145429785 w_k1_INTC_s102
                                 + 0.0426324548264857 w_k1_INTC_s103
                                 + 0.0773190998259118 w_k1_INTC_s104
                                 + 0.0981129344133226 w_k1_INTC_s105
                                 + 0.0238394381348074 w_k1_INTC_s106
                                 + 0.0637587340375677 w_k1_INTC_s107
                                 + 0.0163878410684272 w_k1_INTC_s110
                                 + 0.0361962847229936 w_k1_INTC_s111
                                 + 0.0225669290194596 w_k1_INTC_s112
                                 + 0.0134508936478345 w_k1_INTC_s113
                                 + 0.0184648778930796 w_k1_INTC_s114
                                 + 0.0141119194520993 w_k1_INTC_s115
                                 + 0.0104734014276228 w_k1_INTC_s116
                                 + 0.041087052184281 w_k1_INTC_s117
                                 + 0.0288774783136871 w_k1_KO_s000
                                 + 0.0473059888678168 w_k1_KO_s001
                                 + 0.0382343989580276 w_k1_KO_s002
                                 + 0.0341844823639832 w_k1_KO_s003
                                 + 0.0175527987741697 w_k1_KO_s004
                                 + 0.0288038803965743 w_k1_KO_s005
                                 + 0.0329965151907897 w_k1_KO_s006
                                 + 0.0689629249281769 w_k1_KO_s007
                                 + 0.0181079744391374 w_k1_KO_s010
                                 + 0.0213352662510262 w_k1_KO_s011
                                 + 0.00966758593283195 w_k1_KO_s012
                                 + 0.032570644972959 w_k1_KO_s013
                                 + 0.0255061424320338 w_k1_KO_s014
                                 + 0.00983310702786168 w_k1_KO_s015
                                 + 0.0172464827523386 w_k1_KO_s016
                                 + 0.00948604421147668 w_k1_KO_s017
                                 + 0.0300221761783932 w_k1_KO_s100
                                 + 0.0218327569751671 w_k1_KO_s101
                                 + 0.0542344032705388 w_k1_KO_s102
                                 + 0.0447808184084095 w_k1_KO_s103
                                 + 0.095560805733591 w_k1_KO_s104
                                 + 0.0896124851755482 w_k1_KO_s105
                                 + 0.0253722742672898 w_k1_KO_s106
                                 + 0.058202072397321 w_k1_KO_s107
                                 + 0.0179219812878001 w_k1_KO_s110
                                 + 0.031771858519325 w_k1_KO_s111
                                 + 0.0238612783337472 w_k1_KO_s112
                                 + 0.015237773078923 w_k1_KO_s113
                                 + 0.0206764242865605 w_k1_KO_s114
                                 + 0.0147520349827478 w_k1_KO_s115
                                 + 0.0100944764735473 w_k1_KO_s116
                                 + 0.0481638358450659 w_k1_KO_s117
                                 + 0.03274943226277 w_k1_MAT_s000
                                 + 0.036675188436081 w_k1_MAT_s001
                                 + 0.0395846972370744 w_k1_MAT_s002
                                 + 0.0342303264050796 w_k1_MAT_s003
                                 + 0.0142655436395097 w_k1_MAT_s004
                                 + 0.0316548244232422 w_k1_MAT_s005
                                 + 0.0327425515173569 w_k1_MAT_s006
                                 + 0.0740365235240267 w_k1_MAT_s007
                                 + 0.0161625349929141 w_k1_MAT_s010
                                 + 0.0237064703566652 w_k1_MAT_s011
                                 + 0.00864474444199996 w_k1_MAT_s012
                                 + 0.0253920937608557 w_k1_MAT_s013
                                 + 0.0240218940883962 w_k1_MAT_s014
                                 + 0.00936623123495748 w_k1_MAT_s015
                                 + 0.0147860048925476 w_k1_MAT_s016
                                 + 0.00853285669064418 w_k1_MAT_s017
                                 + 0.0269058369639249 w_k1_MAT_s100
                                 + 0.0168918334256495 w_k1_MAT_s101
                                 + 0.0495752857502847 w_k1_MAT_s102
                                 + 0.0468579203378072 w_k1_MAT_s103
                                 + 0.0862956401546192 w_k1_MAT_s104
                                 + 0.0965559812492393 w_k1_MAT_s105
                                 + 0.0284734202220327 w_k1_MAT_s106
                                 + 0.0545979580557296 w_k1_MAT_s107
                                 + 0.0188452100033431 w_k1_MAT_s110
                                 + 0.0319993723940667 w_k1_MAT_s111
                                 + 0.0221928804086075 w_k1_MAT_s112
                                 + 0.0176191361709191 w_k1_MAT_s113
                                 + 0.0201840722252019 w_k1_MAT_s114
                                 + 0.0134155306483451 w_k1_MAT_s115
                                 + 0.0100686924168582 w_k1_MAT_s116
                                 + 0.04104305197982 w_k1_MAT_s117
                                 + 0.0279370034164437 w_k1_MSFT_s000
                                 + 0.0484293609824493 w_k1_MSFT_s001
                                 + 0.0418234222495731 w_k1_MSFT_s002
                                 + 0.0338744060561856 w_k1_MSFT_s003
                                 + 0.0154103804753781 w_k1_MSFT_s004
                                 + 0.0264541207112251 w_k1_MSFT_s005
                                 + 0.0290866148682226 w_k1_MSFT_s006
                                 + 0.0679077987721043 w_k1_MSFT_s007
                                 + 0.0169354280907692 w_k1_MSFT_s010
                                 + 0.0220233223379698 w_k1_MSFT_s011
                                 + 0.00995989425497103 w_k1_MSFT_s012
                                 + 0.0309165981810561 w_k1_MSFT_s013
                                 + 0.0218792511076065 w_k1_MSFT_s014
                                 + 0.00824081340278925 w_k1_MSFT_s015
                                 + 0.0172876554807887 w_k1_MSFT_s016
                                 + 0.0101937867506691 w_k1_MSFT_s017
                                 + 0.0289595432697247 w_k1_MSFT_s100
                                 + 0.020943763730057 w_k1_MSFT_s101
                                 + 0.0489430892958004 w_k1_MSFT_s102
                                 + 0.0396364741335727 w_k1_MSFT_s103
                                 + 0.085060729791782 w_k1_MSFT_s104
                                 + 0.0935910829861517 w_k1_MSFT_s105
                                 + 0.0252333907221019 w_k1_MSFT_s106
                                 + 0.0622617983835077 w_k1_MSFT_s107
                                 + 0.0177434990112804 w_k1_MSFT_s110
                                 + 0.0337370063527794 w_k1_MSFT_s111
                                 + 0.0220328157569222 w_k1_MSFT_s112
                                 + 0.0157284436241849 w_k1_MSFT_s113
                                 + 0.0204464609742516 w_k1_MSFT_s114
                                 + 0.0146917186675095 w_k1_MSFT_s115
                                 + 0.0103757261567525 w_k1_MSFT_s116
                                 + 0.0419176344256284 w_k1_MSFT_s117
                                 + 0.0305807596206855 w_k1_PG_s000
                                 + 0.0484174248609782 w_k1_PG_s001
                                 + 0.0447821270243787 w_k1_PG_s002
                                 + 0.0353369588313957 w_k1_PG_s003
                                 + 0.0143945680197255 w_k1_PG_s004
                                 + 0.0270638451806292 w_k1_PG_s005
                                 + 0.0285036940268035 w_k1_PG_s006
                                 + 0.0697897289242589 w_k1_PG_s007
                                 + 0.0165764223541575 w_k1_PG_s010
                                 + 0.0215055499109166 w_k1_PG_s011
                                 + 0.00895737903468363 w_k1_PG_s012
                                 + 0.0302892130631963 w_k1_PG_s013
                                 + 0.0238474636472342 w_k1_PG_s014
                                 + 0.0094361787551435 w_k1_PG_s015
                                 + 0.0166596219952155 w_k1_PG_s016
                                 + 0.00901597514052965 w_k1_PG_s017
                                 + 0.0286756617935049 w_k1_PG_s100
                                 + 0.0219038967634367 w_k1_PG_s101
                                 + 0.0486116326165409 w_k1_PG_s102
                                 + 0.0443361338454872 w_k1_PG_s103
                                 + 0.082554135710915 w_k1_PG_s104
                                 + 0.0898227793035351 w_k1_PG_s105
                                 + 0.0272846147486299 w_k1_PG_s106
                                 + 0.0589551685316594 w_k1_PG_s107
                                 + 0.0187016312253208 w_k1_PG_s110
                                 + 0.0328793735291776 w_k1_PG_s111
                                 + 0.0213641204153498 w_k1_PG_s112
                                 + 0.0144878711635487 w_k1_PG_s113
                                 + 0.0214841952915366 w_k1_PG_s114
                                 + 0.0137504890437593 w_k1_PG_s115
                                 + 0.00930394589674456 w_k1_PG_s116
                                 + 0.0438150432785473 w_k1_PG_s117 >= 1.11
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
