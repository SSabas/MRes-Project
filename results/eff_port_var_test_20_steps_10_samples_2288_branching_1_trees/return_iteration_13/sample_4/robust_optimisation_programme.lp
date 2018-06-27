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
 w_k1_ADM_s0_return_constraint:  1.2121 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.98094 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.901994 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.25641 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.15036 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.876812 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.800763 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.933191 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.02642 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.845976 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.754279 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.22578 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.795611 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.22944 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.950652 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.782403 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.909929 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.02461 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.13493 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.710925 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.76102 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.06287 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.860439 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.923369 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.25942 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.833186 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.181 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.36636 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.992069 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.20135 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.02782 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.43142 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.15538 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.711963 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.03684 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.810041 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.20203 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.864315 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18755 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.90672 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.840777 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.245 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.56526 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.877029 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.70149 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.905552 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.969341 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.155 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.585903 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.45083 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.922874 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.08381 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.31871 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.742765 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.871001 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.54928 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.15908 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.935227 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.719776 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.2745 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.77363 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.941957 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.48012 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.793858 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.886499 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.18257 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.16575 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.18843 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.960318 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.54879 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.21351 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.641101 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.14206 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.85413 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.50925 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.984762 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3242 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.922188 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.805766 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.39362 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.47023 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.880407 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.645619 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.812526 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.863035 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.914984 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.615642 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.41075 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.781778 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.22697 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.28225 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.854541 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.992593 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.66992 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.11092 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.81766 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.57901 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.32806 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.750527 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.0146 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.76461 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.701513 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.17515 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.53846 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.09465 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.18473 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.748278 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.71814 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.20292 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.569476 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.0666 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.668695 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.3985 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.889669 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20867 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.948253 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.926813 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.22079 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.22483 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.930167 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.83121 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.795951 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.872076 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.904771 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.62281 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.17347 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.954572 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.17212 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.09097 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.874862 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.04733 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.22952 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.22038 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.83408 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.757874 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.21666 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.779639 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.99511 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.51692 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.91798 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.21117 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.15066 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.0043 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.24518 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.992066 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.44689 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.02998 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.666402 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.02914 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.767353 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.17221 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.872216 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.01108 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.03673 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.865654 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.30418 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.46623 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.885467 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.545003 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.10793 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.887759 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.33471 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.746115 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.39989 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.761884 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.744476 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.29623 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.738205 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.595538 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.71636 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.721597 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.15945 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.688355 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.00196 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.930864 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.986114 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.02579 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.582601 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.687285 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.87187 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.29467 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.908718 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.531912 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.08204 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.47744 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.671671 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.795951 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.967645 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.44927 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.34646 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13742 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.929102 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.830184 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.19187 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.40172 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.92529 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.01161 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.919446 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.0552 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06396 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.799756 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.30362 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.08541 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.2603 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.09408 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.788359 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.987008 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.10551 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.08189 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.881495 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.846545 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.17932 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.763001 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.913587 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.29233 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.835905 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.0656 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.10976 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.873716 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.960619 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.04729 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.45045 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.927166 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.812408 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.11784 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.957946 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.34751 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.986407 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.09773 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.967473 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.862127 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.29199 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.32985 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.917201 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.703398 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.0839 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.796991 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.13111 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.783497 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.39456 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.831603 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.953236 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.15302 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.974415 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.851065 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.56405 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.983394 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.02032 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.692845 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.17097 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.906191 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.00849 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.32908 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.697733 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.979395 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.64032 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.12757 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.11253 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.689289 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.28802 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.31135 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.693274 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.927765 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.802513 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.41879 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.06543 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.05294 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.09545 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.930211 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.36551 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.49518 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.733857 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.393582 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.914067 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.811017 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.24427 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.505719 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.74177 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.695997 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.652496 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.62553 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.749945 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.654656 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.97139 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.710534 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.21408 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.713439 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.971621 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.02226 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.959198 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.26356 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.708069 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.676367 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.79012 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.30169 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.11788 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.649961 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.46136 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.73179 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.670184 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.843919 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.878477 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.55782 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.34783 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12286 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.965962 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.899008 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.25734 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.23766 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.937054 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.866738 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.853029 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.989544 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.95036 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.736127 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.22625 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.918976 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.17838 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.00901 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.949635 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.18859 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.26632 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.10581 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.820736 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.765415 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.16005 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.811685 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.870515 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.45565 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.851203 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.15287 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.20743 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.895824 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.10626 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.902127 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.32311 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.969938 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.715918 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.07412 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.783545 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.09343 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.904308 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11043 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.97705 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.933045 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.14163 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.16548 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.935876 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.772574 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.879975 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.905191 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.960868 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.72502 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.22707 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.899539 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.05688 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.07792 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.93278 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.907164 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.37189 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.08092 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.96604 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.771044 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.00598 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.899165 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.07804 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.32983 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.831534 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.02289 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.27072 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.0878 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.23283 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.89267 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.19015 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.13657 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.72662 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.977076 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.868439 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.16975 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.02852 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0829999805680662 z_k1_s0000
                                 - 0.102999966170011 z_k1_s0001
                                 - 0.146999796254299 z_k1_s0002
                                 - 0.254999787624794 z_k1_s0003
                                 - 0.08800000576855 z_k1_s0004
                                 - 0.219999726421398 z_k1_s0005
                                 - 0.145999860334197 z_k1_s0006
                                 - 0.10999997841069 z_k1_s0007
                                 - 0.151999925946913 z_k1_s0010
                                 - 0.138999863371225 z_k1_s0011
                                 - 0.2109994774831 z_k1_s0012
                                 - 0.330000611844878 z_k1_s0013
                                 - 0.107999897267569 z_k1_s0014
                                 - 0.243999538667588 z_k1_s0015
                                 - 0.142999894832969 z_k1_s0016
                                 - 0.259999664514563 z_k1_s0017
                                 - 0.476999333251941 z_k1_s0020
                                 - 0.240999144415169 z_k1_s0021
                                 - 0.162999858612187 z_k1_s0022
                                 - 0.163999878827573 z_k1_s0023
                                 - 0.276999872169033 z_k1_s0024
                                 - 0.252000512282934 z_k1_s0025
                                 - 0.462000175519693 z_k1_s0026
                                 - 0.254999427430561 z_k1_s0027
                                 - 0.0820000112123491 z_k1_s0030
                                 - 0.193999741048612 z_k1_s0031
                                 - 0.282999650313193 z_k1_s0032
                                 - 0.169999857269437 z_k1_s0033
                                 - 0.0639999622779121 z_k1_s0034
                                 - 0.266999727793743 z_k1_s0035
                                 - 0.136999676822932 z_k1_s0036
                                 - 0.16400056682492 z_k1_s0037
                                 - 0.0950000265335173 z_k1_s0040
                                 - 0.0999999570352643 z_k1_s0041
                                 - 0.27400007085667 z_k1_s0042
                                 - 0.0870000569107026 z_k1_s0043
                                 - 0.196000670109299 z_k1_s0044
                                 - 0.233999549242434 z_k1_s0045
                                 - 0.278000661818223 z_k1_s0046
                                 - 0.0830000047492791 z_k1_s0047
                                 - 0.82399958339432 z_k1_s0050
                                 - 0.354999844054572 z_k1_s0051
                                 - 0.3279996646872 z_k1_s0052
                                 - 1.03199726558307 z_k1_s0053
                                 - 0.489001847721345 z_k1_s0054
                                 - 1.09000095539172 z_k1_s0055
                                 - 0.328999527656482 z_k1_s0056
                                 - 0.401999707305556 z_k1_s0057
                                 - 0.233999583760306 z_k1_s0060
                                 - 0.221000310384754 z_k1_s0061
                                 - 0.181999676258016 z_k1_s0062
                                 - 0.15499934547797 z_k1_s0063
                                 - 0.210999354095824 z_k1_s0064
                                 - 0.112999972164597 z_k1_s0065
                                 - 0.0680000305645378 z_k1_s0066
                                 - 0.222999657442516 z_k1_s0067
                                 - 0.30300006500529 z_k1_s0070
                                 - 0.775000928607965 z_k1_s0071
                                 - 0.631001169888753 z_k1_s0072
                                 - 0.352000301069623 z_k1_s0073
                                 - 0.449000838507214 z_k1_s0074
                                 - 0.718998477104799 z_k1_s0075
                                 - 0.348000180505138 z_k1_s0076
                                 - 0.241000296156803 z_k1_s0077
                                 - 0.0810001945581154 z_k1_s0100
                                 - 0.0800001796561333 z_k1_s0101
                                 - 0.187000199164431 z_k1_s0102
                                 - 0.229001050048219 z_k1_s0103
                                 - 0.0980002228912701 z_k1_s0104
                                 - 0.119000535832893 z_k1_s0105
                                 - 0.0650001811269428 z_k1_s0106
                                 - 0.266000138918042 z_k1_s0107
                                 - 0.0989997431524647 z_k1_s0110
                                 - 0.101000102812048 z_k1_s0111
                                 - 0.125000118727784 z_k1_s0112
                                 - 0.0590000319595182 z_k1_s0113
                                 - 0.174000330389047 z_k1_s0114
                                 - 0.121999579238409 z_k1_s0115
                                 - 0.112000360940051 z_k1_s0116
                                 - 0.0680000164279227 z_k1_s0117
                                 - 0.0549999661175266 z_k1_s0120
                                 - 0.108999729462094 z_k1_s0121
                                 - 0.0899999215741443 z_k1_s0122
                                 - 0.110000185034945 z_k1_s0123
                                 - 0.0580000056366468 z_k1_s0124
                                 - 0.114000111327159 z_k1_s0125
                                 - 0.0279999896452833 z_k1_s0126
                                 - 0.0680000109671012 z_k1_s0127
                                 - 0.0800002423467006 z_k1_s0130
                                 - 0.208000184900084 z_k1_s0131
                                 - 0.137000111592813 z_k1_s0132
                                 - 0.0990002694289504 z_k1_s0133
                                 - 0.123000649533884 z_k1_s0134
                                 - 0.170000130735584 z_k1_s0135
                                 - 0.157000251679727 z_k1_s0136
                                 - 0.0860002843727748 z_k1_s0137
                                 - 0.169000068886998 z_k1_s0140
                                 - 0.198998940195603 z_k1_s0141
                                 - 0.130999623968057 z_k1_s0142
                                 - 0.308998818649148 z_k1_s0143
                                 - 0.0679998518260341 z_k1_s0144
                                 - 0.148999482351708 z_k1_s0145
                                 - 0.178999692656533 z_k1_s0146
                                 - 0.135000276873603 z_k1_s0147
                                 - 0.340000255283182 z_k1_s0150
                                 - 0.117999937833723 z_k1_s0151
                                 - 0.540000268391071 z_k1_s0152
                                 - 0.133999799082492 z_k1_s0153
                                 - 0.302999891708578 z_k1_s0154
                                 - 0.226999793527672 z_k1_s0155
                                 - 0.435998490776712 z_k1_s0156
                                 - 0.231999735605426 z_k1_s0157
                                 - 0.113999767322292 z_k1_s0160
                                 - 0.0779998482468136 z_k1_s0161
                                 - 0.280999930407247 z_k1_s0162
                                 - 0.225000024845467 z_k1_s0163
                                 - 0.133999612308933 z_k1_s0164
                                 - 0.0859998145414022 z_k1_s0165
                                 - 0.284998710807429 z_k1_s0166
                                 - 0.211999036056766 z_k1_s0167
                                 - 0.126000148633839 z_k1_s0170
                                 - 0.31400070396106 z_k1_s0171
                                 - 0.205000908937091 z_k1_s0172
                                 - 0.153999848107483 z_k1_s0173
                                 - 0.142000234219196 z_k1_s0174
                                 - 0.313001036337524 z_k1_s0175
                                 - 0.163999526347722 z_k1_s0176
                                 - 0.0830001812986164 z_k1_s0177
                                 - 0.376001137376144 z_k1_s1000
                                 - 1.25900138321768 z_k1_s1001
                                 - 0.918004514235913 z_k1_s1002
                                 - 0.454000953516846 z_k1_s1003
                                 - 0.393001198487859 z_k1_s1004
                                 - 1.47900288831243 z_k1_s1005
                                 - 0.468001325250909 z_k1_s1006
                                 - 0.723999102877461 z_k1_s1007
                                 - 0.105000136139021 z_k1_s1010
                                 - 0.134000039819213 z_k1_s1011
                                 - 0.17499935889766 z_k1_s1012
                                 - 0.362999632411361 z_k1_s1013
                                 - 0.270000300471729 z_k1_s1014
                                 - 0.1880002735137 z_k1_s1015
                                 - 0.205000935395779 z_k1_s1016
                                 - 0.296000393702392 z_k1_s1017
                                 - 0.253000682395544 z_k1_s1020
                                 - 0.248001011328785 z_k1_s1021
                                 - 0.138000084451823 z_k1_s1022
                                 - 0.395000670300942 z_k1_s1023
                                 - 0.0840001383272847 z_k1_s1024
                                 - 0.217000051911843 z_k1_s1025
                                 - 0.149000193799536 z_k1_s1026
                                 - 0.182000688442844 z_k1_s1027
                                 - 0.18699969544111 z_k1_s1030
                                 - 0.240999402274836 z_k1_s1031
                                 - 0.148999546410774 z_k1_s1032
                                 - 0.615997366397934 z_k1_s1033
                                 - 0.289999313919852 z_k1_s1034
                                 - 0.156999591811865 z_k1_s1035
                                 - 0.356999927340968 z_k1_s1036
                                 - 0.667999393751111 z_k1_s1037
                                 - 0.218999711035846 z_k1_s1040
                                 - 0.191000201794537 z_k1_s1041
                                 - 0.186000224894201 z_k1_s1042
                                 - 0.0970000939673691 z_k1_s1043
                                 - 0.198999803434503 z_k1_s1044
                                 - 0.282000865582359 z_k1_s1045
                                 - 0.387000380489409 z_k1_s1046
                                 - 0.245999947698229 z_k1_s1047
                                 - 0.0830000408611399 z_k1_s1050
                                 - 0.197000442622702 z_k1_s1051
                                 - 0.322000425606238 z_k1_s1052
                                 - 0.147000123428837 z_k1_s1053
                                 - 0.263000863958604 z_k1_s1054
                                 - 0.398000910780913 z_k1_s1055
                                 - 0.13300019705478 z_k1_s1056
                                 - 0.0870000663966491 z_k1_s1057
                                 - 0.318999587151269 z_k1_s1060
                                 - 0.78199725515007 z_k1_s1061
                                 - 0.664999345960892 z_k1_s1062
                                 - 0.290999771342224 z_k1_s1063
                                 - 0.372999505268424 z_k1_s1064
                                 - 0.486998056805979 z_k1_s1065
                                 - 0.957000887451574 z_k1_s1066
                                 - 0.377999426820087 z_k1_s1067
                                 - 0.187000850271831 z_k1_s1070
                                 - 0.363000537350068 z_k1_s1071
                                 - 0.130000127274063 z_k1_s1072
                                 - 0.331000020789805 z_k1_s1073
                                 - 0.110000119824219 z_k1_s1074
                                 - 0.117000114546657 z_k1_s1075
                                 - 0.226999918970555 z_k1_s1076
                                 - 0.112000073259156 z_k1_s1077
                                 - 0.473998182663562 z_k1_s1100
                                 - 0.650997619164397 z_k1_s1101
                                 - 0.415000795827518 z_k1_s1102
                                 - 0.634000900209103 z_k1_s1103
                                 - 0.310999575122369 z_k1_s1104
                                 - 0.306999598133639 z_k1_s1105
                                 - 0.645999012177117 z_k1_s1106
                                 - 0.199999940835392 z_k1_s1107
                                 - 1.62599867823791 z_k1_s1110
                                 - 1.13600144557348 z_k1_s1111
                                 - 0.719002188063515 z_k1_s1112
                                 - 1.10199639674342 z_k1_s1113
                                 - 0.649998747898163 z_k1_s1114
                                 - 0.356999394796252 z_k1_s1115
                                 - 0.484999478184932 z_k1_s1116
                                 - 0.980001520318256 z_k1_s1117
                                 - 0.357999479034436 z_k1_s1120
                                 - 0.446999156805571 z_k1_s1121
                                 - 0.228000051890078 z_k1_s1122
                                 - 0.407999861782251 z_k1_s1123
                                 - 0.837998763013568 z_k1_s1124
                                 - 0.185999918541909 z_k1_s1125
                                 - 0.419000949506745 z_k1_s1126
                                 - 0.251999859289039 z_k1_s1127
                                 - 0.404999725742133 z_k1_s1130
                                 - 1.00300059598437 z_k1_s1131
                                 - 0.295999776435735 z_k1_s1132
                                 - 0.370999981173124 z_k1_s1133
                                 - 0.727998531147507 z_k1_s1134
                                 - 0.748001279717015 z_k1_s1135
                                 - 0.493001963449659 z_k1_s1136
                                 - 0.88399927239322 z_k1_s1137
                                 - 1.61600258098084 z_k1_s1140
                                 - 0.662000493103359 z_k1_s1141
                                 - 1.09600099369006 z_k1_s1142
                                 - 2.19500750076587 z_k1_s1143
                                 - 1.16900139238359 z_k1_s1144
                                 - 2.71300131776126 z_k1_s1145
                                 - 3.00900753391312 z_k1_s1146
                                 - 1.18900129594459 z_k1_s1147
                                 - 0.964000926557864 z_k1_s1150
                                 - 0.867996631325855 z_k1_s1151
                                 - 0.630998286360237 z_k1_s1152
                                 - 1.23899816620274 z_k1_s1153
                                 - 0.630998286360237 z_k1_s1154
                                 - 0.876000404673551 z_k1_s1155
                                 - 0.384999648497708 z_k1_s1156
                                 - 0.260999711523194 z_k1_s1157
                                 - 1.08300106846027 z_k1_s1160
                                 - 0.888002371387285 z_k1_s1161
                                 - 0.602999925575656 z_k1_s1162
                                 - 0.374001126127448 z_k1_s1163
                                 - 0.313000927333622 z_k1_s1164
                                 - 0.27400067381753 z_k1_s1165
                                 - 0.432001028596153 z_k1_s1166
                                 - 1.17400217409931 z_k1_s1167
                                 - 0.712001116335766 z_k1_s1170
                                 - 0.480000536585571 z_k1_s1171
                                 - 0.783001867656768 z_k1_s1172
                                 - 0.465000809067725 z_k1_s1173
                                 - 1.20400100773494 z_k1_s1174
                                 - 0.87900097817232 z_k1_s1175
                                 - 1.89800273584974 z_k1_s1176
                                 - 0.699001399953707 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.776415 w_k1_ADM_s000
                                 + 0.671169 w_k1_AXP_s000
                                 + 0.529353 w_k1_DIS_s000
                                 + 0.638594 w_k1_FDX_s000
                                 + 0.684228 w_k1_IBM_s000
                                 + 0.560671 w_k1_INTC_s000
                                 + 0.611928 w_k1_KO_s000
                                 + 0.80415 w_k1_MAT_s000
                                 + 0.748331 w_k1_MSFT_s000
                                 + 0.849205 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.93653 w_k1_ADM_s000
                                 + 1.56733 w_k1_AXP_s000
                                 + 1.25888 w_k1_DIS_s000
                                 + 1.14971 w_k1_FDX_s000
                                 + 1.31812 w_k1_IBM_s000
                                 + 1.09206 w_k1_INTC_s000 + 1.1047 w_k1_KO_s000
                                 + 1.44274 w_k1_MAT_s000
                                 + 1.07502 w_k1_MSFT_s000
                                 + 1.07568 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.09356 w_k1_ADM_s000
                                 + 0.946045 w_k1_AXP_s000
                                 + 0.851471 w_k1_DIS_s000
                                 + 0.85173 w_k1_FDX_s000
                                 + 1.10463 w_k1_IBM_s000
                                 + 0.855475 w_k1_INTC_s000
                                 + 0.894573 w_k1_KO_s000
                                 + 0.936639 w_k1_MAT_s000
                                 + 0.883831 w_k1_MSFT_s000
                                 + 1.04625 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.836099 w_k1_ADM_s000
                                 + 0.750962 w_k1_AXP_s000
                                 + 0.696111 w_k1_DIS_s000
                                 + 0.824321 w_k1_FDX_s000
                                 + 0.735183 w_k1_IBM_s000
                                 + 0.676825 w_k1_INTC_s000
                                 + 0.779641 w_k1_KO_s000
                                 + 0.964295 w_k1_MAT_s000
                                 + 0.801091 w_k1_MSFT_s000
                                 + 0.817339 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.21371 w_k1_ADM_s000
                                 + 1.23965 w_k1_AXP_s000
                                 + 1.54125 w_k1_DIS_s000 + 1.1947 w_k1_FDX_s000
                                 + 1.11884 w_k1_IBM_s000
                                 + 1.1576 w_k1_INTC_s000 + 1.07359 w_k1_KO_s000
                                 + 0.940206 w_k1_MAT_s000
                                 + 1.35389 w_k1_MSFT_s000
                                 + 1.05801 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.0824 w_k1_ADM_s000
                                 + 1.09159 w_k1_AXP_s000
                                 + 1.06102 w_k1_DIS_s000 + 1.0531 w_k1_FDX_s000
                                 + 1.04378 w_k1_IBM_s000
                                 + 1.00589 w_k1_INTC_s000
                                 + 1.11696 w_k1_KO_s000 + 1.00789 w_k1_MAT_s000
                                 + 1.06782 w_k1_MSFT_s000
                                 + 1.03563 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.15005 w_k1_ADM_s000
                                 + 0.994876 w_k1_AXP_s000
                                 + 1.04758 w_k1_DIS_s000
                                 + 1.02794 w_k1_FDX_s000
                                 + 0.707944 w_k1_IBM_s000
                                 + 0.964826 w_k1_INTC_s000
                                 + 0.877523 w_k1_KO_s000
                                 + 0.587308 w_k1_MAT_s000
                                 + 1.01897 w_k1_MSFT_s000
                                 + 0.968036 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.34963 w_k1_ADM_s000
                                 + 1.61608 w_k1_AXP_s000
                                 + 1.43479 w_k1_DIS_s000
                                 + 1.49709 w_k1_FDX_s000
                                 + 1.10408 w_k1_IBM_s000
                                 + 1.26264 w_k1_INTC_s000
                                 + 1.16211 w_k1_KO_s000 + 1.41247 w_k1_MAT_s000
                                 + 1.39959 w_k1_MSFT_s000 + 1.1259 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.01561 w_k1_ADM_s001
                                 + 0.976091 w_k1_AXP_s001
                                 + 0.787172 w_k1_DIS_s001
                                 + 0.79836 w_k1_FDX_s001
                                 + 1.34571 w_k1_IBM_s001
                                 + 0.908523 w_k1_INTC_s001
                                 + 0.901243 w_k1_KO_s001
                                 + 1.25873 w_k1_MAT_s001
                                 + 0.817541 w_k1_MSFT_s001
                                 + 0.991344 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.22469 w_k1_ADM_s001
                                 + 1.28036 w_k1_AXP_s001 + 1.0416 w_k1_DIS_s001
                                 + 1.07836 w_k1_FDX_s001
                                 + 1.04993 w_k1_IBM_s001
                                 + 1.07009 w_k1_INTC_s001
                                 + 0.911369 w_k1_KO_s001 + 1.0642 w_k1_MAT_s001
                                 + 1.03996 w_k1_MSFT_s001
                                 + 1.07329 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.872379 w_k1_ADM_s001
                                 + 0.846322 w_k1_AXP_s001
                                 + 1.05118 w_k1_DIS_s001 + 1.0358 w_k1_FDX_s001
                                 + 0.884833 w_k1_IBM_s001
                                 + 1.01464 w_k1_INTC_s001
                                 + 0.951382 w_k1_KO_s001
                                 + 1.09964 w_k1_MAT_s001
                                 + 1.18876 w_k1_MSFT_s001
                                 + 1.10821 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.954315 w_k1_ADM_s001
                                 + 0.765914 w_k1_AXP_s001
                                 + 0.77077 w_k1_DIS_s001
                                 + 0.888644 w_k1_FDX_s001
                                 + 0.774174 w_k1_IBM_s001
                                 + 1.06846 w_k1_INTC_s001
                                 + 0.801693 w_k1_KO_s001
                                 + 0.647867 w_k1_MAT_s001
                                 + 0.981559 w_k1_MSFT_s001
                                 + 0.954857 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.692968 w_k1_ADM_s001
                                 + 0.61288 w_k1_AXP_s001
                                 + 0.834279 w_k1_DIS_s001
                                 + 0.820462 w_k1_FDX_s001
                                 + 1.10445 w_k1_IBM_s001
                                 + 0.893745 w_k1_INTC_s001
                                 + 0.85817 w_k1_KO_s001
                                 + 0.994371 w_k1_MAT_s001
                                 + 1.00284 w_k1_MSFT_s001
                                 + 0.87707 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.849044 w_k1_ADM_s001
                                 + 1.25758 w_k1_AXP_s001
                                 + 1.29779 w_k1_DIS_s001
                                 + 1.10678 w_k1_FDX_s001
                                 + 1.40343 w_k1_IBM_s001
                                 + 1.13261 w_k1_INTC_s001
                                 + 1.22705 w_k1_KO_s001 + 1.39994 w_k1_MAT_s001
                                 + 1.12106 w_k1_MSFT_s001
                                 + 1.20141 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.18433 w_k1_ADM_s001
                                 + 1.42477 w_k1_AXP_s001
                                 + 1.62614 w_k1_DIS_s001
                                 + 1.38756 w_k1_FDX_s001
                                 + 1.24038 w_k1_IBM_s001
                                 + 1.4489 w_k1_INTC_s001 + 1.13835 w_k1_KO_s001
                                 + 1.01485 w_k1_MAT_s001
                                 + 1.31955 w_k1_MSFT_s001
                                 + 1.25455 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.909813 w_k1_ADM_s001
                                 + 0.925287 w_k1_AXP_s001
                                 + 0.776751 w_k1_DIS_s001
                                 + 0.860415 w_k1_FDX_s001
                                 + 0.946569 w_k1_IBM_s001
                                 + 0.818602 w_k1_INTC_s001
                                 + 0.856165 w_k1_KO_s001
                                 + 1.16331 w_k1_MAT_s001
                                 + 0.823304 w_k1_MSFT_s001
                                 + 0.982394 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.01555 w_k1_ADM_s002
                                 + 1.25859 w_k1_AXP_s002
                                 + 1.35243 w_k1_DIS_s002
                                 + 1.29924 w_k1_FDX_s002
                                 + 1.25433 w_k1_IBM_s002
                                 + 1.20529 w_k1_INTC_s002
                                 + 1.51255 w_k1_KO_s002 + 1.24289 w_k1_MAT_s002
                                 + 1.22263 w_k1_MSFT_s002
                                 + 1.29004 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.627566 w_k1_ADM_s002
                                 + 1.04966 w_k1_AXP_s002
                                 + 1.09407 w_k1_DIS_s002
                                 + 0.986509 w_k1_FDX_s002
                                 + 1.34458 w_k1_IBM_s002
                                 + 1.20553 w_k1_INTC_s002
                                 + 1.22466 w_k1_KO_s002 + 1.22804 w_k1_MAT_s002
                                 + 1.0929 w_k1_MSFT_s002 + 1.03925 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.675258 w_k1_ADM_s002
                                 + 0.816032 w_k1_AXP_s002
                                 + 0.704024 w_k1_DIS_s002
                                 + 0.806628 w_k1_FDX_s002
                                 + 1.04105 w_k1_IBM_s002
                                 + 0.842364 w_k1_INTC_s002
                                 + 0.997264 w_k1_KO_s002 + 1.1194 w_k1_MAT_s002
                                 + 0.7603 w_k1_MSFT_s002 + 0.88981 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.913861 w_k1_ADM_s002
                                 + 1.17648 w_k1_AXP_s002
                                 + 0.828177 w_k1_DIS_s002
                                 + 0.922944 w_k1_FDX_s002
                                 + 1.03235 w_k1_IBM_s002
                                 + 1.0942 w_k1_INTC_s002 + 1.15157 w_k1_KO_s002
                                 + 1.34464 w_k1_MAT_s002
                                 + 0.846282 w_k1_MSFT_s002
                                 + 0.96832 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.739019 w_k1_ADM_s002
                                 + 0.947117 w_k1_AXP_s002
                                 + 0.763043 w_k1_DIS_s002
                                 + 0.895505 w_k1_FDX_s002
                                 + 0.746981 w_k1_IBM_s002
                                 + 0.925246 w_k1_INTC_s002
                                 + 0.876705 w_k1_KO_s002
                                 + 0.722098 w_k1_MAT_s002
                                 + 0.808276 w_k1_MSFT_s002
                                 + 0.946748 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.714778 w_k1_ADM_s002
                                 + 0.664441 w_k1_AXP_s002
                                 + 0.666594 w_k1_DIS_s002
                                 + 0.771456 w_k1_FDX_s002
                                 + 0.736049 w_k1_IBM_s002
                                 + 0.786084 w_k1_INTC_s002
                                 + 0.859008 w_k1_KO_s002
                                 + 0.756922 w_k1_MAT_s002
                                 + 0.842132 w_k1_MSFT_s002
                                 + 0.833659 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.99822 w_k1_ADM_s002
                                 + 1.36975 w_k1_AXP_s002
                                 + 1.13366 w_k1_DIS_s002
                                 + 1.16151 w_k1_FDX_s002
                                 + 0.88595 w_k1_IBM_s002
                                 + 1.13883 w_k1_INTC_s002
                                 + 1.21286 w_k1_KO_s002 + 1.14822 w_k1_MAT_s002
                                 + 1.07489 w_k1_MSFT_s002
                                 + 1.13953 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.810393 w_k1_ADM_s002
                                 + 0.650848 w_k1_AXP_s002
                                 + 1.01566 w_k1_DIS_s002
                                 + 0.98354 w_k1_FDX_s002
                                 + 0.913543 w_k1_IBM_s002
                                 + 0.972911 w_k1_INTC_s002
                                 + 1.0513 w_k1_KO_s002 + 0.806807 w_k1_MAT_s002
                                 + 1.07528 w_k1_MSFT_s002
                                 + 1.09873 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.860743 w_k1_ADM_s003
                                 + 0.903602 w_k1_AXP_s003
                                 + 0.898464 w_k1_DIS_s003
                                 + 0.905708 w_k1_FDX_s003
                                 + 1.34497 w_k1_IBM_s003
                                 + 0.892394 w_k1_INTC_s003
                                 + 1.13414 w_k1_KO_s003 + 1.3644 w_k1_MAT_s003
                                 + 0.927807 w_k1_MSFT_s003
                                 + 1.05692 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.827303 w_k1_ADM_s003
                                 + 0.57757 w_k1_AXP_s003
                                 + 0.681044 w_k1_DIS_s003
                                 + 0.742385 w_k1_FDX_s003
                                 + 0.889167 w_k1_IBM_s003
                                 + 0.81635 w_k1_INTC_s003
                                 + 0.891142 w_k1_KO_s003
                                 + 0.777231 w_k1_MAT_s003
                                 + 0.906069 w_k1_MSFT_s003
                                 + 0.887621 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.06141 w_k1_ADM_s003
                                 + 0.979701 w_k1_AXP_s003
                                 + 1.13953 w_k1_DIS_s003
                                 + 1.02965 w_k1_FDX_s003
                                 + 0.910213 w_k1_IBM_s003
                                 + 0.976367 w_k1_INTC_s003
                                 + 0.976942 w_k1_KO_s003
                                 + 1.17395 w_k1_MAT_s003
                                 + 1.20669 w_k1_MSFT_s003 + 0.9605 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.4207 w_k1_ADM_s003
                                 + 1.4205 w_k1_AXP_s003 + 1.14856 w_k1_DIS_s003
                                 + 1.2237 w_k1_FDX_s003 + 0.9554 w_k1_IBM_s003
                                 + 0.965187 w_k1_INTC_s003
                                 + 0.885168 w_k1_KO_s003 + 1.2153 w_k1_MAT_s003
                                 + 0.989992 w_k1_MSFT_s003
                                 + 0.933739 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.32953 w_k1_ADM_s003
                                 + 1.42896 w_k1_AXP_s003
                                 + 1.17569 w_k1_DIS_s003
                                 + 1.42051 w_k1_FDX_s003
                                 + 0.764658 w_k1_IBM_s003
                                 + 1.22582 w_k1_INTC_s003
                                 + 0.938184 w_k1_KO_s003 + 1.0843 w_k1_MAT_s003
                                 + 1.20989 w_k1_MSFT_s003
                                 + 1.23036 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.900951 w_k1_ADM_s003
                                 + 0.974523 w_k1_AXP_s003
                                 + 0.861217 w_k1_DIS_s003
                                 + 0.906378 w_k1_FDX_s003
                                 + 0.91597 w_k1_IBM_s003
                                 + 0.853622 w_k1_INTC_s003
                                 + 0.894437 w_k1_KO_s003
                                 + 1.15513 w_k1_MAT_s003
                                 + 0.880232 w_k1_MSFT_s003
                                 + 0.947356 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.778622 w_k1_ADM_s003
                                 + 0.98182 w_k1_AXP_s003
                                 + 1.38924 w_k1_DIS_s003
                                 + 1.01739 w_k1_FDX_s003
                                 + 1.46314 w_k1_IBM_s003
                                 + 1.12847 w_k1_INTC_s003 + 1.215 w_k1_KO_s003
                                 + 1.30409 w_k1_MAT_s003
                                 + 1.11341 w_k1_MSFT_s003
                                 + 1.03704 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.38736 w_k1_ADM_s003
                                 + 1.69942 w_k1_AXP_s003
                                 + 1.67042 w_k1_DIS_s003
                                 + 1.36017 w_k1_FDX_s003
                                 + 1.17128 w_k1_IBM_s003
                                 + 1.22918 w_k1_INTC_s003
                                 + 1.28023 w_k1_KO_s003 + 1.68789 w_k1_MAT_s003
                                 + 1.31735 w_k1_MSFT_s003
                                 + 1.34328 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.925846 w_k1_ADM_s004
                                 + 0.971075 w_k1_AXP_s004
                                 + 0.912198 w_k1_DIS_s004
                                 + 0.994026 w_k1_FDX_s004
                                 + 1.06555 w_k1_IBM_s004
                                 + 1.03016 w_k1_INTC_s004
                                 + 0.907087 w_k1_KO_s004
                                 + 1.17843 w_k1_MAT_s004
                                 + 0.981497 w_k1_MSFT_s004
                                 + 1.01231 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.31598 w_k1_ADM_s004
                                 + 0.934721 w_k1_AXP_s004
                                 + 0.890027 w_k1_DIS_s004
                                 + 1.02062 w_k1_FDX_s004
                                 + 1.16958 w_k1_IBM_s004
                                 + 0.973005 w_k1_INTC_s004
                                 + 1.04259 w_k1_KO_s004 + 1.16591 w_k1_MAT_s004
                                 + 1.01131 w_k1_MSFT_s004
                                 + 1.20415 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.845472 w_k1_ADM_s004
                                 + 0.985967 w_k1_AXP_s004
                                 + 0.903961 w_k1_DIS_s004
                                 + 1.08173 w_k1_FDX_s004
                                 + 0.795366 w_k1_IBM_s004
                                 + 1.10398 w_k1_INTC_s004
                                 + 0.798489 w_k1_KO_s004
                                 + 0.773453 w_k1_MAT_s004
                                 + 1.05378 w_k1_MSFT_s004
                                 + 1.00091 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.7103 w_k1_ADM_s004
                                 + 1.3802 w_k1_AXP_s004 + 1.00445 w_k1_DIS_s004
                                 + 1.06154 w_k1_FDX_s004
                                 + 1.77575 w_k1_IBM_s004
                                 + 1.23546 w_k1_INTC_s004
                                 + 1.26981 w_k1_KO_s004 + 2.66419 w_k1_MAT_s004
                                 + 0.981004 w_k1_MSFT_s004
                                 + 1.22232 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.827269 w_k1_ADM_s004
                                 + 1.12294 w_k1_AXP_s004
                                 + 1.18284 w_k1_DIS_s004
                                 + 1.22111 w_k1_FDX_s004
                                 + 0.948167 w_k1_IBM_s004
                                 + 1.16652 w_k1_INTC_s004
                                 + 1.02084 w_k1_KO_s004 + 1.13216 w_k1_MAT_s004
                                 + 1.14859 w_k1_MSFT_s004
                                 + 1.06542 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.03086 w_k1_ADM_s004
                                 + 1.121 w_k1_AXP_s004 + 1.17956 w_k1_DIS_s004
                                 + 1.20402 w_k1_FDX_s004 + 1.383 w_k1_IBM_s004
                                 + 0.994934 w_k1_INTC_s004
                                 + 1.08896 w_k1_KO_s004 + 1.37418 w_k1_MAT_s004
                                 + 1.05993 w_k1_MSFT_s004
                                 + 1.17396 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.752389 w_k1_ADM_s004
                                 + 0.712207 w_k1_AXP_s004
                                 + 0.647268 w_k1_DIS_s004
                                 + 0.799215 w_k1_FDX_s004
                                 + 0.877168 w_k1_IBM_s004
                                 + 0.71858 w_k1_INTC_s004
                                 + 0.731847 w_k1_KO_s004
                                 + 0.925189 w_k1_MAT_s004
                                 + 0.779368 w_k1_MSFT_s004
                                 + 0.797802 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.872561 w_k1_ADM_s004
                                 + 0.747112 w_k1_AXP_s004
                                 + 0.652664 w_k1_DIS_s004
                                 + 0.967854 w_k1_FDX_s004
                                 + 0.591318 w_k1_IBM_s004
                                 + 1.14863 w_k1_INTC_s004
                                 + 0.811781 w_k1_KO_s004
                                 + 0.392134 w_k1_MAT_s004
                                 + 1.05949 w_k1_MSFT_s004
                                 + 1.01347 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.15332 w_k1_ADM_s005
                                 + 1.17374 w_k1_AXP_s005
                                 + 1.13888 w_k1_DIS_s005
                                 + 1.08121 w_k1_FDX_s005
                                 + 0.978724 w_k1_IBM_s005
                                 + 0.926594 w_k1_INTC_s005
                                 + 0.924904 w_k1_KO_s005
                                 + 0.607593 w_k1_MAT_s005
                                 + 0.941709 w_k1_MSFT_s005
                                 + 0.920951 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.07616 w_k1_ADM_s005
                                 + 0.906205 w_k1_AXP_s005
                                 + 1.36434 w_k1_DIS_s005
                                 + 1.17458 w_k1_FDX_s005
                                 + 0.781736 w_k1_IBM_s005
                                 + 1.25001 w_k1_INTC_s005
                                 + 1.04892 w_k1_KO_s005 + 0.65787 w_k1_MAT_s005
                                 + 1.24939 w_k1_MSFT_s005
                                 + 1.04885 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.928909 w_k1_ADM_s005
                                 + 1.14596 w_k1_AXP_s005
                                 + 1.65292 w_k1_DIS_s005
                                 + 1.32676 w_k1_FDX_s005
                                 + 1.17012 w_k1_IBM_s005
                                 + 1.40946 w_k1_INTC_s005
                                 + 1.20896 w_k1_KO_s005
                                 + 0.997515 w_k1_MAT_s005
                                 + 1.47729 w_k1_MSFT_s005
                                 + 1.06046 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.67434 w_k1_ADM_s005
                                 + 0.681998 w_k1_AXP_s005
                                 + 0.809132 w_k1_DIS_s005
                                 + 0.753959 w_k1_FDX_s005
                                 + 1.09205 w_k1_IBM_s005
                                 + 0.684863 w_k1_INTC_s005
                                 + 0.859132 w_k1_KO_s005
                                 + 0.796746 w_k1_MAT_s005
                                 + 0.837391 w_k1_MSFT_s005
                                 + 0.918586 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.06833 w_k1_ADM_s005
                                 + 1.14291 w_k1_AXP_s005
                                 + 1.03952 w_k1_DIS_s005
                                 + 0.990381 w_k1_FDX_s005
                                 + 1.67305 w_k1_IBM_s005
                                 + 0.935865 w_k1_INTC_s005
                                 + 1.19368 w_k1_KO_s005 + 1.47904 w_k1_MAT_s005
                                 + 0.919189 w_k1_MSFT_s005
                                 + 1.00017 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.961376 w_k1_ADM_s005
                                 + 0.756032 w_k1_AXP_s005
                                 + 0.782814 w_k1_DIS_s005
                                 + 0.871969 w_k1_FDX_s005
                                 + 0.65635 w_k1_IBM_s005
                                 + 1.00206 w_k1_INTC_s005
                                 + 0.772181 w_k1_KO_s005
                                 + 0.467898 w_k1_MAT_s005
                                 + 0.913823 w_k1_MSFT_s005
                                 + 0.885226 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.15837 w_k1_ADM_s005
                                 + 1.02723 w_k1_AXP_s005
                                 + 1.11015 w_k1_DIS_s005
                                 + 1.25624 w_k1_FDX_s005
                                 + 0.801079 w_k1_IBM_s005
                                 + 1.01797 w_k1_INTC_s005 + 1.0097 w_k1_KO_s005
                                 + 0.759368 w_k1_MAT_s005
                                 + 1.28871 w_k1_MSFT_s005
                                 + 0.973645 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.41904 w_k1_ADM_s005
                                 + 1.67591 w_k1_AXP_s005
                                 + 1.85933 w_k1_DIS_s005
                                 + 1.43118 w_k1_FDX_s005
                                 + 1.53436 w_k1_IBM_s005
                                 + 1.25119 w_k1_INTC_s005
                                 + 1.36718 w_k1_KO_s005
                                 + 0.984224 w_k1_MAT_s005
                                 + 1.36724 w_k1_MSFT_s005
                                 + 1.11751 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.987831 w_k1_ADM_s006
                                 + 0.856757 w_k1_AXP_s006
                                 + 0.868715 w_k1_DIS_s006
                                 + 0.816257 w_k1_FDX_s006
                                 + 1.05711 w_k1_IBM_s006
                                 + 0.858638 w_k1_INTC_s006
                                 + 1.03205 w_k1_KO_s006
                                 + 0.945705 w_k1_MAT_s006
                                 + 0.92643 w_k1_MSFT_s006
                                 + 0.786454 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.13431 w_k1_ADM_s006
                                 + 1.09104 w_k1_AXP_s006
                                 + 1.19366 w_k1_DIS_s006
                                 + 1.00709 w_k1_FDX_s006
                                 + 1.31418 w_k1_IBM_s006
                                 + 1.09074 w_k1_INTC_s006
                                 + 1.14489 w_k1_KO_s006 + 1.07604 w_k1_MAT_s006
                                 + 1.0808 w_k1_MSFT_s006 + 1.00604 w_k1_PG_s006
                                 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.17858 w_k1_ADM_s006
                                 + 1.33503 w_k1_AXP_s006
                                 + 1.22159 w_k1_DIS_s006
                                 + 1.23119 w_k1_FDX_s006
                                 + 0.936479 w_k1_IBM_s006
                                 + 1.10467 w_k1_INTC_s006
                                 + 1.16076 w_k1_KO_s006 + 1.22012 w_k1_MAT_s006
                                 + 1.08852 w_k1_MSFT_s006
                                 + 1.12066 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.4087 w_k1_ADM_s006
                                 + 1.47791 w_k1_AXP_s006
                                 + 1.78984 w_k1_DIS_s006
                                 + 1.31148 w_k1_FDX_s006
                                 + 1.35865 w_k1_IBM_s006
                                 + 1.17311 w_k1_INTC_s006 + 1.267 w_k1_KO_s006
                                 + 1.65166 w_k1_MAT_s006
                                 + 1.29124 w_k1_MSFT_s006
                                 + 1.18573 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.02412 w_k1_ADM_s006
                                 + 0.694055 w_k1_AXP_s006
                                 + 0.719556 w_k1_DIS_s006
                                 + 0.803418 w_k1_FDX_s006
                                 + 0.741111 w_k1_IBM_s006
                                 + 1.00101 w_k1_INTC_s006
                                 + 0.739116 w_k1_KO_s006
                                 + 0.469481 w_k1_MAT_s006
                                 + 0.836053 w_k1_MSFT_s006
                                 + 0.847821 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.970163 w_k1_ADM_s006
                                 + 0.797479 w_k1_AXP_s006
                                 + 1.02085 w_k1_DIS_s006
                                 + 0.783678 w_k1_FDX_s006
                                 + 1.2781 w_k1_IBM_s006
                                 + 0.845947 w_k1_INTC_s006
                                 + 1.03605 w_k1_KO_s006 + 1.31153 w_k1_MAT_s006
                                 + 1.00972 w_k1_MSFT_s006
                                 + 0.97888 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.27794 w_k1_ADM_s006
                                 + 1.12526 w_k1_AXP_s006
                                 + 0.906208 w_k1_DIS_s006
                                 + 0.959752 w_k1_FDX_s006
                                 + 0.857094 w_k1_IBM_s006
                                 + 0.85997 w_k1_INTC_s006
                                 + 0.807596 w_k1_KO_s006
                                 + 0.837057 w_k1_MAT_s006
                                 + 0.848452 w_k1_MSFT_s006
                                 + 0.900809 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.21521 w_k1_ADM_s006
                                 + 0.97841 w_k1_AXP_s006
                                 + 1.10245 w_k1_DIS_s006 + 1.0383 w_k1_FDX_s006
                                 + 0.933315 w_k1_IBM_s006
                                 + 0.837243 w_k1_INTC_s006
                                 + 0.844715 w_k1_KO_s006
                                 + 0.710084 w_k1_MAT_s006
                                 + 1.0244 w_k1_MSFT_s006
                                 + 0.933619 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.16616 w_k1_ADM_s007
                                 + 1.29536 w_k1_AXP_s007
                                 + 1.27998 w_k1_DIS_s007
                                 + 1.26108 w_k1_FDX_s007
                                 + 1.01696 w_k1_IBM_s007
                                 + 1.18537 w_k1_INTC_s007
                                 + 1.13313 w_k1_KO_s007 + 1.17155 w_k1_MAT_s007
                                 + 1.29501 w_k1_MSFT_s007
                                 + 1.11526 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.727367 w_k1_ADM_s007
                                 + 0.99368 w_k1_AXP_s007
                                 + 0.969154 w_k1_DIS_s007
                                 + 0.863051 w_k1_FDX_s007
                                 + 1.35889 w_k1_IBM_s007
                                 + 0.997151 w_k1_INTC_s007
                                 + 1.14136 w_k1_KO_s007 + 1.63206 w_k1_MAT_s007
                                 + 1.04146 w_k1_MSFT_s007
                                 + 0.951796 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.891081 w_k1_ADM_s007
                                 + 0.904917 w_k1_AXP_s007
                                 + 0.932603 w_k1_DIS_s007
                                 + 0.997347 w_k1_FDX_s007
                                 + 0.913459 w_k1_IBM_s007
                                 + 0.981988 w_k1_INTC_s007
                                 + 0.965086 w_k1_KO_s007
                                 + 0.895973 w_k1_MAT_s007
                                 + 1.1433 w_k1_MSFT_s007
                                 + 0.941158 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.19335 w_k1_ADM_s007
                                 + 1.79777 w_k1_AXP_s007
                                 + 1.63777 w_k1_DIS_s007
                                 + 1.42229 w_k1_FDX_s007
                                 + 1.29772 w_k1_IBM_s007
                                 + 1.28654 w_k1_INTC_s007
                                 + 1.46829 w_k1_KO_s007 + 1.54683 w_k1_MAT_s007
                                 + 1.33575 w_k1_MSFT_s007
                                 + 1.21054 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.767268 w_k1_ADM_s007
                                 + 0.996034 w_k1_AXP_s007
                                 + 0.754809 w_k1_DIS_s007
                                 + 0.89084 w_k1_FDX_s007
                                 + 0.752697 w_k1_IBM_s007
                                 + 0.946817 w_k1_INTC_s007
                                 + 0.724965 w_k1_KO_s007
                                 + 0.710715 w_k1_MAT_s007
                                 + 0.944231 w_k1_MSFT_s007
                                 + 0.87144 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.615177 w_k1_ADM_s007
                                 + 0.70771 w_k1_AXP_s007
                                 + 0.689305 w_k1_DIS_s007
                                 + 0.707728 w_k1_FDX_s007
                                 + 0.882471 w_k1_IBM_s007
                                 + 0.955086 w_k1_INTC_s007
                                 + 0.887211 w_k1_KO_s007
                                 + 0.768829 w_k1_MAT_s007
                                 + 0.868956 w_k1_MSFT_s007
                                 + 0.755271 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.852164 w_k1_ADM_s007
                                 + 1.08379 w_k1_AXP_s007
                                 + 1.45505 w_k1_DIS_s007
                                 + 1.11646 w_k1_FDX_s007
                                 + 1.29416 w_k1_IBM_s007
                                 + 1.37266 w_k1_INTC_s007
                                 + 1.29604 w_k1_KO_s007 + 1.39074 w_k1_MAT_s007
                                 + 1.34089 w_k1_MSFT_s007 + 1.0646 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.952233 w_k1_ADM_s007
                                 + 1.08219 w_k1_AXP_s007
                                 + 1.02985 w_k1_DIS_s007 + 1.1344 w_k1_FDX_s007
                                 + 0.784287 w_k1_IBM_s007
                                 + 0.992124 w_k1_INTC_s007
                                 + 0.933608 w_k1_KO_s007
                                 + 0.722443 w_k1_MAT_s007
                                 + 1.09563 w_k1_MSFT_s007 + 1.0042 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.980753 w_k1_ADM_s010
                                 + 0.852122 w_k1_AXP_s010
                                 + 0.814243 w_k1_DIS_s010
                                 + 0.907978 w_k1_FDX_s010
                                 + 1.45492 w_k1_IBM_s010
                                 + 1.03031 w_k1_INTC_s010
                                 + 1.12544 w_k1_KO_s010 + 1.16334 w_k1_MAT_s010
                                 + 1.0994 w_k1_MSFT_s010
                                 + 0.961523 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.14334 w_k1_ADM_s010
                                 + 0.799686 w_k1_AXP_s010
                                 + 1.09575 w_k1_DIS_s010
                                 + 1.07662 w_k1_FDX_s010
                                 + 0.766128 w_k1_IBM_s010
                                 + 1.06955 w_k1_INTC_s010
                                 + 1.09225 w_k1_KO_s010
                                 + 0.587309 w_k1_MAT_s010
                                 + 1.18446 w_k1_MSFT_s010
                                 + 1.12254 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.931256 w_k1_ADM_s010
                                 + 1.14029 w_k1_AXP_s010
                                 + 0.808151 w_k1_DIS_s010
                                 + 0.958327 w_k1_FDX_s010
                                 + 1.11802 w_k1_IBM_s010
                                 + 0.932253 w_k1_INTC_s010
                                 + 1.00583 w_k1_KO_s010 + 1.09005 w_k1_MAT_s010
                                 + 0.953902 w_k1_MSFT_s010
                                 + 0.972387 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.22351 w_k1_ADM_s010
                                 + 1.39441 w_k1_AXP_s010
                                 + 1.30482 w_k1_DIS_s010
                                 + 1.32272 w_k1_FDX_s010
                                 + 1.12716 w_k1_IBM_s010
                                 + 1.08224 w_k1_INTC_s010 + 1.2372 w_k1_KO_s010
                                 + 1.10384 w_k1_MAT_s010
                                 + 1.37622 w_k1_MSFT_s010
                                 + 1.20546 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.990828 w_k1_ADM_s010
                                 + 1.07982 w_k1_AXP_s010
                                 + 1.50916 w_k1_DIS_s010
                                 + 1.10336 w_k1_FDX_s010
                                 + 1.42932 w_k1_IBM_s010
                                 + 1.3316 w_k1_INTC_s010 + 1.6039 w_k1_KO_s010
                                 + 0.86027 w_k1_MAT_s010
                                 + 1.32319 w_k1_MSFT_s010
                                 + 1.22414 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.863591 w_k1_ADM_s010
                                 + 0.91386 w_k1_AXP_s010
                                 + 0.660477 w_k1_DIS_s010
                                 + 0.855606 w_k1_FDX_s010
                                 + 0.943121 w_k1_IBM_s010
                                 + 0.754594 w_k1_INTC_s010
                                 + 0.8663 w_k1_KO_s010 + 0.765909 w_k1_MAT_s010
                                 + 0.799316 w_k1_MSFT_s010
                                 + 0.954186 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.887543 w_k1_ADM_s010
                                 + 0.683951 w_k1_AXP_s010
                                 + 0.647455 w_k1_DIS_s010
                                 + 0.76556 w_k1_FDX_s010
                                 + 0.639783 w_k1_IBM_s010
                                 + 0.792564 w_k1_INTC_s010
                                 + 0.773839 w_k1_KO_s010
                                 + 0.500304 w_k1_MAT_s010
                                 + 0.819808 w_k1_MSFT_s010
                                 + 0.98509 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.788302 w_k1_ADM_s010
                                 + 0.702981 w_k1_AXP_s010
                                 + 0.903155 w_k1_DIS_s010
                                 + 0.883416 w_k1_FDX_s010
                                 + 0.931468 w_k1_IBM_s010
                                 + 0.917518 w_k1_INTC_s010
                                 + 1.06169 w_k1_KO_s010
                                 + 0.756948 w_k1_MAT_s010
                                 + 1.07177 w_k1_MSFT_s010
                                 + 0.987776 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.13628 w_k1_ADM_s011
                                 + 1.37509 w_k1_AXP_s011
                                 + 1.20981 w_k1_DIS_s011
                                 + 1.07493 w_k1_FDX_s011
                                 + 1.56261 w_k1_IBM_s011
                                 + 1.17337 w_k1_INTC_s011
                                 + 1.14759 w_k1_KO_s011 + 1.28592 w_k1_MAT_s011
                                 + 0.962184 w_k1_MSFT_s011
                                 + 1.08174 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.10135 w_k1_ADM_s011
                                 + 1.01901 w_k1_AXP_s011
                                 + 1.03575 w_k1_DIS_s011 + 1.0006 w_k1_FDX_s011
                                 + 0.777307 w_k1_IBM_s011
                                 + 1.1188 w_k1_INTC_s011
                                 + 0.753512 w_k1_KO_s011
                                 + 0.854431 w_k1_MAT_s011
                                 + 0.947649 w_k1_MSFT_s011
                                 + 0.851032 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.3262 w_k1_ADM_s011
                                 + 1.22113 w_k1_AXP_s011
                                 + 1.19438 w_k1_DIS_s011
                                 + 1.19033 w_k1_FDX_s011
                                 + 0.917962 w_k1_IBM_s011
                                 + 1.25113 w_k1_INTC_s011
                                 + 0.922999 w_k1_KO_s011
                                 + 0.811567 w_k1_MAT_s011
                                 + 1.20792 w_k1_MSFT_s011
                                 + 1.02026 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.653282 w_k1_ADM_s011
                                 + 0.748913 w_k1_AXP_s011
                                 + 0.695256 w_k1_DIS_s011
                                 + 0.667275 w_k1_FDX_s011
                                 + 0.985618 w_k1_IBM_s011
                                 + 0.779252 w_k1_INTC_s011
                                 + 0.725135 w_k1_KO_s011
                                 + 0.912643 w_k1_MAT_s011
                                 + 0.733016 w_k1_MSFT_s011
                                 + 0.820028 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.858086 w_k1_ADM_s011
                                 + 0.984164 w_k1_AXP_s011
                                 + 0.932705 w_k1_DIS_s011
                                 + 1.00892 w_k1_FDX_s011
                                 + 0.937721 w_k1_IBM_s011
                                 + 0.979193 w_k1_INTC_s011
                                 + 0.794618 w_k1_KO_s011
                                 + 0.997154 w_k1_MAT_s011
                                 + 0.964453 w_k1_MSFT_s011
                                 + 0.851754 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.35306 w_k1_ADM_s011
                                 + 1.20003 w_k1_AXP_s011
                                 + 1.65015 w_k1_DIS_s011
                                 + 1.28026 w_k1_FDX_s011
                                 + 1.30429 w_k1_IBM_s011
                                 + 1.48529 w_k1_INTC_s011
                                 + 1.28402 w_k1_KO_s011 + 1.28702 w_k1_MAT_s011
                                 + 1.40637 w_k1_MSFT_s011
                                 + 1.21657 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.930347 w_k1_ADM_s011
                                 + 1.03551 w_k1_AXP_s011
                                 + 0.862469 w_k1_DIS_s011
                                 + 0.843712 w_k1_FDX_s011
                                 + 1.24572 w_k1_IBM_s011
                                 + 1.07661 w_k1_INTC_s011
                                 + 0.904043 w_k1_KO_s011
                                 + 1.20494 w_k1_MAT_s011
                                 + 0.879924 w_k1_MSFT_s011
                                 + 0.865746 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.777426 w_k1_ADM_s011
                                 + 0.608366 w_k1_AXP_s011
                                 + 0.679278 w_k1_DIS_s011
                                 + 0.772169 w_k1_FDX_s011
                                 + 0.747141 w_k1_IBM_s011
                                 + 1.03304 w_k1_INTC_s011
                                 + 0.82149 w_k1_KO_s011 + 0.76948 w_k1_MAT_s011
                                 + 0.929078 w_k1_MSFT_s011
                                 + 0.911666 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.15102 w_k1_ADM_s012
                                 + 1.11243 w_k1_AXP_s012
                                 + 1.58284 w_k1_DIS_s012
                                 + 1.21209 w_k1_FDX_s012
                                 + 1.24062 w_k1_IBM_s012
                                 + 1.08897 w_k1_INTC_s012
                                 + 1.30913 w_k1_KO_s012 + 1.09602 w_k1_MAT_s012
                                 + 1.1158 w_k1_MSFT_s012 + 1.15976 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.983613 w_k1_ADM_s012
                                 + 0.810816 w_k1_AXP_s012
                                 + 0.764028 w_k1_DIS_s012
                                 + 0.903474 w_k1_FDX_s012
                                 + 0.755749 w_k1_IBM_s012
                                 + 0.806864 w_k1_INTC_s012
                                 + 0.722641 w_k1_KO_s012
                                 + 0.490394 w_k1_MAT_s012
                                 + 0.76508 w_k1_MSFT_s012
                                 + 0.924362 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.30305 w_k1_ADM_s012
                                 + 1.193 w_k1_AXP_s012 + 1.12498 w_k1_DIS_s012
                                 + 1.18922 w_k1_FDX_s012 + 1.181 w_k1_IBM_s012
                                 + 1.02438 w_k1_INTC_s012
                                 + 1.21581 w_k1_KO_s012 + 1.19069 w_k1_MAT_s012
                                 + 0.972775 w_k1_MSFT_s012
                                 + 1.17243 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.03934 w_k1_ADM_s012
                                 + 1.05414 w_k1_AXP_s012
                                 + 0.988838 w_k1_DIS_s012
                                 + 1.09114 w_k1_FDX_s012
                                 + 0.823688 w_k1_IBM_s012
                                 + 1.1888 w_k1_INTC_s012 + 1.0004 w_k1_KO_s012
                                 + 0.821814 w_k1_MAT_s012
                                 + 0.982634 w_k1_MSFT_s012
                                 + 1.1553 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.65004 w_k1_ADM_s012
                                 + 0.910087 w_k1_AXP_s012
                                 + 0.568667 w_k1_DIS_s012
                                 + 0.750382 w_k1_FDX_s012
                                 + 0.807078 w_k1_IBM_s012
                                 + 0.822086 w_k1_INTC_s012
                                 + 0.726484 w_k1_KO_s012
                                 + 0.872105 w_k1_MAT_s012
                                 + 0.677646 w_k1_MSFT_s012
                                 + 0.809972 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.749614 w_k1_ADM_s012
                                 + 0.823767 w_k1_AXP_s012
                                 + 0.773254 w_k1_DIS_s012
                                 + 0.86578 w_k1_FDX_s012
                                 + 1.03665 w_k1_IBM_s012
                                 + 0.874691 w_k1_INTC_s012
                                 + 0.856661 w_k1_KO_s012
                                 + 0.84744 w_k1_MAT_s012
                                 + 0.833553 w_k1_MSFT_s012
                                 + 1.00081 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.874062 w_k1_ADM_s012
                                 + 1.51426 w_k1_AXP_s012
                                 + 0.926827 w_k1_DIS_s012
                                 + 1.01174 w_k1_FDX_s012
                                 + 1.09858 w_k1_IBM_s012
                                 + 0.839285 w_k1_INTC_s012
                                 + 0.905672 w_k1_KO_s012
                                 + 1.19429 w_k1_MAT_s012
                                 + 0.665013 w_k1_MSFT_s012
                                 + 1.11378 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.49833 w_k1_ADM_s012
                                 + 1.12721 w_k1_AXP_s012
                                 + 1.40875 w_k1_DIS_s012
                                 + 1.30925 w_k1_FDX_s012
                                 + 0.824015 w_k1_IBM_s012
                                 + 1.36947 w_k1_INTC_s012 + 1.0624 w_k1_KO_s012
                                 + 0.556076 w_k1_MAT_s012
                                 + 1.02386 w_k1_MSFT_s012
                                 + 1.11246 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.22843 w_k1_ADM_s013
                                 + 0.884505 w_k1_AXP_s013
                                 + 1.31283 w_k1_DIS_s013
                                 + 1.17835 w_k1_FDX_s013
                                 + 0.924017 w_k1_IBM_s013
                                 + 1.20758 w_k1_INTC_s013
                                 + 1.21033 w_k1_KO_s013
                                 + 0.577421 w_k1_MAT_s013
                                 + 1.24848 w_k1_MSFT_s013
                                 + 0.981777 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.01903 w_k1_ADM_s013
                                 + 0.803495 w_k1_AXP_s013
                                 + 0.807034 w_k1_DIS_s013
                                 + 0.804556 w_k1_FDX_s013
                                 + 1.09567 w_k1_IBM_s013
                                 + 1.01399 w_k1_INTC_s013
                                 + 0.903532 w_k1_KO_s013
                                 + 1.07887 w_k1_MAT_s013
                                 + 0.888418 w_k1_MSFT_s013
                                 + 0.915488 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.923526 w_k1_ADM_s013
                                 + 0.735686 w_k1_AXP_s013
                                 + 0.573005 w_k1_DIS_s013
                                 + 0.745237 w_k1_FDX_s013
                                 + 0.752959 w_k1_IBM_s013
                                 + 0.931104 w_k1_INTC_s013
                                 + 0.652602 w_k1_KO_s013
                                 + 0.587752 w_k1_MAT_s013
                                 + 0.776321 w_k1_MSFT_s013
                                 + 0.77114 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.884265 w_k1_ADM_s013
                                 + 1.35794 w_k1_AXP_s013 + 1.7009 w_k1_DIS_s013
                                 + 1.47403 w_k1_FDX_s013
                                 + 1.44447 w_k1_IBM_s013
                                 + 1.64271 w_k1_INTC_s013
                                 + 1.29255 w_k1_KO_s013 + 1.54845 w_k1_MAT_s013
                                 + 1.36378 w_k1_MSFT_s013
                                 + 1.22565 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.06248 w_k1_ADM_s013
                                 + 0.977985 w_k1_AXP_s013
                                 + 0.78506 w_k1_DIS_s013
                                 + 1.03143 w_k1_FDX_s013
                                 + 0.780437 w_k1_IBM_s013
                                 + 1.10702 w_k1_INTC_s013
                                 + 0.797372 w_k1_KO_s013
                                 + 0.887897 w_k1_MAT_s013
                                 + 0.875752 w_k1_MSFT_s013
                                 + 0.824006 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.834287 w_k1_ADM_s013
                                 + 0.921954 w_k1_AXP_s013
                                 + 0.794932 w_k1_DIS_s013
                                 + 0.806181 w_k1_FDX_s013
                                 + 1.58416 w_k1_IBM_s013
                                 + 1.03508 w_k1_INTC_s013
                                 + 1.06386 w_k1_KO_s013 + 1.38421 w_k1_MAT_s013
                                 + 0.810038 w_k1_MSFT_s013
                                 + 0.908737 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.97539 w_k1_ADM_s013
                                 + 1.28502 w_k1_AXP_s013 + 1.2516 w_k1_DIS_s013
                                 + 1.36026 w_k1_FDX_s013
                                 + 0.984009 w_k1_IBM_s013
                                 + 1.29461 w_k1_INTC_s013
                                 + 0.994469 w_k1_KO_s013
                                 + 1.00218 w_k1_MAT_s013
                                 + 1.2109 w_k1_MSFT_s013 + 1.05694 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.13127 w_k1_ADM_s013
                                 + 0.969624 w_k1_AXP_s013
                                 + 1.10742 w_k1_DIS_s013
                                 + 1.19468 w_k1_FDX_s013
                                 + 0.613527 w_k1_IBM_s013
                                 + 1.3424 w_k1_INTC_s013
                                 + 0.729485 w_k1_KO_s013
                                 + 0.419987 w_k1_MAT_s013
                                 + 1.08833 w_k1_MSFT_s013
                                 + 0.827247 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.888878 w_k1_ADM_s014
                                 + 1.09175 w_k1_AXP_s014
                                 + 1.01466 w_k1_DIS_s014
                                 + 1.08705 w_k1_FDX_s014
                                 + 0.843799 w_k1_IBM_s014
                                 + 0.987242 w_k1_INTC_s014
                                 + 0.93434 w_k1_KO_s014
                                 + 0.835852 w_k1_MAT_s014
                                 + 0.94767 w_k1_MSFT_s014
                                 + 0.982649 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.723674 w_k1_ADM_s014
                                 + 0.980236 w_k1_AXP_s014
                                 + 0.691012 w_k1_DIS_s014
                                 + 0.759876 w_k1_FDX_s014
                                 + 1.22013 w_k1_IBM_s014
                                 + 1.03745 w_k1_INTC_s014
                                 + 0.903287 w_k1_KO_s014 + 1.1418 w_k1_MAT_s014
                                 + 0.826773 w_k1_MSFT_s014
                                 + 0.81825 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.48317 w_k1_ADM_s014
                                 + 1.55324 w_k1_AXP_s014
                                 + 1.67348 w_k1_DIS_s014 + 1.4555 w_k1_FDX_s014
                                 + 1.26632 w_k1_IBM_s014
                                 + 1.52899 w_k1_INTC_s014
                                 + 1.41122 w_k1_KO_s014 + 1.35758 w_k1_MAT_s014
                                 + 1.44923 w_k1_MSFT_s014
                                 + 1.24274 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.864094 w_k1_ADM_s014
                                 + 0.703906 w_k1_AXP_s014
                                 + 0.760105 w_k1_DIS_s014
                                 + 0.840018 w_k1_FDX_s014
                                 + 0.73523 w_k1_IBM_s014
                                 + 0.875315 w_k1_INTC_s014
                                 + 0.788164 w_k1_KO_s014
                                 + 0.754631 w_k1_MAT_s014
                                 + 0.886597 w_k1_MSFT_s014
                                 + 0.888982 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.804591 w_k1_ADM_s014
                                 + 1.33748 w_k1_AXP_s014
                                 + 1.00844 w_k1_DIS_s014
                                 + 1.00562 w_k1_FDX_s014
                                 + 1.00056 w_k1_IBM_s014
                                 + 0.940886 w_k1_INTC_s014
                                 + 0.879498 w_k1_KO_s014
                                 + 1.54452 w_k1_MAT_s014
                                 + 0.917917 w_k1_MSFT_s014
                                 + 0.938827 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.763205 w_k1_ADM_s014
                                 + 1.17073 w_k1_AXP_s014
                                 + 0.79786 w_k1_DIS_s014
                                 + 0.876897 w_k1_FDX_s014
                                 + 1.03456 w_k1_IBM_s014
                                 + 0.849373 w_k1_INTC_s014
                                 + 0.93655 w_k1_KO_s014 + 1.25286 w_k1_MAT_s014
                                 + 0.813536 w_k1_MSFT_s014
                                 + 1.02217 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.05646 w_k1_ADM_s014
                                 + 1.12728 w_k1_AXP_s014 + 1.3083 w_k1_DIS_s014
                                 + 1.14195 w_k1_FDX_s014
                                 + 1.02012 w_k1_IBM_s014
                                 + 0.909311 w_k1_INTC_s014
                                 + 0.992882 w_k1_KO_s014
                                 + 1.14555 w_k1_MAT_s014
                                 + 0.977218 w_k1_MSFT_s014
                                 + 1.15453 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.834975 w_k1_ADM_s014
                                 + 1.61688 w_k1_AXP_s014
                                 + 1.28526 w_k1_DIS_s014
                                 + 1.10067 w_k1_FDX_s014
                                 + 1.34396 w_k1_IBM_s014
                                 + 1.49898 w_k1_INTC_s014
                                 + 1.27455 w_k1_KO_s014 + 1.56499 w_k1_MAT_s014
                                 + 1.05143 w_k1_MSFT_s014
                                 + 1.06716 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.981878 w_k1_ADM_s015
                                 + 0.707745 w_k1_AXP_s015
                                 + 0.823459 w_k1_DIS_s015
                                 + 0.846536 w_k1_FDX_s015
                                 + 0.975772 w_k1_IBM_s015
                                 + 0.826619 w_k1_INTC_s015
                                 + 0.811774 w_k1_KO_s015
                                 + 0.755752 w_k1_MAT_s015
                                 + 0.899961 w_k1_MSFT_s015
                                 + 0.871132 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.48721 w_k1_ADM_s015
                                 + 0.981057 w_k1_AXP_s015
                                 + 0.992222 w_k1_DIS_s015
                                 + 1.21346 w_k1_FDX_s015
                                 + 0.716577 w_k1_IBM_s015
                                 + 0.930442 w_k1_INTC_s015
                                 + 0.782685 w_k1_KO_s015
                                 + 0.702617 w_k1_MAT_s015
                                 + 1.11781 w_k1_MSFT_s015
                                 + 0.94183 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.09315 w_k1_ADM_s015
                                 + 1.00453 w_k1_AXP_s015
                                 + 1.23722 w_k1_DIS_s015
                                 + 1.13034 w_k1_FDX_s015
                                 + 0.929444 w_k1_IBM_s015
                                 + 1.11182 w_k1_INTC_s015
                                 + 1.08255 w_k1_KO_s015
                                 + 0.774521 w_k1_MAT_s015
                                 + 1.2732 w_k1_MSFT_s015 + 1.07661 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.944877 w_k1_ADM_s015
                                 + 0.645472 w_k1_AXP_s015
                                 + 0.847108 w_k1_DIS_s015
                                 + 0.855584 w_k1_FDX_s015
                                 + 0.847881 w_k1_IBM_s015
                                 + 1.03998 w_k1_INTC_s015
                                 + 0.819228 w_k1_KO_s015
                                 + 0.480378 w_k1_MAT_s015
                                 + 1.09061 w_k1_MSFT_s015
                                 + 0.808548 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.70186 w_k1_ADM_s015
                                 + 1.43125 w_k1_AXP_s015
                                 + 1.12248 w_k1_DIS_s015
                                 + 1.22769 w_k1_FDX_s015
                                 + 0.965503 w_k1_IBM_s015
                                 + 1.0426 w_k1_INTC_s015 + 1.02728 w_k1_KO_s015
                                 + 0.898824 w_k1_MAT_s015
                                 + 1.13656 w_k1_MSFT_s015
                                 + 1.04046 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.853594 w_k1_ADM_s015
                                 + 0.668654 w_k1_AXP_s015
                                 + 0.638914 w_k1_DIS_s015
                                 + 0.816609 w_k1_FDX_s015
                                 + 0.623858 w_k1_IBM_s015
                                 + 0.761582 w_k1_INTC_s015
                                 + 0.666851 w_k1_KO_s015
                                 + 0.61266 w_k1_MAT_s015
                                 + 0.884075 w_k1_MSFT_s015
                                 + 0.88747 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.29343 w_k1_ADM_s015
                                 + 1.27396 w_k1_AXP_s015
                                 + 1.23062 w_k1_DIS_s015
                                 + 1.11753 w_k1_FDX_s015
                                 + 1.54464 w_k1_IBM_s015
                                 + 0.97074 w_k1_INTC_s015
                                 + 1.12849 w_k1_KO_s015 + 1.39976 w_k1_MAT_s015
                                 + 1.00601 w_k1_MSFT_s015
                                 + 1.10465 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.748615 w_k1_ADM_s015
                                 + 1.05633 w_k1_AXP_s015
                                 + 0.951331 w_k1_DIS_s015
                                 + 0.859159 w_k1_FDX_s015
                                 + 1.22861 w_k1_IBM_s015
                                 + 1.01568 w_k1_INTC_s015
                                 + 0.928439 w_k1_KO_s015
                                 + 1.35614 w_k1_MAT_s015
                                 + 0.901925 w_k1_MSFT_s015
                                 + 1.00436 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.982733 w_k1_ADM_s016
                                 + 1.47116 w_k1_AXP_s016
                                 + 1.38207 w_k1_DIS_s016 + 1.2536 w_k1_FDX_s016
                                 + 1.45393 w_k1_IBM_s016
                                 + 1.08159 w_k1_INTC_s016
                                 + 1.23771 w_k1_KO_s016 + 1.64683 w_k1_MAT_s016
                                 + 1.18029 w_k1_MSFT_s016
                                 + 1.21576 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.00599 w_k1_ADM_s016
                                 + 1.38264 w_k1_AXP_s016 + 1.0372 w_k1_DIS_s016
                                 + 1.09418 w_k1_FDX_s016
                                 + 1.05701 w_k1_IBM_s016
                                 + 1.27899 w_k1_INTC_s016
                                 + 1.06612 w_k1_KO_s016 + 1.00219 w_k1_MAT_s016
                                 + 1.04704 w_k1_MSFT_s016
                                 + 1.18442 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.804868 w_k1_ADM_s016
                                 + 0.793717 w_k1_AXP_s016
                                 + 1.1737 w_k1_DIS_s016
                                 + 0.961474 w_k1_FDX_s016
                                 + 1.02375 w_k1_IBM_s016
                                 + 0.800876 w_k1_INTC_s016
                                 + 1.06825 w_k1_KO_s016 + 1.08827 w_k1_MAT_s016
                                 + 1.10039 w_k1_MSFT_s016 + 1.0166 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.02096 w_k1_ADM_s016
                                 + 0.95567 w_k1_AXP_s016
                                 + 0.897084 w_k1_DIS_s016
                                 + 0.991861 w_k1_FDX_s016
                                 + 0.757398 w_k1_IBM_s016
                                 + 0.853398 w_k1_INTC_s016
                                 + 0.947337 w_k1_KO_s016
                                 + 0.749264 w_k1_MAT_s016
                                 + 0.89141 w_k1_MSFT_s016
                                 + 1.02671 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.878578 w_k1_ADM_s016
                                 + 0.956556 w_k1_AXP_s016
                                 + 0.811437 w_k1_DIS_s016
                                 + 0.770981 w_k1_FDX_s016
                                 + 1.36083 w_k1_IBM_s016
                                 + 0.865499 w_k1_INTC_s016
                                 + 1.12034 w_k1_KO_s016 + 1.33278 w_k1_MAT_s016
                                 + 0.826313 w_k1_MSFT_s016
                                 + 1.10475 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.750051 w_k1_ADM_s016
                                 + 1.08223 w_k1_AXP_s016
                                 + 1.29135 w_k1_DIS_s016
                                 + 0.978495 w_k1_FDX_s016
                                 + 1.63017 w_k1_IBM_s016
                                 + 1.17208 w_k1_INTC_s016
                                 + 1.33494 w_k1_KO_s016 + 1.55268 w_k1_MAT_s016
                                 + 1.2223 w_k1_MSFT_s016 + 1.25829 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.23115 w_k1_ADM_s016
                                 + 1.29065 w_k1_AXP_s016
                                 + 1.53759 w_k1_DIS_s016
                                 + 1.30165 w_k1_FDX_s016
                                 + 0.90135 w_k1_IBM_s016
                                 + 1.14903 w_k1_INTC_s016
                                 + 1.14331 w_k1_KO_s016
                                 + 0.850882 w_k1_MAT_s016
                                 + 1.27702 w_k1_MSFT_s016
                                 + 1.04935 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.61025 w_k1_ADM_s016
                                 + 0.627136 w_k1_AXP_s016
                                 + 0.783688 w_k1_DIS_s016
                                 + 0.759329 w_k1_FDX_s016
                                 + 0.740353 w_k1_IBM_s016
                                 + 0.653337 w_k1_INTC_s016
                                 + 0.880848 w_k1_KO_s016
                                 + 0.635483 w_k1_MAT_s016
                                 + 0.787112 w_k1_MSFT_s016
                                 + 0.927338 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.29374 w_k1_ADM_s017
                                 + 1.51608 w_k1_AXP_s017
                                 + 1.88774 w_k1_DIS_s017
                                 + 1.43304 w_k1_FDX_s017
                                 + 1.16745 w_k1_IBM_s017 + 1.168 w_k1_INTC_s017
                                 + 1.27231 w_k1_KO_s017 + 1.41822 w_k1_MAT_s017
                                 + 1.27854 w_k1_MSFT_s017 + 1.4567 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.12753 w_k1_ADM_s017
                                 + 0.862471 w_k1_AXP_s017
                                 + 1.07449 w_k1_DIS_s017
                                 + 1.02532 w_k1_FDX_s017
                                 + 0.860112 w_k1_IBM_s017
                                 + 1.02693 w_k1_INTC_s017 + 1.0207 w_k1_KO_s017
                                 + 0.965899 w_k1_MAT_s017
                                 + 1.12886 w_k1_MSFT_s017
                                 + 1.10252 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.920348 w_k1_ADM_s017
                                 + 0.790488 w_k1_AXP_s017
                                 + 1.10344 w_k1_DIS_s017
                                 + 1.02862 w_k1_FDX_s017
                                 + 1.32967 w_k1_IBM_s017
                                 + 0.972652 w_k1_INTC_s017
                                 + 1.13155 w_k1_KO_s017 + 1.14114 w_k1_MAT_s017
                                 + 1.06245 w_k1_MSFT_s017
                                 + 1.16968 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.703664 w_k1_ADM_s017
                                 + 0.843856 w_k1_AXP_s017
                                 + 0.643441 w_k1_DIS_s017
                                 + 0.740229 w_k1_FDX_s017
                                 + 1.06611 w_k1_IBM_s017
                                 + 0.786858 w_k1_INTC_s017
                                 + 0.77576 w_k1_KO_s017
                                 + 0.965244 w_k1_MAT_s017
                                 + 0.788511 w_k1_MSFT_s017
                                 + 0.959555 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.34353 w_k1_ADM_s017
                                 + 1.38546 w_k1_AXP_s017
                                 + 0.932934 w_k1_DIS_s017
                                 + 1.14612 w_k1_FDX_s017
                                 + 0.956809 w_k1_IBM_s017
                                 + 1.08835 w_k1_INTC_s017
                                 + 0.87285 w_k1_KO_s017 + 1.25542 w_k1_MAT_s017
                                 + 1.01588 w_k1_MSFT_s017
                                 + 1.16478 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.20012 w_k1_ADM_s017
                                 + 1.31668 w_k1_AXP_s017
                                 + 1.15844 w_k1_DIS_s017
                                 + 1.14897 w_k1_FDX_s017
                                 + 1.20653 w_k1_IBM_s017
                                 + 1.26773 w_k1_INTC_s017
                                 + 1.14663 w_k1_KO_s017 + 1.37919 w_k1_MAT_s017
                                 + 1.18102 w_k1_MSFT_s017
                                 + 1.12039 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.946789 w_k1_ADM_s017
                                 + 0.638268 w_k1_AXP_s017
                                 + 0.693171 w_k1_DIS_s017
                                 + 0.734523 w_k1_FDX_s017
                                 + 0.696137 w_k1_IBM_s017
                                 + 0.646994 w_k1_INTC_s017
                                 + 0.69997 w_k1_KO_s017 + 0.97916 w_k1_MAT_s017
                                 + 0.79991 w_k1_MSFT_s017
                                 + 0.898253 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.919918 w_k1_ADM_s017
                                 + 0.823541 w_k1_AXP_s017
                                 + 0.804854 w_k1_DIS_s017
                                 + 0.793234 w_k1_FDX_s017
                                 + 0.937007 w_k1_IBM_s017
                                 + 0.903409 w_k1_INTC_s017
                                 + 0.933868 w_k1_KO_s017
                                 + 1.16965 w_k1_MAT_s017
                                 + 0.947202 w_k1_MSFT_s017
                                 + 1.01647 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.43023 w_k1_ADM_s100
                                 + 1.26358 w_k1_AXP_s100
                                 + 1.51943 w_k1_DIS_s100
                                 + 1.37136 w_k1_FDX_s100
                                 + 0.768851 w_k1_IBM_s100
                                 + 1.28541 w_k1_INTC_s100
                                 + 1.09416 w_k1_KO_s100
                                 + 0.611717 w_k1_MAT_s100
                                 + 1.21271 w_k1_MSFT_s100
                                 + 1.12086 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.846586 w_k1_ADM_s100
                                 + 0.814197 w_k1_AXP_s100
                                 + 0.844931 w_k1_DIS_s100
                                 + 0.904411 w_k1_FDX_s100
                                 + 1.03073 w_k1_IBM_s100
                                 + 0.861312 w_k1_INTC_s100
                                 + 0.984963 w_k1_KO_s100
                                 + 0.833672 w_k1_MAT_s100
                                 + 0.876617 w_k1_MSFT_s100
                                 + 1.00793 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.04253 w_k1_ADM_s100
                                 + 1.14376 w_k1_AXP_s100
                                 + 1.14263 w_k1_DIS_s100
                                 + 1.13226 w_k1_FDX_s100
                                 + 1.09253 w_k1_IBM_s100
                                 + 1.26535 w_k1_INTC_s100
                                 + 1.05488 w_k1_KO_s100 + 1.09215 w_k1_MAT_s100
                                 + 1.13845 w_k1_MSFT_s100
                                 + 1.08031 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.7333 w_k1_ADM_s100
                                 + 0.696212 w_k1_AXP_s100
                                 + 0.57889 w_k1_DIS_s100
                                 + 0.638577 w_k1_FDX_s100
                                 + 0.857827 w_k1_IBM_s100
                                 + 0.743819 w_k1_INTC_s100
                                 + 0.826743 w_k1_KO_s100
                                 + 0.714693 w_k1_MAT_s100
                                 + 0.695839 w_k1_MSFT_s100
                                 + 0.785965 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.838669 w_k1_ADM_s100
                                 + 0.956674 w_k1_AXP_s100
                                 + 1.03742 w_k1_DIS_s100
                                 + 0.883144 w_k1_FDX_s100
                                 + 1.56636 w_k1_IBM_s100
                                 + 0.890367 w_k1_INTC_s100
                                 + 1.11479 w_k1_KO_s100 + 1.2439 w_k1_MAT_s100
                                 + 0.891703 w_k1_MSFT_s100
                                 + 0.968853 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.784924 w_k1_ADM_s100
                                 + 0.829759 w_k1_AXP_s100
                                 + 0.802267 w_k1_DIS_s100
                                 + 0.906993 w_k1_FDX_s100
                                 + 0.703866 w_k1_IBM_s100
                                 + 1.03001 w_k1_INTC_s100
                                 + 0.810999 w_k1_KO_s100
                                 + 0.635344 w_k1_MAT_s100
                                 + 0.851649 w_k1_MSFT_s100
                                 + 0.891431 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.37676 w_k1_ADM_s100
                                 + 1.34612 w_k1_AXP_s100 + 1.5299 w_k1_DIS_s100
                                 + 1.25958 w_k1_FDX_s100
                                 + 1.41452 w_k1_IBM_s100
                                 + 1.10037 w_k1_INTC_s100
                                 + 1.23832 w_k1_KO_s100 + 1.26279 w_k1_MAT_s100
                                 + 1.21867 w_k1_MSFT_s100
                                 + 1.13797 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.33186 w_k1_ADM_s100
                                 + 1.25886 w_k1_AXP_s100
                                 + 1.31756 w_k1_DIS_s100
                                 + 1.29408 w_k1_FDX_s100
                                 + 0.924345 w_k1_IBM_s100
                                 + 1.03962 w_k1_INTC_s100
                                 + 1.09888 w_k1_KO_s100
                                 + 0.904734 w_k1_MAT_s100
                                 + 1.16279 w_k1_MSFT_s100
                                 + 1.12912 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01167 w_k1_ADM_s101
                                 + 1.19776 w_k1_AXP_s101
                                 + 0.809909 w_k1_DIS_s101
                                 + 0.943796 w_k1_FDX_s101
                                 + 1.10911 w_k1_IBM_s101
                                 + 1.0408 w_k1_INTC_s101 + 1.04883 w_k1_KO_s101
                                 + 1.58599 w_k1_MAT_s101
                                 + 0.972173 w_k1_MSFT_s101
                                 + 0.993025 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.05685 w_k1_ADM_s101
                                 + 0.855711 w_k1_AXP_s101
                                 + 1.15281 w_k1_DIS_s101
                                 + 0.982416 w_k1_FDX_s101
                                 + 1.11846 w_k1_IBM_s101
                                 + 0.987012 w_k1_INTC_s101
                                 + 1.30212 w_k1_KO_s101
                                 + 0.907159 w_k1_MAT_s101
                                 + 1.12672 w_k1_MSFT_s101
                                 + 0.991233 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.910521 w_k1_ADM_s101
                                 + 0.792074 w_k1_AXP_s101
                                 + 0.68999 w_k1_DIS_s101
                                 + 0.848252 w_k1_FDX_s101
                                 + 0.658241 w_k1_IBM_s101
                                 + 0.774393 w_k1_INTC_s101
                                 + 0.720837 w_k1_KO_s101
                                 + 0.537043 w_k1_MAT_s101
                                 + 0.902943 w_k1_MSFT_s101
                                 + 0.716418 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.6449 w_k1_ADM_s101
                                 + 1.38704 w_k1_AXP_s101
                                 + 1.42711 w_k1_DIS_s101
                                 + 1.23261 w_k1_FDX_s101
                                 + 1.36931 w_k1_IBM_s101
                                 + 1.07352 w_k1_INTC_s101
                                 + 1.35567 w_k1_KO_s101 + 1.28658 w_k1_MAT_s101
                                 + 1.3521 w_k1_MSFT_s101 + 1.12551 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.17369 w_k1_ADM_s101
                                 + 1.13025 w_k1_AXP_s101
                                 + 1.22922 w_k1_DIS_s101
                                 + 1.07983 w_k1_FDX_s101
                                 + 0.816244 w_k1_IBM_s101
                                 + 0.993812 w_k1_INTC_s101
                                 + 0.948444 w_k1_KO_s101
                                 + 1.02038 w_k1_MAT_s101
                                 + 1.28225 w_k1_MSFT_s101
                                 + 0.986669 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.944136 w_k1_ADM_s101
                                 + 0.888522 w_k1_AXP_s101
                                 + 0.711951 w_k1_DIS_s101
                                 + 0.817317 w_k1_FDX_s101
                                 + 0.939265 w_k1_IBM_s101
                                 + 0.988554 w_k1_INTC_s101
                                 + 0.844115 w_k1_KO_s101
                                 + 0.690311 w_k1_MAT_s101
                                 + 0.919051 w_k1_MSFT_s101
                                 + 0.883121 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.21347 w_k1_ADM_s101
                                 + 0.886757 w_k1_AXP_s101
                                 + 1.00979 w_k1_DIS_s101
                                 + 0.932471 w_k1_FDX_s101
                                 + 1.38688 w_k1_IBM_s101
                                 + 0.947519 w_k1_INTC_s101 + 1.05 w_k1_KO_s101
                                 + 1.09193 w_k1_MAT_s101
                                 + 1.02989 w_k1_MSFT_s101
                                 + 0.859209 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.937388 w_k1_ADM_s101
                                 + 0.895308 w_k1_AXP_s101
                                 + 0.949938 w_k1_DIS_s101
                                 + 0.994001 w_k1_FDX_s101
                                 + 0.81159 w_k1_IBM_s101
                                 + 0.837436 w_k1_INTC_s101
                                 + 0.981799 w_k1_KO_s101
                                 + 0.900544 w_k1_MAT_s101
                                 + 1.07172 w_k1_MSFT_s101
                                 + 0.893974 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.10819 w_k1_ADM_s102
                                 + 0.796633 w_k1_AXP_s102
                                 + 1.09291 w_k1_DIS_s102
                                 + 1.01148 w_k1_FDX_s102
                                 + 1.06008 w_k1_IBM_s102
                                 + 0.91727 w_k1_INTC_s102
                                 + 1.13899 w_k1_KO_s102
                                 + 0.965364 w_k1_MAT_s102
                                 + 0.910586 w_k1_MSFT_s102
                                 + 1.02017 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.03996 w_k1_ADM_s102
                                 + 1.14166 w_k1_AXP_s102
                                 + 1.05465 w_k1_DIS_s102
                                 + 0.958877 w_k1_FDX_s102
                                 + 1.31355 w_k1_IBM_s102
                                 + 0.843499 w_k1_INTC_s102
                                 + 1.19184 w_k1_KO_s102 + 1.29741 w_k1_MAT_s102
                                 + 0.855897 w_k1_MSFT_s102
                                 + 0.917041 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.28982 w_k1_ADM_s102
                                 + 1.50879 w_k1_AXP_s102
                                 + 1.72611 w_k1_DIS_s102
                                 + 1.49787 w_k1_FDX_s102
                                 + 0.994024 w_k1_IBM_s102
                                 + 1.64386 w_k1_INTC_s102
                                 + 1.24779 w_k1_KO_s102
                                 + 0.969553 w_k1_MAT_s102
                                 + 1.41542 w_k1_MSFT_s102
                                 + 1.08529 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.00891 w_k1_ADM_s102
                                 + 1.02738 w_k1_AXP_s102
                                 + 0.938273 w_k1_DIS_s102
                                 + 0.962158 w_k1_FDX_s102
                                 + 0.936499 w_k1_IBM_s102
                                 + 0.836341 w_k1_INTC_s102
                                 + 0.942851 w_k1_KO_s102 + 1.0128 w_k1_MAT_s102
                                 + 0.844158 w_k1_MSFT_s102
                                 + 0.938524 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.21878 w_k1_ADM_s102
                                 + 1.2716 w_k1_AXP_s102 + 1.56792 w_k1_DIS_s102
                                 + 1.27678 w_k1_FDX_s102
                                 + 1.13902 w_k1_IBM_s102
                                 + 0.995653 w_k1_INTC_s102
                                 + 1.16877 w_k1_KO_s102
                                 + 0.866284 w_k1_MAT_s102
                                 + 1.12486 w_k1_MSFT_s102
                                 + 0.959908 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.25128 w_k1_ADM_s102
                                 + 0.781099 w_k1_AXP_s102
                                 + 0.715069 w_k1_DIS_s102
                                 + 0.892372 w_k1_FDX_s102
                                 + 0.693598 w_k1_IBM_s102
                                 + 0.842422 w_k1_INTC_s102
                                 + 0.760515 w_k1_KO_s102
                                 + 0.578882 w_k1_MAT_s102
                                 + 0.905378 w_k1_MSFT_s102
                                 + 0.764096 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.962293 w_k1_ADM_s102
                                 + 0.678963 w_k1_AXP_s102
                                 + 0.599265 w_k1_DIS_s102
                                 + 0.67916 w_k1_FDX_s102
                                 + 0.817282 w_k1_IBM_s102
                                 + 0.715786 w_k1_INTC_s102
                                 + 0.788925 w_k1_KO_s102
                                 + 0.435073 w_k1_MAT_s102
                                 + 0.671752 w_k1_MSFT_s102
                                 + 0.750377 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.24125 w_k1_ADM_s102
                                 + 1.26288 w_k1_AXP_s102
                                 + 1.38928 w_k1_DIS_s102
                                 + 1.06052 w_k1_FDX_s102
                                 + 1.38298 w_k1_IBM_s102
                                 + 1.17202 w_k1_INTC_s102
                                 + 1.35954 w_k1_KO_s102 + 1.03799 w_k1_MAT_s102
                                 + 1.07781 w_k1_MSFT_s102 + 1.1221 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.931322 w_k1_ADM_s103
                                 + 0.748202 w_k1_AXP_s103
                                 + 1.00163 w_k1_DIS_s103
                                 + 0.855748 w_k1_FDX_s103
                                 + 1.00506 w_k1_IBM_s103
                                 + 0.881071 w_k1_INTC_s103
                                 + 0.953062 w_k1_KO_s103
                                 + 0.714446 w_k1_MAT_s103
                                 + 0.87186 w_k1_MSFT_s103
                                 + 1.05607 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.1147 w_k1_ADM_s103
                                 + 1.0466 w_k1_AXP_s103 + 1.15025 w_k1_DIS_s103
                                 + 1.02136 w_k1_FDX_s103
                                 + 1.44593 w_k1_IBM_s103
                                 + 0.879023 w_k1_INTC_s103
                                 + 1.29082 w_k1_KO_s103 + 1.27802 w_k1_MAT_s103
                                 + 1.06416 w_k1_MSFT_s103
                                 + 1.05409 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.46854 w_k1_ADM_s103
                                 + 1.60245 w_k1_AXP_s103
                                 + 1.84667 w_k1_DIS_s103
                                 + 1.48993 w_k1_FDX_s103
                                 + 1.38127 w_k1_IBM_s103
                                 + 1.16744 w_k1_INTC_s103
                                 + 1.56037 w_k1_KO_s103 + 1.67706 w_k1_MAT_s103
                                 + 1.44342 w_k1_MSFT_s103
                                 + 1.59972 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.37629 w_k1_ADM_s103
                                 + 1.34495 w_k1_AXP_s103 + 1.3943 w_k1_DIS_s103
                                 + 1.29707 w_k1_FDX_s103
                                 + 1.02844 w_k1_IBM_s103
                                 + 0.943823 w_k1_INTC_s103
                                 + 1.18084 w_k1_KO_s103 + 1.09523 w_k1_MAT_s103
                                 + 1.06425 w_k1_MSFT_s103
                                 + 1.11936 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.908721 w_k1_ADM_s103
                                 + 0.788272 w_k1_AXP_s103
                                 + 0.974895 w_k1_DIS_s103
                                 + 0.939296 w_k1_FDX_s103
                                 + 1.15645 w_k1_IBM_s103
                                 + 0.98226 w_k1_INTC_s103
                                 + 1.24198 w_k1_KO_s103
                                 + 0.799569 w_k1_MAT_s103
                                 + 1.06698 w_k1_MSFT_s103
                                 + 1.10496 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.15639 w_k1_ADM_s103
                                 + 1.15222 w_k1_AXP_s103
                                 + 1.59182 w_k1_DIS_s103
                                 + 1.14171 w_k1_FDX_s103
                                 + 1.09693 w_k1_IBM_s103
                                 + 0.969974 w_k1_INTC_s103
                                 + 1.39227 w_k1_KO_s103 + 1.37251 w_k1_MAT_s103
                                 + 1.21741 w_k1_MSFT_s103
                                 + 1.31989 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.827321 w_k1_ADM_s103
                                 + 0.710085 w_k1_AXP_s103
                                 + 0.522245 w_k1_DIS_s103
                                 + 0.689439 w_k1_FDX_s103
                                 + 0.908769 w_k1_IBM_s103
                                 + 0.660354 w_k1_INTC_s103
                                 + 0.700461 w_k1_KO_s103
                                 + 0.838951 w_k1_MAT_s103
                                 + 0.581453 w_k1_MSFT_s103
                                 + 0.81446 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01312 w_k1_ADM_s103
                                 + 0.906364 w_k1_AXP_s103
                                 + 0.830445 w_k1_DIS_s103
                                 + 0.969748 w_k1_FDX_s103
                                 + 0.804456 w_k1_IBM_s103
                                 + 0.973151 w_k1_INTC_s103
                                 + 0.863883 w_k1_KO_s103
                                 + 0.895534 w_k1_MAT_s103
                                 + 0.979235 w_k1_MSFT_s103
                                 + 0.931027 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.895263 w_k1_ADM_s104
                                 + 0.932365 w_k1_AXP_s104
                                 + 0.773232 w_k1_DIS_s104
                                 + 0.857056 w_k1_FDX_s104
                                 + 1.06475 w_k1_IBM_s104
                                 + 0.776044 w_k1_INTC_s104
                                 + 0.961134 w_k1_KO_s104
                                 + 1.05735 w_k1_MAT_s104
                                 + 0.743723 w_k1_MSFT_s104
                                 + 0.970631 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.19431 w_k1_ADM_s104
                                 + 1.29834 w_k1_AXP_s104
                                 + 1.42492 w_k1_DIS_s104
                                 + 1.37943 w_k1_FDX_s104
                                 + 0.874383 w_k1_IBM_s104
                                 + 1.00115 w_k1_INTC_s104
                                 + 1.01933 w_k1_KO_s104
                                 + 0.982199 w_k1_MAT_s104
                                 + 1.17145 w_k1_MSFT_s104
                                 + 1.04499 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.01457 w_k1_ADM_s104
                                 + 1.2411 w_k1_AXP_s104 + 1.02855 w_k1_DIS_s104
                                 + 0.922552 w_k1_FDX_s104
                                 + 1.36165 w_k1_IBM_s104
                                 + 0.888808 w_k1_INTC_s104
                                 + 1.03859 w_k1_KO_s104 + 1.27671 w_k1_MAT_s104
                                 + 0.82661 w_k1_MSFT_s104
                                 + 1.01561 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.746504 w_k1_ADM_s104
                                 + 0.801748 w_k1_AXP_s104
                                 + 0.891001 w_k1_DIS_s104
                                 + 0.771559 w_k1_FDX_s104
                                 + 1.22378 w_k1_IBM_s104
                                 + 0.816015 w_k1_INTC_s104
                                 + 0.910168 w_k1_KO_s104
                                 + 1.30876 w_k1_MAT_s104
                                 + 0.885097 w_k1_MSFT_s104
                                 + 0.87457 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.2217 w_k1_ADM_s104
                                 + 1.61399 w_k1_AXP_s104 + 1.4897 w_k1_DIS_s104
                                 + 1.12508 w_k1_FDX_s104
                                 + 1.42487 w_k1_IBM_s104
                                 + 1.00906 w_k1_INTC_s104
                                 + 1.25984 w_k1_KO_s104 + 1.70044 w_k1_MAT_s104
                                 + 1.10974 w_k1_MSFT_s104
                                 + 1.10804 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.932308 w_k1_ADM_s104
                                 + 1.21148 w_k1_AXP_s104
                                 + 1.23443 w_k1_DIS_s104
                                 + 1.06976 w_k1_FDX_s104
                                 + 1.03059 w_k1_IBM_s104
                                 + 1.11676 w_k1_INTC_s104
                                 + 1.03464 w_k1_KO_s104 + 1.08826 w_k1_MAT_s104
                                 + 1.12984 w_k1_MSFT_s104
                                 + 1.18226 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.08742 w_k1_ADM_s104
                                 + 1.06256 w_k1_AXP_s104
                                 + 0.874647 w_k1_DIS_s104
                                 + 0.985592 w_k1_FDX_s104
                                 + 0.790155 w_k1_IBM_s104
                                 + 1.00154 w_k1_INTC_s104
                                 + 0.775948 w_k1_KO_s104 + 0.766 w_k1_MAT_s104
                                 + 1.03188 w_k1_MSFT_s104
                                 + 0.914478 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.831467 w_k1_ADM_s104
                                 + 0.757147 w_k1_AXP_s104
                                 + 0.674086 w_k1_DIS_s104
                                 + 0.767773 w_k1_FDX_s104
                                 + 0.780557 w_k1_IBM_s104
                                 + 0.721044 w_k1_INTC_s104
                                 + 0.712583 w_k1_KO_s104
                                 + 0.55781 w_k1_MAT_s104
                                 + 0.688054 w_k1_MSFT_s104
                                 + 0.837821 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.933596 w_k1_ADM_s105
                                 + 0.950767 w_k1_AXP_s105
                                 + 1.1643 w_k1_DIS_s105 + 1.02742 w_k1_FDX_s105
                                 + 0.615846 w_k1_IBM_s105
                                 + 0.924491 w_k1_INTC_s105
                                 + 0.756972 w_k1_KO_s105
                                 + 0.424803 w_k1_MAT_s105
                                 + 1.0525 w_k1_MSFT_s105
                                 + 0.952895 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.714262 w_k1_ADM_s105
                                 + 0.705186 w_k1_AXP_s105
                                 + 0.591757 w_k1_DIS_s105
                                 + 0.672052 w_k1_FDX_s105
                                 + 1.03001 w_k1_IBM_s105
                                 + 0.710446 w_k1_INTC_s105
                                 + 0.811677 w_k1_KO_s105
                                 + 0.781224 w_k1_MAT_s105
                                 + 0.69924 w_k1_MSFT_s105
                                 + 0.856236 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.26986 w_k1_ADM_s105
                                 + 1.48551 w_k1_AXP_s105
                                 + 1.74091 w_k1_DIS_s105
                                 + 1.38401 w_k1_FDX_s105
                                 + 1.05889 w_k1_IBM_s105
                                 + 1.11858 w_k1_INTC_s105
                                 + 1.37302 w_k1_KO_s105 + 1.21499 w_k1_MAT_s105
                                 + 1.20289 w_k1_MSFT_s105
                                 + 1.09588 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.569638 w_k1_ADM_s105
                                 + 0.666781 w_k1_AXP_s105
                                 + 0.550939 w_k1_DIS_s105
                                 + 0.703005 w_k1_FDX_s105
                                 + 0.657801 w_k1_IBM_s105
                                 + 0.677257 w_k1_INTC_s105
                                 + 0.627948 w_k1_KO_s105
                                 + 0.718674 w_k1_MAT_s105
                                 + 0.766538 w_k1_MSFT_s105
                                 + 0.765859 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.701631 w_k1_ADM_s105
                                 + 1.07288 w_k1_AXP_s105
                                 + 1.09524 w_k1_DIS_s105
                                 + 0.954698 w_k1_FDX_s105
                                 + 1.52944 w_k1_IBM_s105
                                 + 0.949368 w_k1_INTC_s105
                                 + 1.28523 w_k1_KO_s105 + 1.73331 w_k1_MAT_s105
                                 + 0.950389 w_k1_MSFT_s105
                                 + 1.08495 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.72349 w_k1_ADM_s105
                                 + 0.897506 w_k1_AXP_s105
                                 + 0.83331 w_k1_DIS_s105
                                 + 0.93615 w_k1_FDX_s105
                                 + 0.928497 w_k1_IBM_s105
                                 + 0.711716 w_k1_INTC_s105
                                 + 0.85639 w_k1_KO_s105 + 1.02505 w_k1_MAT_s105
                                 + 0.839652 w_k1_MSFT_s105
                                 + 0.863766 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.809858 w_k1_ADM_s105
                                 + 1.29748 w_k1_AXP_s105
                                 + 0.982009 w_k1_DIS_s105
                                 + 1.07346 w_k1_FDX_s105
                                 + 0.775991 w_k1_IBM_s105
                                 + 0.860688 w_k1_INTC_s105
                                 + 0.949238 w_k1_KO_s105
                                 + 1.07943 w_k1_MAT_s105
                                 + 0.975883 w_k1_MSFT_s105
                                 + 0.887416 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.980305 w_k1_ADM_s105
                                 + 0.885323 w_k1_AXP_s105
                                 + 1.12281 w_k1_DIS_s105
                                 + 1.04712 w_k1_FDX_s105
                                 + 0.772803 w_k1_IBM_s105
                                 + 0.98746 w_k1_INTC_s105
                                 + 1.00704 w_k1_KO_s105
                                 + 0.908041 w_k1_MAT_s105
                                 + 1.11504 w_k1_MSFT_s105
                                 + 1.06276 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.46922 w_k1_ADM_s106
                                 + 1.30053 w_k1_AXP_s106
                                 + 1.74843 w_k1_DIS_s106
                                 + 1.37335 w_k1_FDX_s106
                                 + 1.42268 w_k1_IBM_s106
                                 + 1.41152 w_k1_INTC_s106
                                 + 1.61913 w_k1_KO_s106 + 1.61029 w_k1_MAT_s106
                                 + 1.48936 w_k1_MSFT_s106
                                 + 1.30923 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.952311 w_k1_ADM_s106
                                 + 0.720259 w_k1_AXP_s106
                                 + 0.820238 w_k1_DIS_s106
                                 + 0.991324 w_k1_FDX_s106
                                 + 0.727356 w_k1_IBM_s106
                                 + 0.9664 w_k1_INTC_s106
                                 + 0.826366 w_k1_KO_s106
                                 + 0.782758 w_k1_MAT_s106
                                 + 0.96412 w_k1_MSFT_s106
                                 + 0.954171 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.19144 w_k1_ADM_s106
                                 + 0.830501 w_k1_AXP_s106
                                 + 1.21819 w_k1_DIS_s106
                                 + 1.10787 w_k1_FDX_s106
                                 + 1.02225 w_k1_IBM_s106
                                 + 1.17753 w_k1_INTC_s106 + 1.2447 w_k1_KO_s106
                                 + 1.02929 w_k1_MAT_s106
                                 + 1.30828 w_k1_MSFT_s106
                                 + 1.17736 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.29336 w_k1_ADM_s106
                                 + 0.847362 w_k1_AXP_s106
                                 + 1.29562 w_k1_DIS_s106
                                 + 1.19186 w_k1_FDX_s106
                                 + 0.750678 w_k1_IBM_s106
                                 + 1.33609 w_k1_INTC_s106
                                 + 0.981452 w_k1_KO_s106
                                 + 0.691924 w_k1_MAT_s106
                                 + 1.51881 w_k1_MSFT_s106
                                 + 1.02878 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.53446 w_k1_ADM_s106
                                 + 1.31206 w_k1_AXP_s106
                                 + 1.37006 w_k1_DIS_s106
                                 + 1.31552 w_k1_FDX_s106
                                 + 1.08271 w_k1_IBM_s106
                                 + 1.00136 w_k1_INTC_s106
                                 + 1.13786 w_k1_KO_s106 + 1.0216 w_k1_MAT_s106
                                 + 1.08528 w_k1_MSFT_s106
                                 + 1.15853 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.15964 w_k1_ADM_s106
                                 + 0.711227 w_k1_AXP_s106
                                 + 0.794339 w_k1_DIS_s106
                                 + 0.74421 w_k1_FDX_s106
                                 + 1.31703 w_k1_IBM_s106
                                 + 0.773904 w_k1_INTC_s106
                                 + 0.991941 w_k1_KO_s106
                                 + 0.900689 w_k1_MAT_s106
                                 + 0.954394 w_k1_MSFT_s106
                                 + 0.880799 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.880851 w_k1_ADM_s106
                                 + 0.720737 w_k1_AXP_s106
                                 + 0.72358 w_k1_DIS_s106
                                 + 0.803818 w_k1_FDX_s106
                                 + 0.850626 w_k1_IBM_s106
                                 + 0.8587 w_k1_INTC_s106
                                 + 0.899223 w_k1_KO_s106
                                 + 1.04152 w_k1_MAT_s106
                                 + 0.905506 w_k1_MSFT_s106
                                 + 0.850495 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.39715 w_k1_ADM_s106
                                 + 1.04984 w_k1_AXP_s106
                                 + 0.984976 w_k1_DIS_s106
                                 + 0.956471 w_k1_FDX_s106 + 1.41 w_k1_IBM_s106
                                 + 0.802794 w_k1_INTC_s106
                                 + 1.27177 w_k1_KO_s106 + 1.75667 w_k1_MAT_s106
                                 + 0.950456 w_k1_MSFT_s106
                                 + 1.02115 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.881409 w_k1_ADM_s107
                                 + 1.23792 w_k1_AXP_s107
                                 + 0.802697 w_k1_DIS_s107
                                 + 0.872741 w_k1_FDX_s107
                                 + 1.29413 w_k1_IBM_s107
                                 + 0.918583 w_k1_INTC_s107
                                 + 0.923991 w_k1_KO_s107
                                 + 1.45105 w_k1_MAT_s107
                                 + 0.829897 w_k1_MSFT_s107
                                 + 0.960972 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.688045 w_k1_ADM_s107
                                 + 0.826491 w_k1_AXP_s107
                                 + 0.753093 w_k1_DIS_s107
                                 + 0.785607 w_k1_FDX_s107
                                 + 0.862418 w_k1_IBM_s107
                                 + 0.865217 w_k1_INTC_s107
                                 + 0.755618 w_k1_KO_s107
                                 + 0.756739 w_k1_MAT_s107
                                 + 0.809061 w_k1_MSFT_s107
                                 + 0.799059 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.841481 w_k1_ADM_s107
                                 + 1.12352 w_k1_AXP_s107
                                 + 1.57098 w_k1_DIS_s107
                                 + 1.18101 w_k1_FDX_s107
                                 + 1.15772 w_k1_IBM_s107
                                 + 1.29334 w_k1_INTC_s107
                                 + 1.15821 w_k1_KO_s107
                                 + 0.856518 w_k1_MAT_s107
                                 + 1.22236 w_k1_MSFT_s107
                                 + 1.01385 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.841104 w_k1_ADM_s107
                                 + 0.987605 w_k1_AXP_s107
                                 + 1.05321 w_k1_DIS_s107
                                 + 1.02599 w_k1_FDX_s107 + 1.0692 w_k1_IBM_s107
                                 + 0.976967 w_k1_INTC_s107
                                 + 0.961477 w_k1_KO_s107
                                 + 0.76571 w_k1_MAT_s107
                                 + 0.904896 w_k1_MSFT_s107
                                 + 0.996511 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.51747 w_k1_ADM_s107
                                 + 1.51518 w_k1_AXP_s107
                                 + 1.29201 w_k1_DIS_s107
                                 + 1.28786 w_k1_FDX_s107
                                 + 1.06482 w_k1_IBM_s107
                                 + 1.06747 w_k1_INTC_s107
                                 + 1.08203 w_k1_KO_s107 + 0.88226 w_k1_MAT_s107
                                 + 1.03859 w_k1_MSFT_s107
                                 + 0.963782 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.25087 w_k1_ADM_s107
                                 + 0.899437 w_k1_AXP_s107
                                 + 1.10132 w_k1_DIS_s107
                                 + 1.04266 w_k1_FDX_s107
                                 + 0.832565 w_k1_IBM_s107
                                 + 0.992869 w_k1_INTC_s107
                                 + 0.999106 w_k1_KO_s107
                                 + 0.572396 w_k1_MAT_s107
                                 + 1.13966 w_k1_MSFT_s107
                                 + 0.92318 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.24696 w_k1_ADM_s107
                                 + 1.18373 w_k1_AXP_s107
                                 + 0.995254 w_k1_DIS_s107
                                 + 1.02789 w_k1_FDX_s107
                                 + 0.801904 w_k1_IBM_s107
                                 + 1.00124 w_k1_INTC_s107
                                 + 0.966966 w_k1_KO_s107
                                 + 0.763551 w_k1_MAT_s107
                                 + 0.990007 w_k1_MSFT_s107
                                 + 0.912265 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.29485 w_k1_ADM_s107
                                 + 1.39041 w_k1_AXP_s107
                                 + 1.48273 w_k1_DIS_s107
                                 + 1.16433 w_k1_FDX_s107
                                 + 1.49509 w_k1_IBM_s107
                                 + 1.27044 w_k1_INTC_s107
                                 + 1.34122 w_k1_KO_s107 + 1.51041 w_k1_MAT_s107
                                 + 1.27057 w_k1_MSFT_s107
                                 + 1.10579 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.97039 w_k1_ADM_s110
                                 + 0.769935 w_k1_AXP_s110
                                 + 0.893325 w_k1_DIS_s110
                                 + 0.921896 w_k1_FDX_s110
                                 + 0.822977 w_k1_IBM_s110
                                 + 0.828882 w_k1_INTC_s110
                                 + 0.97719 w_k1_KO_s110
                                 + 0.799637 w_k1_MAT_s110
                                 + 0.999981 w_k1_MSFT_s110
                                 + 0.958518 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.958227 w_k1_ADM_s110
                                 + 0.813618 w_k1_AXP_s110
                                 + 0.895562 w_k1_DIS_s110
                                 + 0.874721 w_k1_FDX_s110
                                 + 1.12368 w_k1_IBM_s110
                                 + 0.855332 w_k1_INTC_s110
                                 + 1.01634 w_k1_KO_s110 + 1.01549 w_k1_MAT_s110
                                 + 0.975751 w_k1_MSFT_s110
                                 + 0.910672 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.787217 w_k1_ADM_s110
                                 + 0.6262 w_k1_AXP_s110
                                 + 0.583671 w_k1_DIS_s110
                                 + 0.675838 w_k1_FDX_s110
                                 + 0.972003 w_k1_IBM_s110
                                 + 0.659516 w_k1_INTC_s110
                                 + 0.822058 w_k1_KO_s110
                                 + 0.843903 w_k1_MAT_s110
                                 + 0.755263 w_k1_MSFT_s110
                                 + 0.722182 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.11347 w_k1_ADM_s110
                                 + 1.30999 w_k1_AXP_s110 + 1.0254 w_k1_DIS_s110
                                 + 1.07929 w_k1_FDX_s110
                                 + 1.01707 w_k1_IBM_s110
                                 + 1.06277 w_k1_INTC_s110
                                 + 0.966405 w_k1_KO_s110
                                 + 1.21274 w_k1_MAT_s110
                                 + 1.04548 w_k1_MSFT_s110
                                 + 1.03757 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.923756 w_k1_ADM_s110
                                 + 0.705193 w_k1_AXP_s110
                                 + 0.637632 w_k1_DIS_s110
                                 + 0.809689 w_k1_FDX_s110
                                 + 0.67204 w_k1_IBM_s110
                                 + 0.68667 w_k1_INTC_s110
                                 + 0.703693 w_k1_KO_s110
                                 + 0.727918 w_k1_MAT_s110
                                 + 0.843935 w_k1_MSFT_s110
                                 + 0.839357 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.62542 w_k1_ADM_s110
                                 + 1.31429 w_k1_AXP_s110
                                 + 1.57091 w_k1_DIS_s110
                                 + 1.36989 w_k1_FDX_s110
                                 + 1.65238 w_k1_IBM_s110
                                 + 1.43908 w_k1_INTC_s110
                                 + 1.29051 w_k1_KO_s110 + 1.47783 w_k1_MAT_s110
                                 + 1.30933 w_k1_MSFT_s110
                                 + 1.22579 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.07159 w_k1_ADM_s110
                                 + 1.22722 w_k1_AXP_s110
                                 + 1.39716 w_k1_DIS_s110
                                 + 1.26386 w_k1_FDX_s110
                                 + 1.04948 w_k1_IBM_s110
                                 + 1.29699 w_k1_INTC_s110
                                 + 1.15332 w_k1_KO_s110 + 1.1446 w_k1_MAT_s110
                                 + 1.30758 w_k1_MSFT_s110
                                 + 1.22089 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.07999 w_k1_ADM_s110
                                 + 0.934566 w_k1_AXP_s110
                                 + 1.03217 w_k1_DIS_s110
                                 + 0.991729 w_k1_FDX_s110
                                 + 0.680396 w_k1_IBM_s110
                                 + 0.958713 w_k1_INTC_s110
                                 + 0.891141 w_k1_KO_s110
                                 + 0.660253 w_k1_MAT_s110
                                 + 1.06227 w_k1_MSFT_s110
                                 + 0.851713 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.927893 w_k1_ADM_s111
                                 + 0.812501 w_k1_AXP_s111
                                 + 0.942716 w_k1_DIS_s111
                                 + 0.93341 w_k1_FDX_s111
                                 + 1.01535 w_k1_IBM_s111
                                 + 0.86174 w_k1_INTC_s111
                                 + 0.949588 w_k1_KO_s111
                                 + 0.687729 w_k1_MAT_s111
                                 + 0.950264 w_k1_MSFT_s111
                                 + 1.11017 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.34902 w_k1_ADM_s111
                                 + 1.43198 w_k1_AXP_s111
                                 + 1.19391 w_k1_DIS_s111
                                 + 1.28677 w_k1_FDX_s111
                                 + 0.819866 w_k1_IBM_s111
                                 + 1.29819 w_k1_INTC_s111
                                 + 0.98526 w_k1_KO_s111
                                 + 0.639492 w_k1_MAT_s111
                                 + 1.17478 w_k1_MSFT_s111
                                 + 1.12882 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.879245 w_k1_ADM_s111
                                 + 1.07119 w_k1_AXP_s111
                                 + 0.944832 w_k1_DIS_s111
                                 + 0.911871 w_k1_FDX_s111
                                 + 1.50004 w_k1_IBM_s111
                                 + 0.983689 w_k1_INTC_s111
                                 + 1.00163 w_k1_KO_s111 + 1.17217 w_k1_MAT_s111
                                 + 0.895352 w_k1_MSFT_s111
                                 + 1.05353 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.28358 w_k1_ADM_s111
                                 + 1.12313 w_k1_AXP_s111
                                 + 1.52906 w_k1_DIS_s111 + 1.1703 w_k1_FDX_s111
                                 + 1.37878 w_k1_IBM_s111
                                 + 1.24243 w_k1_INTC_s111
                                 + 1.26139 w_k1_KO_s111
                                 + 0.886335 w_k1_MAT_s111
                                 + 1.28331 w_k1_MSFT_s111
                                 + 1.16168 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.604435 w_k1_ADM_s111
                                 + 0.620968 w_k1_AXP_s111
                                 + 0.63891 w_k1_DIS_s111
                                 + 0.706994 w_k1_FDX_s111
                                 + 0.777424 w_k1_IBM_s111
                                 + 0.761906 w_k1_INTC_s111
                                 + 0.733814 w_k1_KO_s111
                                 + 0.697581 w_k1_MAT_s111
                                 + 0.859833 w_k1_MSFT_s111
                                 + 0.876213 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.02636 w_k1_ADM_s111
                                 + 1.75968 w_k1_AXP_s111
                                 + 1.47018 w_k1_DIS_s111
                                 + 1.17341 w_k1_FDX_s111 + 2.1841 w_k1_IBM_s111
                                 + 1.20195 w_k1_INTC_s111
                                 + 1.51503 w_k1_KO_s111 + 1.73742 w_k1_MAT_s111
                                 + 1.08859 w_k1_MSFT_s111
                                 + 1.28912 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.990805 w_k1_ADM_s111
                                 + 1.13512 w_k1_AXP_s111
                                 + 0.818633 w_k1_DIS_s111
                                 + 0.919646 w_k1_FDX_s111
                                 + 0.917723 w_k1_IBM_s111
                                 + 0.761063 w_k1_INTC_s111
                                 + 0.867914 w_k1_KO_s111
                                 + 0.98608 w_k1_MAT_s111
                                 + 0.849585 w_k1_MSFT_s111
                                 + 1.0136 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.97608 w_k1_ADM_s111
                                 + 0.879616 w_k1_AXP_s111
                                 + 0.81488 w_k1_DIS_s111
                                 + 0.947264 w_k1_FDX_s111
                                 + 0.660568 w_k1_IBM_s111
                                 + 0.920244 w_k1_INTC_s111
                                 + 0.738555 w_k1_KO_s111
                                 + 0.436116 w_k1_MAT_s111
                                 + 0.978428 w_k1_MSFT_s111
                                 + 0.888338 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.944034 w_k1_ADM_s112
                                 + 0.993311 w_k1_AXP_s112
                                 + 0.774501 w_k1_DIS_s112
                                 + 0.878347 w_k1_FDX_s112
                                 + 1.08372 w_k1_IBM_s112
                                 + 1.09669 w_k1_INTC_s112
                                 + 0.923883 w_k1_KO_s112
                                 + 1.34634 w_k1_MAT_s112
                                 + 0.89446 w_k1_MSFT_s112
                                 + 0.956109 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.981422 w_k1_ADM_s112
                                 + 1.19119 w_k1_AXP_s112
                                 + 1.22211 w_k1_DIS_s112
                                 + 1.18233 w_k1_FDX_s112
                                 + 0.966576 w_k1_IBM_s112
                                 + 1.37879 w_k1_INTC_s112
                                 + 0.996537 w_k1_KO_s112
                                 + 0.888537 w_k1_MAT_s112
                                 + 1.21934 w_k1_MSFT_s112
                                 + 1.07838 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.864191 w_k1_ADM_s112
                                 + 0.984066 w_k1_AXP_s112
                                 + 1.36591 w_k1_DIS_s112
                                 + 0.974057 w_k1_FDX_s112
                                 + 1.29622 w_k1_IBM_s112
                                 + 1.1529 w_k1_INTC_s112 + 1.29413 w_k1_KO_s112
                                 + 1.36968 w_k1_MAT_s112
                                 + 1.22837 w_k1_MSFT_s112
                                 + 1.13287 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.803215 w_k1_ADM_s112
                                 + 0.739351 w_k1_AXP_s112
                                 + 0.674856 w_k1_DIS_s112
                                 + 0.874654 w_k1_FDX_s112
                                 + 0.623401 w_k1_IBM_s112
                                 + 0.923154 w_k1_INTC_s112
                                 + 0.70003 w_k1_KO_s112
                                 + 0.868661 w_k1_MAT_s112
                                 + 0.961877 w_k1_MSFT_s112
                                 + 0.886975 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.842282 w_k1_ADM_s112
                                 + 0.916428 w_k1_AXP_s112
                                 + 0.941331 w_k1_DIS_s112
                                 + 1.03252 w_k1_FDX_s112
                                 + 0.76641 w_k1_IBM_s112
                                 + 1.26911 w_k1_INTC_s112
                                 + 0.841155 w_k1_KO_s112
                                 + 0.918863 w_k1_MAT_s112
                                 + 1.05587 w_k1_MSFT_s112
                                 + 0.969217 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.10702 w_k1_ADM_s112
                                 + 1.5067 w_k1_AXP_s112
                                 + 0.989426 w_k1_DIS_s112
                                 + 1.03704 w_k1_FDX_s112 + 1.3744 w_k1_IBM_s112
                                 + 1.45054 w_k1_INTC_s112 + 1.0919 w_k1_KO_s112
                                 + 1.59219 w_k1_MAT_s112
                                 + 1.13382 w_k1_MSFT_s112
                                 + 1.05869 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.08041 w_k1_ADM_s112
                                 + 1.08634 w_k1_AXP_s112 + 1.0316 w_k1_DIS_s112
                                 + 0.935822 w_k1_FDX_s112
                                 + 1.70934 w_k1_IBM_s112 + 1.423 w_k1_INTC_s112
                                 + 1.30996 w_k1_KO_s112 + 1.61889 w_k1_MAT_s112
                                 + 1.13031 w_k1_MSFT_s112
                                 + 0.982363 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.34681 w_k1_ADM_s112
                                 + 1.17032 w_k1_AXP_s112
                                 + 1.53849 w_k1_DIS_s112
                                 + 1.28816 w_k1_FDX_s112
                                 + 1.11858 w_k1_IBM_s112
                                 + 1.37998 w_k1_INTC_s112
                                 + 1.19507 w_k1_KO_s112 + 1.15494 w_k1_MAT_s112
                                 + 1.5684 w_k1_MSFT_s112 + 1.19587 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.23444 w_k1_ADM_s113
                                 + 1.02667 w_k1_AXP_s113
                                 + 1.25605 w_k1_DIS_s113
                                 + 1.28199 w_k1_FDX_s113
                                 + 0.914529 w_k1_IBM_s113
                                 + 1.18741 w_k1_INTC_s113
                                 + 1.00935 w_k1_KO_s113 + 0.91699 w_k1_MAT_s113
                                 + 1.23663 w_k1_MSFT_s113
                                 + 1.11721 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.00477 w_k1_ADM_s113
                                 + 1.10025 w_k1_AXP_s113
                                 + 0.970412 w_k1_DIS_s113
                                 + 1.04333 w_k1_FDX_s113
                                 + 0.90693 w_k1_IBM_s113
                                 + 1.04701 w_k1_INTC_s113
                                 + 0.923896 w_k1_KO_s113 + 0.9255 w_k1_MAT_s113
                                 + 1.05053 w_k1_MSFT_s113
                                 + 1.04518 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.73128 w_k1_ADM_s113
                                 + 1.61464 w_k1_AXP_s113 + 1.7159 w_k1_DIS_s113
                                 + 1.39468 w_k1_FDX_s113
                                 + 1.60641 w_k1_IBM_s113
                                 + 1.34563 w_k1_INTC_s113
                                 + 1.40446 w_k1_KO_s113 + 1.277 w_k1_MAT_s113
                                 + 1.40815 w_k1_MSFT_s113
                                 + 1.42906 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.848789 w_k1_ADM_s113
                                 + 1.20526 w_k1_AXP_s113
                                 + 0.934054 w_k1_DIS_s113
                                 + 1.0387 w_k1_FDX_s113 + 1.2858 w_k1_IBM_s113
                                 + 0.995604 w_k1_INTC_s113
                                 + 0.956009 w_k1_KO_s113
                                 + 1.50584 w_k1_MAT_s113
                                 + 0.96679 w_k1_MSFT_s113
                                 + 1.11859 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.28908 w_k1_ADM_s113
                                 + 1.38761 w_k1_AXP_s113
                                 + 1.46208 w_k1_DIS_s113 + 1.2919 w_k1_FDX_s113
                                 + 1.22543 w_k1_IBM_s113
                                 + 1.35966 w_k1_INTC_s113
                                 + 1.27351 w_k1_KO_s113 + 1.10617 w_k1_MAT_s113
                                 + 1.20117 w_k1_MSFT_s113 + 1.1585 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.819869 w_k1_ADM_s113
                                 + 0.804427 w_k1_AXP_s113
                                 + 0.691672 w_k1_DIS_s113
                                 + 0.838994 w_k1_FDX_s113
                                 + 0.766148 w_k1_IBM_s113
                                 + 0.820008 w_k1_INTC_s113
                                 + 0.750289 w_k1_KO_s113
                                 + 0.872219 w_k1_MAT_s113
                                 + 0.748133 w_k1_MSFT_s113
                                 + 0.888228 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.752352 w_k1_ADM_s113
                                 + 0.72167 w_k1_AXP_s113
                                 + 0.634334 w_k1_DIS_s113
                                 + 0.72682 w_k1_FDX_s113
                                 + 1.10117 w_k1_IBM_s113
                                 + 0.679919 w_k1_INTC_s113
                                 + 0.813076 w_k1_KO_s113
                                 + 0.954361 w_k1_MAT_s113
                                 + 0.69711 w_k1_MSFT_s113
                                 + 0.833003 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.09378 w_k1_ADM_s113
                                 + 0.751626 w_k1_AXP_s113
                                 + 0.79505 w_k1_DIS_s113
                                 + 0.919516 w_k1_FDX_s113
                                 + 0.81141 w_k1_IBM_s113
                                 + 0.828698 w_k1_INTC_s113
                                 + 0.794218 w_k1_KO_s113
                                 + 0.577815 w_k1_MAT_s113
                                 + 0.827454 w_k1_MSFT_s113
                                 + 0.981538 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.40262 w_k1_ADM_s114
                                 + 1.2457 w_k1_AXP_s114 + 1.5911 w_k1_DIS_s114
                                 + 1.35705 w_k1_FDX_s114
                                 + 1.36283 w_k1_IBM_s114
                                 + 1.32279 w_k1_INTC_s114
                                 + 1.43954 w_k1_KO_s114 + 1.2689 w_k1_MAT_s114
                                 + 1.2367 w_k1_MSFT_s114 + 1.09166 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.17288 w_k1_ADM_s114
                                 + 1.05986 w_k1_AXP_s114
                                 + 0.826457 w_k1_DIS_s114
                                 + 0.916953 w_k1_FDX_s114
                                 + 1.55901 w_k1_IBM_s114
                                 + 0.860942 w_k1_INTC_s114
                                 + 1.05892 w_k1_KO_s114 + 1.96248 w_k1_MAT_s114
                                 + 0.797705 w_k1_MSFT_s114
                                 + 1.07954 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.06962 w_k1_ADM_s114
                                 + 0.879437 w_k1_AXP_s114
                                 + 1.08334 w_k1_DIS_s114
                                 + 1.10984 w_k1_FDX_s114
                                 + 0.785625 w_k1_IBM_s114
                                 + 0.89305 w_k1_INTC_s114
                                 + 0.903557 w_k1_KO_s114
                                 + 0.637891 w_k1_MAT_s114
                                 + 1.00822 w_k1_MSFT_s114
                                 + 0.949929 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.26141 w_k1_ADM_s114
                                 + 0.94438 w_k1_AXP_s114
                                 + 0.998814 w_k1_DIS_s114
                                 + 1.06668 w_k1_FDX_s114
                                 + 1.11288 w_k1_IBM_s114
                                 + 1.08042 w_k1_INTC_s114
                                 + 1.08057 w_k1_KO_s114 + 1.15917 w_k1_MAT_s114
                                 + 0.985882 w_k1_MSFT_s114
                                 + 1.02228 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.938078 w_k1_ADM_s114
                                 + 0.841629 w_k1_AXP_s114
                                 + 1.19279 w_k1_DIS_s114
                                 + 0.956659 w_k1_FDX_s114
                                 + 1.38888 w_k1_IBM_s114
                                 + 0.881254 w_k1_INTC_s114
                                 + 1.25149 w_k1_KO_s114 + 1.29633 w_k1_MAT_s114
                                 + 0.937136 w_k1_MSFT_s114
                                 + 0.990744 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.884723 w_k1_ADM_s114
                                 + 0.840634 w_k1_AXP_s114
                                 + 0.679992 w_k1_DIS_s114
                                 + 0.806696 w_k1_FDX_s114
                                 + 1.04072 w_k1_IBM_s114
                                 + 0.923437 w_k1_INTC_s114
                                 + 0.918185 w_k1_KO_s114
                                 + 1.02451 w_k1_MAT_s114
                                 + 0.863868 w_k1_MSFT_s114
                                 + 0.928201 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.989431 w_k1_ADM_s114
                                 + 0.65826 w_k1_AXP_s114
                                 + 0.74283 w_k1_DIS_s114
                                 + 0.884144 w_k1_FDX_s114
                                 + 0.753672 w_k1_IBM_s114
                                 + 0.83266 w_k1_INTC_s114
                                 + 0.843697 w_k1_KO_s114
                                 + 0.773775 w_k1_MAT_s114
                                 + 0.833189 w_k1_MSFT_s114
                                 + 0.863881 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.31992 w_k1_ADM_s114
                                 + 1.22607 w_k1_AXP_s114
                                 + 1.28797 w_k1_DIS_s114
                                 + 1.29239 w_k1_FDX_s114
                                 + 0.84165 w_k1_IBM_s114
                                 + 1.27305 w_k1_INTC_s114
                                 + 1.13727 w_k1_KO_s114
                                 + 0.809535 w_k1_MAT_s114
                                 + 1.14447 w_k1_MSFT_s114
                                 + 1.04888 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.21866 w_k1_ADM_s115
                                 + 1.16317 w_k1_AXP_s115
                                 + 1.34808 w_k1_DIS_s115
                                 + 1.10602 w_k1_FDX_s115
                                 + 1.41964 w_k1_IBM_s115
                                 + 0.943152 w_k1_INTC_s115
                                 + 1.47584 w_k1_KO_s115 + 1.7382 w_k1_MAT_s115
                                 + 1.09339 w_k1_MSFT_s115
                                 + 1.20082 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.925564 w_k1_ADM_s115
                                 + 0.627529 w_k1_AXP_s115
                                 + 0.791254 w_k1_DIS_s115
                                 + 0.852544 w_k1_FDX_s115
                                 + 0.735751 w_k1_IBM_s115
                                 + 0.821185 w_k1_INTC_s115
                                 + 0.834851 w_k1_KO_s115
                                 + 0.727324 w_k1_MAT_s115
                                 + 0.88152 w_k1_MSFT_s115
                                 + 0.839459 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.03597 w_k1_ADM_s115
                                 + 0.861583 w_k1_AXP_s115
                                 + 0.728844 w_k1_DIS_s115
                                 + 0.80874 w_k1_FDX_s115
                                 + 0.868587 w_k1_IBM_s115
                                 + 0.749073 w_k1_INTC_s115
                                 + 0.896437 w_k1_KO_s115
                                 + 1.09759 w_k1_MAT_s115
                                 + 0.884624 w_k1_MSFT_s115
                                 + 0.939288 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.55694 w_k1_ADM_s115
                                 + 1.42091 w_k1_AXP_s115
                                 + 1.36508 w_k1_DIS_s115
                                 + 1.30331 w_k1_FDX_s115
                                 + 1.05262 w_k1_IBM_s115
                                 + 0.940665 w_k1_INTC_s115
                                 + 1.20183 w_k1_KO_s115 + 1.33334 w_k1_MAT_s115
                                 + 1.22767 w_k1_MSFT_s115
                                 + 1.15639 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02177 w_k1_ADM_s115
                                 + 1.13321 w_k1_AXP_s115
                                 + 0.797878 w_k1_DIS_s115
                                 + 0.879306 w_k1_FDX_s115
                                 + 1.18545 w_k1_IBM_s115
                                 + 0.86891 w_k1_INTC_s115
                                 + 1.05966 w_k1_KO_s115 + 1.76942 w_k1_MAT_s115
                                 + 0.909755 w_k1_MSFT_s115
                                 + 0.970955 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.09448 w_k1_ADM_s115
                                 + 0.866152 w_k1_AXP_s115
                                 + 1.03093 w_k1_DIS_s115
                                 + 0.940064 w_k1_FDX_s115
                                 + 0.990935 w_k1_IBM_s115
                                 + 0.903901 w_k1_INTC_s115
                                 + 0.946843 w_k1_KO_s115
                                 + 0.994944 w_k1_MAT_s115
                                 + 1.01368 w_k1_MSFT_s115
                                 + 0.979786 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.40296 w_k1_ADM_s115
                                 + 0.854922 w_k1_AXP_s115
                                 + 1.11049 w_k1_DIS_s115
                                 + 1.03744 w_k1_FDX_s115
                                 + 0.779474 w_k1_IBM_s115
                                 + 0.79025 w_k1_INTC_s115
                                 + 1.09317 w_k1_KO_s115 + 0.76033 w_k1_MAT_s115
                                 + 1.01098 w_k1_MSFT_s115
                                 + 1.00221 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.969597 w_k1_ADM_s115
                                 + 0.672898 w_k1_AXP_s115
                                 + 0.963168 w_k1_DIS_s115
                                 + 0.736767 w_k1_FDX_s115
                                 + 1.39393 w_k1_IBM_s115
                                 + 0.68272 w_k1_INTC_s115 + 1.1442 w_k1_KO_s115
                                 + 1.1457 w_k1_MAT_s115
                                 + 0.788324 w_k1_MSFT_s115
                                 + 0.858402 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.26674 w_k1_ADM_s116
                                 + 1.13821 w_k1_AXP_s116
                                 + 0.887115 w_k1_DIS_s116
                                 + 0.983513 w_k1_FDX_s116
                                 + 1.27601 w_k1_IBM_s116
                                 + 0.739842 w_k1_INTC_s116
                                 + 0.916207 w_k1_KO_s116
                                 + 1.34466 w_k1_MAT_s116
                                 + 0.859412 w_k1_MSFT_s116
                                 + 0.986581 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.0166 w_k1_ADM_s116
                                 + 0.792545 w_k1_AXP_s116
                                 + 0.919283 w_k1_DIS_s116
                                 + 0.923466 w_k1_FDX_s116
                                 + 0.97163 w_k1_IBM_s116
                                 + 0.8443 w_k1_INTC_s116
                                 + 0.889744 w_k1_KO_s116
                                 + 0.977757 w_k1_MAT_s116
                                 + 1.09267 w_k1_MSFT_s116
                                 + 0.967136 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.26877 w_k1_ADM_s116
                                 + 1.14305 w_k1_AXP_s116
                                 + 1.40694 w_k1_DIS_s116
                                 + 1.22115 w_k1_FDX_s116 + 1.341 w_k1_IBM_s116
                                 + 0.981203 w_k1_INTC_s116
                                 + 1.16913 w_k1_KO_s116 + 1.33009 w_k1_MAT_s116
                                 + 1.13785 w_k1_MSFT_s116
                                 + 1.17808 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.22899 w_k1_ADM_s116
                                 + 1.50934 w_k1_AXP_s116
                                 + 1.13081 w_k1_DIS_s116
                                 + 1.28036 w_k1_FDX_s116
                                 + 0.813576 w_k1_IBM_s116
                                 + 0.939634 w_k1_INTC_s116
                                 + 0.860604 w_k1_KO_s116
                                 + 1.40701 w_k1_MAT_s116
                                 + 1.21969 w_k1_MSFT_s116
                                 + 1.12503 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.20599 w_k1_ADM_s116
                                 + 1.06288 w_k1_AXP_s116
                                 + 1.28146 w_k1_DIS_s116 + 1.1625 w_k1_FDX_s116
                                 + 0.826944 w_k1_IBM_s116
                                 + 0.982956 w_k1_INTC_s116
                                 + 0.995775 w_k1_KO_s116
                                 + 0.772806 w_k1_MAT_s116
                                 + 1.26402 w_k1_MSFT_s116
                                 + 1.05319 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.24996 w_k1_ADM_s116
                                 + 1.47555 w_k1_AXP_s116
                                 + 1.54127 w_k1_DIS_s116
                                 + 1.32897 w_k1_FDX_s116
                                 + 1.81257 w_k1_IBM_s116
                                 + 1.23953 w_k1_INTC_s116
                                 + 1.32138 w_k1_KO_s116 + 1.79368 w_k1_MAT_s116
                                 + 1.39701 w_k1_MSFT_s116
                                 + 1.12939 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.937349 w_k1_ADM_s116
                                 + 0.831061 w_k1_AXP_s116
                                 + 0.735516 w_k1_DIS_s116
                                 + 0.981928 w_k1_FDX_s116
                                 + 0.560733 w_k1_IBM_s116
                                 + 0.71105 w_k1_INTC_s116
                                 + 0.636878 w_k1_KO_s116 + 0.9089 w_k1_MAT_s116
                                 + 0.933472 w_k1_MSFT_s116
                                 + 0.94155 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.928185 w_k1_ADM_s116
                                 + 0.862172 w_k1_AXP_s116
                                 + 0.737978 w_k1_DIS_s116
                                 + 0.841174 w_k1_FDX_s116
                                 + 0.729054 w_k1_IBM_s116
                                 + 0.738174 w_k1_INTC_s116
                                 + 0.772654 w_k1_KO_s116
                                 + 0.948305 w_k1_MAT_s116
                                 + 0.834337 w_k1_MSFT_s116
                                 + 0.92909 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.34026 w_k1_ADM_s117
                                 + 1.04218 w_k1_AXP_s117
                                 + 1.20281 w_k1_DIS_s117 + 1.2022 w_k1_FDX_s117
                                 + 0.741545 w_k1_IBM_s117
                                 + 1.21057 w_k1_INTC_s117
                                 + 0.912491 w_k1_KO_s117
                                 + 0.815523 w_k1_MAT_s117
                                 + 1.15796 w_k1_MSFT_s117
                                 + 0.952769 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.63631 w_k1_ADM_s117
                                 + 1.48152 w_k1_AXP_s117
                                 + 1.86026 w_k1_DIS_s117
                                 + 1.41301 w_k1_FDX_s117
                                 + 1.30034 w_k1_IBM_s117
                                 + 1.26684 w_k1_INTC_s117
                                 + 1.30916 w_k1_KO_s117 + 1.05982 w_k1_MAT_s117
                                 + 1.24057 w_k1_MSFT_s117
                                 + 1.16483 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.10014 w_k1_ADM_s117
                                 + 0.841119 w_k1_AXP_s117
                                 + 0.939972 w_k1_DIS_s117
                                 + 0.96637 w_k1_FDX_s117
                                 + 0.665967 w_k1_IBM_s117
                                 + 1.00563 w_k1_INTC_s117
                                 + 0.814597 w_k1_KO_s117
                                 + 0.607542 w_k1_MAT_s117
                                 + 0.983795 w_k1_MSFT_s117
                                 + 0.923162 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.16933 w_k1_ADM_s117
                                 + 1.03309 w_k1_AXP_s117
                                 + 0.985121 w_k1_DIS_s117
                                 + 0.900031 w_k1_FDX_s117
                                 + 1.70353 w_k1_IBM_s117
                                 + 0.962965 w_k1_INTC_s117
                                 + 1.1275 w_k1_KO_s117 + 1.59783 w_k1_MAT_s117
                                 + 0.910616 w_k1_MSFT_s117
                                 + 0.990435 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.848962 w_k1_ADM_s117
                                 + 0.683008 w_k1_AXP_s117
                                 + 0.674978 w_k1_DIS_s117
                                 + 0.758986 w_k1_FDX_s117
                                 + 0.785501 w_k1_IBM_s117
                                 + 0.825779 w_k1_INTC_s117
                                 + 0.822852 w_k1_KO_s117
                                 + 0.863077 w_k1_MAT_s117
                                 + 0.801631 w_k1_MSFT_s117
                                 + 0.846416 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00204 w_k1_ADM_s117
                                 + 0.982003 w_k1_AXP_s117
                                 + 0.87459 w_k1_DIS_s117
                                 + 0.823483 w_k1_FDX_s117
                                 + 1.19224 w_k1_IBM_s117
                                 + 0.938325 w_k1_INTC_s117
                                 + 0.928778 w_k1_KO_s117
                                 + 0.997043 w_k1_MAT_s117
                                 + 0.822445 w_k1_MSFT_s117
                                 + 0.827839 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.12529 w_k1_ADM_s117
                                 + 1.12704 w_k1_AXP_s117
                                 + 1.17968 w_k1_DIS_s117
                                 + 1.05586 w_k1_FDX_s117
                                 + 0.983392 w_k1_IBM_s117
                                 + 1.2316 w_k1_INTC_s117 + 1.08546 w_k1_KO_s117
                                 + 1.12209 w_k1_MAT_s117
                                 + 1.20724 w_k1_MSFT_s117
                                 + 0.961596 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.21489 w_k1_ADM_s117
                                 + 0.994425 w_k1_AXP_s117
                                 + 1.23992 w_k1_DIS_s117
                                 + 0.949713 w_k1_FDX_s117
                                 + 1.35697 w_k1_IBM_s117
                                 + 1.08839 w_k1_INTC_s117
                                 + 1.31174 w_k1_KO_s117 + 1.02348 w_k1_MAT_s117
                                 + 1.12917 w_k1_MSFT_s117
                                 + 1.11475 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0119616375397474 w_k1_ADM_s000
                                 + 0.0151151739185974 w_k1_ADM_s001
                                 + 0.0194825975064011 w_k1_ADM_s002
                                 + 0.0143281728984161 w_k1_ADM_s003
                                 + 0.0119796212713771 w_k1_ADM_s004
                                 + 0.0485484365826945 w_k1_ADM_s005
                                 + 0.0159829208460915 w_k1_ADM_s006
                                 + 0.0321224633133235 w_k1_ADM_s007
                                 + 0.0109248677529376 w_k1_ADM_s010
                                 + 0.00899490886474223 w_k1_ADM_s011
                                 + 0.00651640188515519 w_k1_ADM_s012
                                 + 0.0120423966248591 w_k1_ADM_s013
                                 + 0.0122578662995702 w_k1_ADM_s014
                                 + 0.026732851472072 w_k1_ADM_s015
                                 + 0.0130886421087496 w_k1_ADM_s016
                                 + 0.0161213965842889 w_k1_ADM_s017
                                 + 0.0599268093211366 w_k1_ADM_s100
                                 + 0.0202490629744841 w_k1_ADM_s101
                                 + 0.0185799506473477 w_k1_ADM_s102
                                 + 0.0292660008056393 w_k1_ADM_s103
                                 + 0.0181669951073413 w_k1_ADM_s104
                                 + 0.0137630786486285 w_k1_ADM_s105
                                 + 0.048902517818414 w_k1_ADM_s106
                                 + 0.0154373994467514 w_k1_ADM_s107
                                 + 0.0381093828381553 w_k1_ADM_s110
                                 + 0.0728431275939375 w_k1_ADM_s111
                                 + 0.0300523133876411 w_k1_ADM_s112
                                 + 0.0522461511133543 w_k1_ADM_s113
                                 + 0.150276547497467 w_k1_ADM_s114
                                 + 0.0695762058895531 w_k1_ADM_s115
                                 + 0.0571392865450566 w_k1_ADM_s116
                                 + 0.0803280591806217 w_k1_ADM_s117
                                 + 0.0121996475437193 w_k1_AXP_s000
                                 + 0.0157501739980223 w_k1_AXP_s001
                                 + 0.024078507989038 w_k1_AXP_s002
                                 + 0.0146974927002895 w_k1_AXP_s003
                                 + 0.0131837152110961 w_k1_AXP_s004
                                 + 0.0476319268992373 w_k1_AXP_s005
                                 + 0.0144491414123621 w_k1_AXP_s006
                                 + 0.039604460933917 w_k1_AXP_s007
                                 + 0.0111157529565609 w_k1_AXP_s010
                                 + 0.00910875365476682 w_k1_AXP_s011
                                 + 0.00638631975962321 w_k1_AXP_s012
                                 + 0.0103527470680352 w_k1_AXP_s013
                                 + 0.014860003557243 w_k1_AXP_s014
                                 + 0.0237130356657833 w_k1_AXP_s015
                                 + 0.0143565332940399 w_k1_AXP_s016
                                 + 0.0153573481183121 w_k1_AXP_s017
                                 + 0.0601082986219026 w_k1_AXP_s100
                                 + 0.0180154547705974 w_k1_AXP_s101
                                 + 0.0170603421073496 w_k1_AXP_s102
                                 + 0.0272784080641556 w_k1_AXP_s103
                                 + 0.0202107672610587 w_k1_AXP_s104
                                 + 0.0168315222333544 w_k1_AXP_s105
                                 + 0.0369932262884135 w_k1_AXP_s106
                                 + 0.0170080063773661 w_k1_AXP_s107
                                 + 0.0358751678693159 w_k1_AXP_s110
                                 + 0.0740012081149893 w_k1_AXP_s111
                                 + 0.0321239752545463 w_k1_AXP_s112
                                 + 0.0507654902181145 w_k1_AXP_s113
                                 + 0.124544820080272 w_k1_AXP_s114
                                 + 0.0594872622067242 w_k1_AXP_s115
                                 + 0.0529841353937128 w_k1_AXP_s116
                                 + 0.0711190254056668 w_k1_AXP_s117
                                 + 0.0115610262813666 w_k1_DIS_s000
                                 + 0.0158184040170225 w_k1_DIS_s001
                                 + 0.0232144356269558 w_k1_DIS_s002
                                 + 0.0149300212339042 w_k1_DIS_s003
                                 + 0.012527000695896 w_k1_DIS_s004
                                 + 0.0527424050007366 w_k1_DIS_s005
                                 + 0.0154148120879585 w_k1_DIS_s006
                                 + 0.0389296932710875 w_k1_DIS_s007
                                 + 0.011123627673746 w_k1_DIS_s010
                                 + 0.00921095842071508 w_k1_DIS_s011
                                 + 0.00623235359555416 w_k1_DIS_s012
                                 + 0.0104322263036703 w_k1_DIS_s013
                                 + 0.0135823531464839 w_k1_DIS_s014
                                 + 0.0242107150611233 w_k1_DIS_s015
                                 + 0.0159425101387299 w_k1_DIS_s016
                                 + 0.0157609466684762 w_k1_DIS_s017
                                 + 0.062110028149913 w_k1_DIS_s100
                                 + 0.0183223202284096 w_k1_DIS_s101
                                 + 0.0177589905667705 w_k1_DIS_s102
                                 + 0.0287236434273002 w_k1_DIS_s103
                                 + 0.0186810948235993 w_k1_DIS_s104
                                 + 0.0170277434070888 w_k1_DIS_s105
                                 + 0.043489533069422 w_k1_DIS_s106
                                 + 0.0163928258397154 w_k1_DIS_s107
                                 + 0.0368834154749188 w_k1_DIS_s110
                                 + 0.0738925762855836 w_k1_DIS_s111
                                 + 0.0320312981165373 w_k1_DIS_s112
                                 + 0.0493377891523324 w_k1_DIS_s113
                                 + 0.135038540149443 w_k1_DIS_s114
                                 + 0.0622306668506444 w_k1_DIS_s115
                                 + 0.0505591953741083 w_k1_DIS_s116
                                 + 0.0763059109545965 w_k1_DIS_s117
                                 + 0.0115840347999139 w_k1_FDX_s000
                                 + 0.0156384104256944 w_k1_FDX_s001
                                 + 0.0237020943126279 w_k1_FDX_s002
                                 + 0.0141307599707385 w_k1_FDX_s003
                                 + 0.0140883529067281 w_k1_FDX_s004
                                 + 0.0494453425761835 w_k1_FDX_s005
                                 + 0.0139580570475828 w_k1_FDX_s006
                                 + 0.0375170855907349 w_k1_FDX_s007
                                 + 0.0113648521263742 w_k1_FDX_s010
                                 + 0.00874386282298113 w_k1_FDX_s011
                                 + 0.00651777539637726 w_k1_FDX_s012
                                 + 0.0108986441907077 w_k1_FDX_s013
                                 + 0.0133719922973787 w_k1_FDX_s014
                                 + 0.0239996871382674 w_k1_FDX_s015
                                 + 0.0144100599060123 w_k1_FDX_s016
                                 + 0.0153605574236698 w_k1_FDX_s017
                                 + 0.0619854176335985 w_k1_FDX_s100
                                 + 0.0175721538327251 w_k1_FDX_s101
                                 + 0.0167557164606911 w_k1_FDX_s102
                                 + 0.0277272631163191 w_k1_FDX_s103
                                 + 0.0179346314134585 w_k1_FDX_s104
                                 + 0.0162420892676419 w_k1_FDX_s105
                                 + 0.042807917050061 w_k1_FDX_s106
                                 + 0.0156890531961365 w_k1_FDX_s107
                                 + 0.0365832912379853 w_k1_FDX_s110
                                 + 0.0717759949859212 w_k1_FDX_s111
                                 + 0.0319675832275112 w_k1_FDX_s112
                                 + 0.0510309390894922 w_k1_FDX_s113
                                 + 0.138617283404255 w_k1_FDX_s114
                                 + 0.0590136891235585 w_k1_FDX_s115
                                 + 0.0524013133301715 w_k1_FDX_s116
                                 + 0.0701493611488668 w_k1_FDX_s117
                                 + 0.0109530686210021 w_k1_IBM_s000
                                 + 0.016778637323118 w_k1_IBM_s001
                                 + 0.0229601384500443 w_k1_IBM_s002
                                 + 0.0138883566205512 w_k1_IBM_s003
                                 + 0.0139300134475017 w_k1_IBM_s004
                                 + 0.0500868945979209 w_k1_IBM_s005
                                 + 0.0148603677366589 w_k1_IBM_s006
                                 + 0.0400630968843881 w_k1_IBM_s007
                                 + 0.0118799217816611 w_k1_IBM_s010
                                 + 0.00918716212973296 w_k1_IBM_s011
                                 + 0.00599288235914188 w_k1_IBM_s012
                                 + 0.0112053435488065 w_k1_IBM_s013
                                 + 0.0136470228607095 w_k1_IBM_s014
                                 + 0.0242449550924201 w_k1_IBM_s015
                                 + 0.0144266526928793 w_k1_IBM_s016
                                 + 0.0155938911528673 w_k1_IBM_s017
                                 + 0.0596700448367895 w_k1_IBM_s100
                                 + 0.0180009202415342 w_k1_IBM_s101
                                 + 0.0172072294856315 w_k1_IBM_s102
                                 + 0.0274513297069837 w_k1_IBM_s103
                                 + 0.0184414464179505 w_k1_IBM_s104
                                 + 0.0163391508918624 w_k1_IBM_s105
                                 + 0.0431313780973552 w_k1_IBM_s106
                                 + 0.016234934378688 w_k1_IBM_s107
                                 + 0.0370013427467255 w_k1_IBM_s110
                                 + 0.0755776917836248 w_k1_IBM_s111
                                 + 0.032659016071422 w_k1_IBM_s112
                                 + 0.0495791984753703 w_k1_IBM_s113
                                 + 0.142538329664166 w_k1_IBM_s114
                                 + 0.061394237466039 w_k1_IBM_s115
                                 + 0.0521125903186226 w_k1_IBM_s116
                                 + 0.0727446975652671 w_k1_IBM_s117
                                 + 0.0106028187259547 w_k1_INTC_s000
                                 + 0.0164642823474935 w_k1_INTC_s001
                                 + 0.0241082610750549 w_k1_INTC_s002
                                 + 0.0133449669585901 w_k1_INTC_s003
                                 + 0.0136169619431048 w_k1_INTC_s004
                                 + 0.0476412131026105 w_k1_INTC_s005
                                 + 0.0137684301707237 w_k1_INTC_s006
                                 + 0.0403307329418597 w_k1_INTC_s007
                                 + 0.0110705600183501 w_k1_INTC_s010
                                 + 0.00973941641221367 w_k1_INTC_s011
                                 + 0.00634823420786602 w_k1_INTC_s012
                                 + 0.0122853567657591 w_k1_INTC_s013
                                 + 0.0139972918297677 w_k1_INTC_s014
                                 + 0.0227824828887825 w_k1_INTC_s015
                                 + 0.0132287678169595 w_k1_INTC_s016
                                 + 0.0152263205212994 w_k1_INTC_s017
                                 + 0.0620796179739356 w_k1_INTC_s100
                                 + 0.0166305106091792 w_k1_INTC_s101
                                 + 0.0158486792246835 w_k1_INTC_s102
                                 + 0.0245489539526206 w_k1_INTC_s103
                                 + 0.0168634845900911 w_k1_INTC_s104
                                 + 0.0140976004550167 w_k1_INTC_s105
                                 + 0.0425348387163399 w_k1_INTC_s106
                                 + 0.0158051888646928 w_k1_INTC_s107
                                 + 0.0358215232082703 w_k1_INTC_s110
                                 + 0.0714765078971478 w_k1_INTC_s111
                                 + 0.0392574642924175 w_k1_INTC_s112
                                 + 0.0496969497002483 w_k1_INTC_s113
                                 + 0.136125010379249 w_k1_INTC_s114
                                 + 0.0508266526301258 w_k1_INTC_s115
                                 + 0.0431516999576839 w_k1_INTC_s116
                                 + 0.0762259438762429 w_k1_INTC_s117
                                 + 0.0109104154386284 w_k1_KO_s000
                                 + 0.0150643776995192 w_k1_KO_s001
                                 + 0.0265577864509385 w_k1_KO_s002
                                 + 0.0136810471713418 w_k1_KO_s003
                                 + 0.0124542507954013 w_k1_KO_s004
                                 + 0.0472483383273273 w_k1_KO_s005
                                 + 0.0141847675879344 w_k1_KO_s006
                                 + 0.0399313764983146 w_k1_KO_s007
                                 + 0.0124293364897645 w_k1_KO_s010
                                 + 0.00799902047542527 w_k1_KO_s011
                                 + 0.00607634312637716 w_k1_KO_s012
                                 + 0.0099993227519656 w_k1_KO_s013
                                 + 0.0131520908890557 w_k1_KO_s014
                                 + 0.0223277213397563 w_k1_KO_s015
                                 + 0.015150953954638 w_k1_KO_s016
                                 + 0.0150739471281751 w_k1_KO_s017
                                 + 0.0600790910091575 w_k1_KO_s100
                                 + 0.0182350303773671 w_k1_KO_s101
                                 + 0.0175656050496725 w_k1_KO_s102
                                 + 0.0285509492052237 w_k1_KO_s103
                                 + 0.0170470949895661 w_k1_KO_s104
                                 + 0.0164987288135532 w_k1_KO_s105
                                 + 0.0452482483538923 w_k1_KO_s106
                                 + 0.015215300850755 w_k1_KO_s107
                                 + 0.0361698357196564 w_k1_KO_s110
                                 + 0.0693607284789091 w_k1_KO_s111
                                 + 0.0311488679802526 w_k1_KO_s112
                                 + 0.0469712061893433 w_k1_KO_s113
                                 + 0.142344013432219 w_k1_KO_s114
                                 + 0.0641965879098742 w_k1_KO_s115
                                 + 0.0466516704569788 w_k1_KO_s116
                                 + 0.0722443705148655 w_k1_KO_s117
                                 + 0.0114451958110281 w_k1_MAT_s000
                                 + 0.0168163082620627 w_k1_MAT_s001
                                 + 0.0241877607944423 w_k1_MAT_s002
                                 + 0.0163478107400826 w_k1_MAT_s003
                                 + 0.0150546458701831 w_k1_MAT_s004
                                 + 0.037623764382864 w_k1_MAT_s005
                                 + 0.0139969884588749 w_k1_MAT_s006
                                 + 0.0425965763767891 w_k1_MAT_s007
                                 + 0.0100715403495349 w_k1_MAT_s010
                                 + 0.00886694697555841 w_k1_MAT_s011
                                 + 0.00529739205868235 w_k1_MAT_s012
                                 + 0.0104240733855133 w_k1_MAT_s013
                                 + 0.0148752839634587 w_k1_MAT_s014
                                 + 0.0215881137567115 w_k1_MAT_s015
                                 + 0.0142964136942891 w_k1_MAT_s016
                                 + 0.0173219183498627 w_k1_MAT_s017
                                 + 0.052812142223241 w_k1_MAT_s100
                                 + 0.0174478836914367 w_k1_MAT_s101
                                 + 0.0155197807342476 w_k1_MAT_s102
                                 + 0.0271122280172349 w_k1_MAT_s103
                                 + 0.0186251875178282 w_k1_MAT_s104
                                 + 0.0177242949505962 w_k1_MAT_s105
                                 + 0.0449206926001021 w_k1_MAT_s106
                                 + 0.0141735287141559 w_k1_MAT_s107
                                 + 0.037107431605047 w_k1_MAT_s110
                                 + 0.0564356834605672 w_k1_MAT_s111
                                 + 0.0358137941971865 w_k1_MAT_s112
                                 + 0.0467531702606059 w_k1_MAT_s113
                                 + 0.141789659409551 w_k1_MAT_s114
                                 + 0.0723135090318966 w_k1_MAT_s115
                                 + 0.0589210263670012 w_k1_MAT_s116
                                 + 0.070687574112412 w_k1_MAT_s117
                                 + 0.0116382627702777 w_k1_MSFT_s000
                                 + 0.016281622935996 w_k1_MSFT_s001
                                 + 0.0231620579478101 w_k1_MSFT_s002
                                 + 0.0144268625814338 w_k1_MSFT_s003
                                 + 0.0134620644283 w_k1_MSFT_s004
                                 + 0.0498740504927835 w_k1_MSFT_s005
                                 + 0.0143053359820282 w_k1_MSFT_s006
                                 + 0.0417054428890951 w_k1_MSFT_s007
                                 + 0.0124051490037822 w_k1_MSFT_s010
                                 + 0.00886327539896513 w_k1_MSFT_s011
                                 + 0.00562973525701206 w_k1_MSFT_s012
                                 + 0.0105517338688902 w_k1_MSFT_s013
                                 + 0.0128898898236031 w_k1_MSFT_s014
                                 + 0.0246448594547317 w_k1_MSFT_s015
                                 + 0.0147265776595331 w_k1_MSFT_s016
                                 + 0.0157851133536024 w_k1_MSFT_s017
                                 + 0.0594288593398767 w_k1_MSFT_s100
                                 + 0.0194923275126672 w_k1_MSFT_s101
                                 + 0.0155862206514228 w_k1_MSFT_s102
                                 + 0.0265240700698726 w_k1_MSFT_s103
                                 + 0.017342795470189 w_k1_MSFT_s104
                                 + 0.0153605697924933 w_k1_MSFT_s105
                                 + 0.0463646192111634 w_k1_MSFT_s106
                                 + 0.0152192938962801 w_k1_MSFT_s107
                                 + 0.0380704729755405 w_k1_MSFT_s110
                                 + 0.072560683829782 w_k1_MSFT_s111
                                 + 0.0350231999675841 w_k1_MSFT_s112
                                 + 0.0483920795366654 w_k1_MSFT_s113
                                 + 0.131026566045461 w_k1_MSFT_s114
                                 + 0.0595547968775289 w_k1_MSFT_s115
                                 + 0.0520452199382926 w_k1_MSFT_s116
                                 + 0.0738242430242301 w_k1_MSFT_s117
                                 + 0.0112962410962463 w_k1_PG_s000
                                 + 0.0167149608724245 w_k1_PG_s001
                                 + 0.0244861999851613 w_k1_PG_s002
                                 + 0.0138348197082165 w_k1_PG_s003
                                 + 0.0138661238919805 w_k1_PG_s004
                                 + 0.0465056319316948 w_k1_PG_s005
                                 + 0.01353067044871 w_k1_PG_s006
                                 + 0.0364235835892671 w_k1_PG_s007
                                 + 0.0118586933997994 w_k1_PG_s010
                                 + 0.00824544273978447 w_k1_PG_s011
                                 + 0.00665047743101995 w_k1_PG_s012
                                 + 0.00988872022764477 w_k1_PG_s013
                                 + 0.0133326180039412 w_k1_PG_s014
                                 + 0.0232838770371462 w_k1_PG_s015
                                 + 0.0149956217976352 w_k1_PG_s016
                                 + 0.0166505578480343 w_k1_PG_s017
                                 + 0.0608823096425767 w_k1_PG_s100
                                 + 0.0164420864930963 w_k1_PG_s101
                                 + 0.0156848908970919 w_k1_PG_s102
                                 + 0.028197476602773 w_k1_PG_s103
                                 + 0.0179980361165219 w_k1_PG_s104
                                 + 0.0155283352435506 w_k1_PG_s105
                                 + 0.0430711840989756 w_k1_PG_s106
                                 + 0.0147636816153643 w_k1_PG_s107
                                 + 0.0360110050401014 w_k1_PG_s110
                                 + 0.0751705412812766 w_k1_PG_s111
                                 + 0.0316659057314629 w_k1_PG_s112
                                 + 0.0512491718736164 w_k1_PG_s113
                                 + 0.132867525540341 w_k1_PG_s114
                                 + 0.0599255043029751 w_k1_PG_s115
                                 + 0.0519503929389587 w_k1_PG_s116
                                 + 0.0677196292023258 w_k1_PG_s117 >= 1.16
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
