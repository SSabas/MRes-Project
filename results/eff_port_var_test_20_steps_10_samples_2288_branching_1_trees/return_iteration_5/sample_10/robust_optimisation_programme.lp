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
 w_k1_ADM_s0_return_constraint:  1.04118 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.899748 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.12057 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06093 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.771454 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.12283 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.36549 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.695627 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.815986 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.19143 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.927722 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.06328 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.28187 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.97314 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.754257 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.3386 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.983542 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.561125 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.891048 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.01126 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.06541 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.983524 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.31544 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.15683 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.30702 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.56197 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.956792 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.39113 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.06678 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.19028 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.19588 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.844817 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.20888 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.18759 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.45693 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.00907 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.99555 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.637549 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.1532 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.847345 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.01148 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.02714 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.778072 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.22164 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.45968 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.576528 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.70579 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.18899 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.992469 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.39221 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.37218 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.993859 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.812611 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.84786 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.6203 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.500286 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.37323 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.04398 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.10484 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.896428 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.4624 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.827106 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.27312 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.50304 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.75649 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.06822 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.16138 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.03193 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.30823 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.916007 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.38443 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.24402 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.51307 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.927199 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.752167 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.699132 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.14522 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.818816 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.03765 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.0524 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.746259 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.23573 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.89976 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.721414 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.637263 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.64966 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.905772 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.32315 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.3892 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.07234 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.707344 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.82526 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.03952 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.410498 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.14953 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.39994 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.18339 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.693305 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.57415 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.773665 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.28232 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.73121 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.72892 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.19471 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.907522 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.989245 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.40765 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.932895 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.00875 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.51771 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.86982 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.2141 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.808558 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.661038 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.07384 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.905254 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.03311 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.918114 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.822538 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.16867 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.46328 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.703185 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.77207 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.15254 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.945444 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.27249 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.42189 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.990989 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.794261 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.5668 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.2007 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.622011 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.00488 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.13487 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.14058 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.918357 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.19889 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.893697 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.18946 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.39611 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.808117 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.10436 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.976181 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.900648 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.12297 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.897427 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.11766 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.38231 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.40051 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.05614 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.914208 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.685463 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15553 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.7715 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.937399 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.48863 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.01953 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.07074 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.15067 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.04337 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.661455 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.71527 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.929671 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.33506 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.806312 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.27344 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.99956 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.28615 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.993917 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.616846 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.59587 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.52329 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.980932 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.676323 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.73508 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.592587 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.883962 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.12584 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.906176 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.05843 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.872073 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.30762 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.79909 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.34782 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.988131 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.886255 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.35439 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.11913 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.69448 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.06019 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.03074 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.933433 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.00987 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.00043 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.811132 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01036 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.482 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.832754 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.87104 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06923 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.981079 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.12475 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.25423 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.888248 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.845532 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.38079 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.08521 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.644829 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.08864 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.29263 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.16787 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.871949 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.37025 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.831996 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.20401 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.40641 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.948729 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.0817 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.07888 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.13595 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.3105 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.08611 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.01929 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.34966 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.53569 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.24309 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.00966 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.887391 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.08171 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.849673 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.03421 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.27142 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.866468 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.18408 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.34103 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.843293 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.641064 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.59407 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.93714 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.18239 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.03165 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.07585 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.889745 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.34695 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.920436 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.521457 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.23303 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.47519 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.04158 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.713272 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.52084 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.71328 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.17431 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.29577 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.959332 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.17445 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.861132 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.06846 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.35042 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.01223 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.908447 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.00142 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.28515 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.05614 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.754996 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.743489 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.31438 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.787271 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.867891 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.21509 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.979603 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.222 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.10236 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.708108 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.679652 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.85344 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.06944 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.34133 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.845591 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.42434 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.0522 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.59447 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.36004 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.511624 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.90146 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.47884 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.956633 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.771429 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.61947 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.608042 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.792577 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.02779 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.790021 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.997357 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.90312 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.23523 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.86923 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.850237 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.16118 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.729329 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.53529 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.04565 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.453107 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.881866 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.05856 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.934108 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.05458 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.942541 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.767477 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.10222 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.44089 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.770353 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.787868 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.17751 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.933486 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.1591 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.22875 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.03126 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.900886 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.39357 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.0833 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.624396 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.17968 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.39888 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.19943 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.85689 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.3784 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.899712 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.2779 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.37566 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.948245 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.18789 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.02641 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.10112 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.2583 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.958053 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.984326 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.16905 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.34068 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.20581 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.94076 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.757336 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06753 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.894351 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.990108 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.0107 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.935378 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11428 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.33213 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.852193 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.785692 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.31766 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.01288 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.17772 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.07575 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01715 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.894323 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.28614 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.12929 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.698122 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.11891 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.19938 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.12262 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.900926 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.15039 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.839933 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.08582 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.14865 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.886694 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.03846 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.900776 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.986022 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.28048 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.934328 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.03769 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.06735 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.33311 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.06993 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.901703 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.823878 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.664999932736041 z_k1_s0000
                                 - 0.353999082342561 z_k1_s0001
                                 - 1.49599426519849 z_k1_s0002
                                 - 1.3329978751097 z_k1_s0003
                                 - 0.726998521162534 z_k1_s0004
                                 - 0.457999284033161 z_k1_s0005
                                 - 0.630999492891831 z_k1_s0006
                                 - 0.607998863303805 z_k1_s0007
                                 - 0.910001095943556 z_k1_s0010
                                 - 0.315000181595861 z_k1_s0011
                                 - 0.272999900283099 z_k1_s0012
                                 - 0.268999853086647 z_k1_s0013
                                 - 1.149999642731 z_k1_s0014
                                 - 0.346000011743405 z_k1_s0015
                                 - 0.778001680960478 z_k1_s0016
                                 - 0.243999879483798 z_k1_s0017
                                 - 1.06700119798373 z_k1_s0020
                                 - 1.06700119798373 z_k1_s0021
                                 - 2.58900010358411 z_k1_s0022
                                 - 1.12700093630354 z_k1_s0023
                                 - 1.63799716778434 z_k1_s0024
                                 - 1.95400441290903 z_k1_s0025
                                 - 1.63799716778434 z_k1_s0026
                                 - 1.23899675213085 z_k1_s0027
                                 - 0.137999982391136 z_k1_s0030
                                 - 0.498000261498364 z_k1_s0031
                                 - 0.231000063393834 z_k1_s0032
                                 - 0.285999646202002 z_k1_s0033
                                 - 0.438000036980538 z_k1_s0034
                                 - 0.216999922006365 z_k1_s0035
                                 - 0.633000766663471 z_k1_s0036
                                 - 0.22899996690993 z_k1_s0037
                                 - 1.4540005280588 z_k1_s0040
                                 - 2.45900453903667 z_k1_s0041
                                 - 1.65500373445516 z_k1_s0042
                                 - 1.0030001096259 z_k1_s0043
                                 - 2.18599551921324 z_k1_s0044
                                 - 0.85300082072366 z_k1_s0045
                                 - 1.55099800861692 z_k1_s0046
                                 - 0.860000651301053 z_k1_s0047
                                 - 0.414999595954317 z_k1_s0050
                                 - 0.596999797437714 z_k1_s0051
                                 - 0.386999404310898 z_k1_s0052
                                 - 0.827000252867057 z_k1_s0053
                                 - 0.533000851108227 z_k1_s0054
                                 - 0.513999564987202 z_k1_s0055
                                 - 1.59499805080921 z_k1_s0056
                                 - 0.352999768286077 z_k1_s0057
                                 - 0.307999743810174 z_k1_s0060
                                 - 0.321999848460575 z_k1_s0061
                                 - 0.938000432580348 z_k1_s0062
                                 - 0.320999840985547 z_k1_s0063
                                 - 0.47899846354142 z_k1_s0064
                                 - 0.262999772933692 z_k1_s0065
                                 - 0.662000562471283 z_k1_s0066
                                 - 0.361999781961914 z_k1_s0067
                                 - 0.501001817059659 z_k1_s0070
                                 - 1.09400446452729 z_k1_s0071
                                 - 0.374001531434393 z_k1_s0072
                                 - 0.86800119504237 z_k1_s0073
                                 - 0.713003607651492 z_k1_s0074
                                 - 0.525002009096431 z_k1_s0075
                                 - 0.932004723684728 z_k1_s0076
                                 - 0.649000079009134 z_k1_s0077
                                 - 0.306001083083145 z_k1_s0100
                                 - 0.441998904453405 z_k1_s0101
                                 - 0.190000077914365 z_k1_s0102
                                 - 0.600000246045364 z_k1_s0103
                                 - 0.170999963722928 z_k1_s0104
                                 - 0.251999890539051 z_k1_s0105
                                 - 0.458998964624691 z_k1_s0106
                                 - 0.240000098418145 z_k1_s0107
                                 - 0.213001089206649 z_k1_s0110
                                 - 0.211001578981913 z_k1_s0111
                                 - 0.271001885800688 z_k1_s0112
                                 - 0.166001348867833 z_k1_s0113
                                 - 0.169001364208771 z_k1_s0114
                                 - 0.113000527841769 z_k1_s0115
                                 - 0.17900041534012 z_k1_s0116
                                 - 0.178001410231587 z_k1_s0117
                                 - 0.137000258761319 z_k1_s0120
                                 - 0.0820001803696109 z_k1_s0121
                                 - 0.0850001712782169 z_k1_s0122
                                 - 0.107000349675259 z_k1_s0123
                                 - 0.121000429782385 z_k1_s0124
                                 - 0.0790001894610048 z_k1_s0125
                                 - 0.106000475239355 z_k1_s0126
                                 - 0.202000368115197 z_k1_s0127
                                 - 0.0970003364237684 z_k1_s0130
                                 - 0.0770001675589114 z_k1_s0131
                                 - 0.0700000949261893 z_k1_s0132
                                 - 0.0730001518259695 z_k1_s0133
                                 - 0.142000287918997 z_k1_s0134
                                 - 0.113000038554947 z_k1_s0135
                                 - 0.254000006840193 z_k1_s0136
                                 - 0.0760001185256023 z_k1_s0137
                                 - 0.143000895751235 z_k1_s0140
                                 - 0.26800081807697 z_k1_s0141
                                 - 0.138999854210061 z_k1_s0142
                                 - 0.103000241565211 z_k1_s0143
                                 - 0.0610001276113949 z_k1_s0144
                                 - 0.116999961248978 z_k1_s0145
                                 - 0.112000516077402 z_k1_s0146
                                 - 0.11800048688497 z_k1_s0147
                                 - 0.0610000799060347 z_k1_s0150
                                 - 0.0650001367852077 z_k1_s0151
                                 - 0.128000470131216 z_k1_s0152
                                 - 0.06400008506535 z_k1_s0153
                                 - 0.093999911658116 z_k1_s0154
                                 - 0.20400035083344 z_k1_s0155
                                 - 0.047000105829316 z_k1_s0156
                                 - 0.0870001496201436 z_k1_s0157
                                 - 0.280000537042911 z_k1_s0160
                                 - 0.418999608479582 z_k1_s0161
                                 - 0.666001786810271 z_k1_s0162
                                 - 0.238000456486475 z_k1_s0163
                                 - 0.328000393992635 z_k1_s0164
                                 - 0.162000271531889 z_k1_s0165
                                 - 0.42800215322551 z_k1_s0166
                                 - 0.2220005041699 z_k1_s0167
                                 - 0.382000137350632 z_k1_s0170
                                 - 0.107000152089402 z_k1_s0171
                                 - 0.524999752662179 z_k1_s0172
                                 - 0.23599967170321 z_k1_s0173
                                 - 0.173000227464159 z_k1_s0174
                                 - 0.399001302584416 z_k1_s0175
                                 - 0.165000078836807 z_k1_s0176
                                 - 0.205000164373054 z_k1_s0177
                                 - 0.30100022320485 z_k1_s1000
                                 - 0.236000369207301 z_k1_s1001
                                 - 0.135000213033839 z_k1_s1002
                                 - 0.211000225515755 z_k1_s1003
                                 - 0.213000254331094 z_k1_s1004
                                 - 0.401999729877722 z_k1_s1005
                                 - 0.423000248938982 z_k1_s1006
                                 - 0.244000051468266 z_k1_s1007
                                 - 0.272000326391689 z_k1_s1010
                                 - 0.611000355368354 z_k1_s1011
                                 - 0.409002054351977 z_k1_s1012
                                 - 0.701999668252069 z_k1_s1013
                                 - 0.257000411854778 z_k1_s1014
                                 - 0.186000141819214 z_k1_s1015
                                 - 0.224000157193015 z_k1_s1016
                                 - 0.501000560497141 z_k1_s1017
                                 - 0.461999915463242 z_k1_s1020
                                 - 0.171999478634463 z_k1_s1021
                                 - 0.303998234370293 z_k1_s1022
                                 - 0.431997771893826 z_k1_s1023
                                 - 0.496998412805627 z_k1_s1024
                                 - 0.481998675517119 z_k1_s1025
                                 - 0.171999478634463 z_k1_s1026
                                 - 0.147999632073611 z_k1_s1027
                                 - 0.199999942753474 z_k1_s1030
                                 - 0.161000219356216 z_k1_s1031
                                 - 0.0919998409467634 z_k1_s1032
                                 - 0.140999340281971 z_k1_s1033
                                 - 0.0499999856883685 z_k1_s1034
                                 - 0.208999453537987 z_k1_s1035
                                 - 0.0459999204733817 z_k1_s1036
                                 - 0.206999808030011 z_k1_s1037
                                 - 0.363000374793124 z_k1_s1040
                                 - 0.318000761406369 z_k1_s1041
                                 - 0.389000090127578 z_k1_s1042
                                 - 0.238000406805715 z_k1_s1043
                                 - 0.511000451623235 z_k1_s1044
                                 - 0.282000519095917 z_k1_s1045
                                 - 0.526999639981691 z_k1_s1046
                                 - 0.130000231475408 z_k1_s1047
                                 - 0.20999989436684 z_k1_s1050
                                 - 0.118999880007915 z_k1_s1051
                                 - 0.0970001144264811 z_k1_s1052
                                 - 0.112999779940359 z_k1_s1053
                                 - 0.129999762798633 z_k1_s1054
                                 - 0.0899999676143516 z_k1_s1055
                                 - 0.0719999380115049 z_k1_s1056
                                 - 0.0709999814668728 z_k1_s1057
                                 - 0.10400023064893 z_k1_s1060
                                 - 0.0519999683245561 z_k1_s1061
                                 - 0.0529999352058176 z_k1_s1062
                                 - 0.0659999456619453 z_k1_s1063
                                 - 0.062999971518206 z_k1_s1064
                                 - 0.0479999537996007 z_k1_s1065
                                 - 0.209999660060838 z_k1_s1066
                                 - 0.139000173992403 z_k1_s1067
                                 - 0.389000779495294 z_k1_s1070
                                 - 0.30699917361588 z_k1_s1071
                                 - 0.490001113564879 z_k1_s1072
                                 - 0.201000181686583 z_k1_s1073
                                 - 0.21000024615618 z_k1_s1074
                                 - 0.304001219061109 z_k1_s1075
                                 - 0.257000193408185 z_k1_s1076
                                 - 0.538000738468781 z_k1_s1077
                                 - 0.207000668995559 z_k1_s1100
                                 - 0.0880000937299436 z_k1_s1101
                                 - 0.196999578241859 z_k1_s1102
                                 - 0.204999806444642 z_k1_s1103
                                 - 0.243000084907637 z_k1_s1104
                                 - 0.371000514151272 z_k1_s1105
                                 - 0.13600001304624 z_k1_s1106
                                 - 0.163999845155713 z_k1_s1107
                                 - 0.581999461516346 z_k1_s1110
                                 - 0.154999986974348 z_k1_s1111
                                 - 0.553001807490021 z_k1_s1112
                                 - 0.364999106552079 z_k1_s1113
                                 - 0.66600060214816 z_k1_s1114
                                 - 0.425000365575018 z_k1_s1115
                                 - 0.225000211833842 z_k1_s1116
                                 - 0.138999962235048 z_k1_s1117
                                 - 0.204000855754862 z_k1_s1120
                                 - 0.209999750540868 z_k1_s1121
                                 - 0.339001053459991 z_k1_s1122
                                 - 0.115000149848799 z_k1_s1123
                                 - 0.319000838503527 z_k1_s1124
                                 - 0.216000316328207 z_k1_s1125
                                 - 0.0730000326404921 z_k1_s1126
                                 - 0.19500084257452 z_k1_s1127
                                 - 0.163000565597055 z_k1_s1130
                                 - 0.225000302498563 z_k1_s1131
                                 - 0.160000802843962 z_k1_s1132
                                 - 0.133000154063572 z_k1_s1133
                                 - 0.163000565597055 z_k1_s1134
                                 - 0.274999596386423 z_k1_s1135
                                 - 0.215000165320735 z_k1_s1136
                                 - 0.0580000996307605 z_k1_s1137
                                 - 0.186999577689384 z_k1_s1140
                                 - 0.174999487473982 z_k1_s1141
                                 - 0.1160000363486 z_k1_s1142
                                 - 0.122999984507548 z_k1_s1143
                                 - 0.246000282415206 z_k1_s1144
                                 - 0.257000626279417 z_k1_s1145
                                 - 0.257000626279417 z_k1_s1146
                                 - 0.20599945936368 z_k1_s1147
                                 - 0.24600050795523 z_k1_s1150
                                 - 0.241000466167511 z_k1_s1151
                                 - 0.166999847709267 z_k1_s1152
                                 - 0.157999772491373 z_k1_s1153
                                 - 0.10600011189937 z_k1_s1154
                                 - 0.0860000995485486 z_k1_s1155
                                 - 0.317999561697833 z_k1_s1156
                                 - 0.226000340804354 z_k1_s1157
                                 - 1.11499988067387 z_k1_s1160
                                 - 0.838003357660876 z_k1_s1161
                                 - 1.12099978219398 z_k1_s1162
                                 - 0.268000520937006 z_k1_s1163
                                 - 0.548000844914236 z_k1_s1164
                                 - 1.22100241886716 z_k1_s1165
                                 - 0.382000574921222 z_k1_s1166
                                 - 0.924004085118136 z_k1_s1167
                                 - 0.0930001535974601 z_k1_s1170
                                 - 0.234000966536044 z_k1_s1171
                                 - 0.121000149712781 z_k1_s1172
                                 - 0.169000975554049 z_k1_s1173
                                 - 0.431999607064668 z_k1_s1174
                                 - 0.251000631177203 z_k1_s1175
                                 - 0.241000632564588 z_k1_s1176
                                 - 0.124000149296566 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.948993 w_k1_ADM_s000
                                 + 0.895729 w_k1_AXP_s000
                                 + 1.12554 w_k1_DIS_s000
                                 + 1.15244 w_k1_FDX_s000
                                 + 0.806921 w_k1_IBM_s000
                                 + 1.07543 w_k1_INTC_s000
                                 + 0.983336 w_k1_KO_s000
                                 + 1.06339 w_k1_MAT_s000
                                 + 1.09205 w_k1_MSFT_s000
                                 + 0.962209 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.05461 w_k1_ADM_s000
                                 + 0.978521 w_k1_AXP_s000
                                 + 0.844029 w_k1_DIS_s000
                                 + 0.98209 w_k1_FDX_s000
                                 + 0.848858 w_k1_IBM_s000
                                 + 0.838004 w_k1_INTC_s000
                                 + 0.803625 w_k1_KO_s000
                                 + 0.91154 w_k1_MAT_s000
                                 + 0.875281 w_k1_MSFT_s000
                                 + 0.86418 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.713033 w_k1_ADM_s000
                                 + 0.608281 w_k1_AXP_s000
                                 + 0.645278 w_k1_DIS_s000
                                 + 0.770869 w_k1_FDX_s000
                                 + 0.876002 w_k1_IBM_s000
                                 + 0.714306 w_k1_INTC_s000
                                 + 0.853719 w_k1_KO_s000 + 0.9143 w_k1_MAT_s000
                                 + 0.81202 w_k1_MSFT_s000
                                 + 0.780229 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.20017 w_k1_ADM_s000
                                 + 1.16131 w_k1_AXP_s000
                                 + 1.29314 w_k1_DIS_s000
                                 + 1.13958 w_k1_FDX_s000
                                 + 1.21713 w_k1_IBM_s000
                                 + 1.05609 w_k1_INTC_s000
                                 + 1.27258 w_k1_KO_s000 + 1.50183 w_k1_MAT_s000
                                 + 1.13046 w_k1_MSFT_s000
                                 + 1.02324 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0093 w_k1_ADM_s000
                                 + 1.08758 w_k1_AXP_s000
                                 + 0.81834 w_k1_DIS_s000
                                 + 0.989925 w_k1_FDX_s000
                                 + 1.1212 w_k1_IBM_s000
                                 + 0.928601 w_k1_INTC_s000
                                 + 0.962231 w_k1_KO_s000 + 1.5647 w_k1_MAT_s000
                                 + 0.9146 w_k1_MSFT_s000
                                 + 0.932081 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.755771 w_k1_ADM_s000
                                 + 0.747633 w_k1_AXP_s000
                                 + 0.742515 w_k1_DIS_s000
                                 + 0.888115 w_k1_FDX_s000
                                 + 0.643005 w_k1_IBM_s000
                                 + 0.643875 w_k1_INTC_s000
                                 + 0.688753 w_k1_KO_s000
                                 + 0.701524 w_k1_MAT_s000
                                 + 0.795781 w_k1_MSFT_s000
                                 + 0.828993 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.15188 w_k1_ADM_s000
                                 + 1.09034 w_k1_AXP_s000
                                 + 1.22597 w_k1_DIS_s000
                                 + 1.01511 w_k1_FDX_s000 + 1.7195 w_k1_IBM_s000
                                 + 1.16151 w_k1_INTC_s000
                                 + 1.52581 w_k1_KO_s000 + 1.73027 w_k1_MAT_s000
                                 + 1.14412 w_k1_MSFT_s000
                                 + 1.17396 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.03482 w_k1_ADM_s000
                                 + 1.60828 w_k1_AXP_s000
                                 + 1.90795 w_k1_DIS_s000
                                 + 1.58924 w_k1_FDX_s000
                                 + 1.47884 w_k1_IBM_s000
                                 + 1.41518 w_k1_INTC_s000
                                 + 1.73512 w_k1_KO_s000 + 1.43987 w_k1_MAT_s000
                                 + 1.45595 w_k1_MSFT_s000
                                 + 1.35461 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.00307 w_k1_ADM_s001
                                 + 1.32668 w_k1_AXP_s001
                                 + 1.31043 w_k1_DIS_s001
                                 + 1.23832 w_k1_FDX_s001
                                 + 1.14708 w_k1_IBM_s001
                                 + 1.16184 w_k1_INTC_s001 + 1.295 w_k1_KO_s001
                                 + 1.76948 w_k1_MAT_s001
                                 + 1.12703 w_k1_MSFT_s001 + 1.1673 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.946757 w_k1_ADM_s001
                                 + 1.37921 w_k1_AXP_s001
                                 + 1.00944 w_k1_DIS_s001
                                 + 1.00246 w_k1_FDX_s001
                                 + 1.77791 w_k1_IBM_s001
                                 + 1.25832 w_k1_INTC_s001
                                 + 1.41799 w_k1_KO_s001 + 1.84266 w_k1_MAT_s001
                                 + 0.979962 w_k1_MSFT_s001
                                 + 1.10687 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.79515 w_k1_ADM_s001
                                 + 0.866516 w_k1_AXP_s001
                                 + 0.902606 w_k1_DIS_s001
                                 + 0.877905 w_k1_FDX_s001
                                 + 0.689333 w_k1_IBM_s001
                                 + 0.974178 w_k1_INTC_s001
                                 + 0.821741 w_k1_KO_s001
                                 + 0.700633 w_k1_MAT_s001
                                 + 0.983236 w_k1_MSFT_s001
                                 + 0.923251 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.03153 w_k1_ADM_s001
                                 + 1.58997 w_k1_AXP_s001
                                 + 1.77641 w_k1_DIS_s001
                                 + 1.33885 w_k1_FDX_s001 + 1.4746 w_k1_IBM_s001
                                 + 1.28324 w_k1_INTC_s001 + 1.4753 w_k1_KO_s001
                                 + 2.14935 w_k1_MAT_s001
                                 + 1.53983 w_k1_MSFT_s001
                                 + 1.23147 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.17552 w_k1_ADM_s001
                                 + 1.00399 w_k1_AXP_s001
                                 + 0.95967 w_k1_DIS_s001
                                 + 0.998097 w_k1_FDX_s001
                                 + 1.06251 w_k1_IBM_s001
                                 + 0.897526 w_k1_INTC_s001
                                 + 1.01251 w_k1_KO_s001 + 1.41671 w_k1_MAT_s001
                                 + 1.07675 w_k1_MSFT_s001
                                 + 1.04123 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.978309 w_k1_ADM_s001
                                 + 1.04845 w_k1_AXP_s001
                                 + 0.990168 w_k1_DIS_s001
                                 + 1.12372 w_k1_FDX_s001
                                 + 0.751363 w_k1_IBM_s001
                                 + 0.99992 w_k1_INTC_s001
                                 + 0.962025 w_k1_KO_s001
                                 + 0.837983 w_k1_MAT_s001
                                 + 1.07491 w_k1_MSFT_s001
                                 + 0.96709 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.87825 w_k1_ADM_s001
                                 + 0.963574 w_k1_AXP_s001
                                 + 0.675494 w_k1_DIS_s001
                                 + 0.73703 w_k1_FDX_s001
                                 + 1.06211 w_k1_IBM_s001
                                 + 0.724277 w_k1_INTC_s001
                                 + 0.839284 w_k1_KO_s001 + 1.1618 w_k1_MAT_s001
                                 + 0.735199 w_k1_MSFT_s001
                                 + 0.923349 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.740767 w_k1_ADM_s001
                                 + 0.64367 w_k1_AXP_s001
                                 + 0.524593 w_k1_DIS_s001
                                 + 0.722885 w_k1_FDX_s001
                                 + 0.721713 w_k1_IBM_s001
                                 + 0.792523 w_k1_INTC_s001
                                 + 0.642696 w_k1_KO_s001
                                 + 0.759722 w_k1_MAT_s001
                                 + 0.750739 w_k1_MSFT_s001
                                 + 0.774865 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.864811 w_k1_ADM_s002
                                 + 0.928823 w_k1_AXP_s002
                                 + 0.837894 w_k1_DIS_s002
                                 + 0.835252 w_k1_FDX_s002
                                 + 1.49229 w_k1_IBM_s002
                                 + 0.877777 w_k1_INTC_s002
                                 + 1.22763 w_k1_KO_s002 + 1.15793 w_k1_MAT_s002
                                 + 0.883307 w_k1_MSFT_s002
                                 + 0.991012 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.809592 w_k1_ADM_s002
                                 + 0.824014 w_k1_AXP_s002
                                 + 1.17575 w_k1_DIS_s002
                                 + 1.04515 w_k1_FDX_s002
                                 + 0.945166 w_k1_IBM_s002
                                 + 0.915166 w_k1_INTC_s002
                                 + 1.18148 w_k1_KO_s002
                                 + 0.641155 w_k1_MAT_s002
                                 + 1.16254 w_k1_MSFT_s002
                                 + 1.07249 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.96937 w_k1_ADM_s002
                                 + 0.898273 w_k1_AXP_s002
                                 + 0.82441 w_k1_DIS_s002
                                 + 0.951317 w_k1_FDX_s002
                                 + 0.874516 w_k1_IBM_s002
                                 + 0.848123 w_k1_INTC_s002
                                 + 0.954942 w_k1_KO_s002
                                 + 1.10241 w_k1_MAT_s002
                                 + 0.934663 w_k1_MSFT_s002
                                 + 0.947081 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.716128 w_k1_ADM_s002
                                 + 0.68692 w_k1_AXP_s002
                                 + 0.68707 w_k1_DIS_s002
                                 + 0.680435 w_k1_FDX_s002
                                 + 1.15451 w_k1_IBM_s002
                                 + 0.664976 w_k1_INTC_s002
                                 + 0.924125 w_k1_KO_s002
                                 + 0.931758 w_k1_MAT_s002
                                 + 0.643313 w_k1_MSFT_s002
                                 + 0.871612 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.984943 w_k1_ADM_s002
                                 + 1.27672 w_k1_AXP_s002
                                 + 1.28389 w_k1_DIS_s002
                                 + 1.08937 w_k1_FDX_s002
                                 + 1.39827 w_k1_IBM_s002
                                 + 1.21127 w_k1_INTC_s002
                                 + 1.37127 w_k1_KO_s002 + 1.28804 w_k1_MAT_s002
                                 + 1.14552 w_k1_MSFT_s002
                                 + 1.20907 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.12178 w_k1_ADM_s002
                                 + 1.24592 w_k1_AXP_s002
                                 + 1.16251 w_k1_DIS_s002
                                 + 1.25784 w_k1_FDX_s002
                                 + 0.816197 w_k1_IBM_s002
                                 + 1.04288 w_k1_INTC_s002
                                 + 1.15384 w_k1_KO_s002
                                 + 0.995879 w_k1_MAT_s002
                                 + 1.11869 w_k1_MSFT_s002
                                 + 1.21173 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.766195 w_k1_ADM_s002
                                 + 0.585871 w_k1_AXP_s002
                                 + 0.587561 w_k1_DIS_s002
                                 + 0.671748 w_k1_FDX_s002
                                 + 0.705641 w_k1_IBM_s002
                                 + 0.640713 w_k1_INTC_s002
                                 + 0.703241 w_k1_KO_s002
                                 + 0.594789 w_k1_MAT_s002
                                 + 0.681667 w_k1_MSFT_s002
                                 + 0.870503 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.30114 w_k1_ADM_s002
                                 + 1.32499 w_k1_AXP_s002
                                 + 1.71266 w_k1_DIS_s002
                                 + 1.57666 w_k1_FDX_s002
                                 + 1.04656 w_k1_IBM_s002
                                 + 1.35566 w_k1_INTC_s002
                                 + 1.24308 w_k1_KO_s002 + 1.02023 w_k1_MAT_s002
                                 + 1.33014 w_k1_MSFT_s002
                                 + 1.15638 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.27181 w_k1_ADM_s003
                                 + 1.48864 w_k1_AXP_s003
                                 + 1.04331 w_k1_DIS_s003 + 1.1815 w_k1_FDX_s003
                                 + 1.32001 w_k1_IBM_s003
                                 + 1.15032 w_k1_INTC_s003
                                 + 1.06597 w_k1_KO_s003 + 1.13411 w_k1_MAT_s003
                                 + 1.03206 w_k1_MSFT_s003
                                 + 1.09377 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.899341 w_k1_ADM_s003
                                 + 1.19459 w_k1_AXP_s003
                                 + 0.896834 w_k1_DIS_s003
                                 + 0.996832 w_k1_FDX_s003
                                 + 1.26441 w_k1_IBM_s003
                                 + 1.0126 w_k1_INTC_s003 + 1.00553 w_k1_KO_s003
                                 + 1.11863 w_k1_MAT_s003
                                 + 0.891141 w_k1_MSFT_s003
                                 + 0.990245 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.711961 w_k1_ADM_s003
                                 + 0.62694 w_k1_AXP_s003
                                 + 0.740817 w_k1_DIS_s003
                                 + 0.812031 w_k1_FDX_s003
                                 + 0.603506 w_k1_IBM_s003
                                 + 0.894004 w_k1_INTC_s003
                                 + 0.617059 w_k1_KO_s003
                                 + 0.473289 w_k1_MAT_s003
                                 + 0.845346 w_k1_MSFT_s003
                                 + 0.779439 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.46711 w_k1_ADM_s003
                                 + 1.94757 w_k1_AXP_s003
                                 + 1.78807 w_k1_DIS_s003
                                 + 1.42145 w_k1_FDX_s003
                                 + 1.63458 w_k1_IBM_s003
                                 + 1.4182 w_k1_INTC_s003 + 1.38413 w_k1_KO_s003
                                 + 1.43337 w_k1_MAT_s003
                                 + 1.25029 w_k1_MSFT_s003
                                 + 1.21906 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.677562 w_k1_ADM_s003
                                 + 0.851131 w_k1_AXP_s003
                                 + 0.611855 w_k1_DIS_s003
                                 + 0.723523 w_k1_FDX_s003
                                 + 0.964399 w_k1_IBM_s003
                                 + 0.906251 w_k1_INTC_s003
                                 + 0.810652 w_k1_KO_s003
                                 + 0.746991 w_k1_MAT_s003
                                 + 0.778395 w_k1_MSFT_s003
                                 + 0.848061 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.803117 w_k1_ADM_s003
                                 + 1.01987 w_k1_AXP_s003
                                 + 1.22996 w_k1_DIS_s003
                                 + 1.14963 w_k1_FDX_s003
                                 + 1.14903 w_k1_IBM_s003
                                 + 1.36412 w_k1_INTC_s003
                                 + 1.11773 w_k1_KO_s003
                                 + 0.960777 w_k1_MAT_s003
                                 + 1.03866 w_k1_MSFT_s003
                                 + 1.12988 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.777709 w_k1_ADM_s003
                                 + 0.782796 w_k1_AXP_s003
                                 + 0.9969 w_k1_DIS_s003
                                 + 0.975824 w_k1_FDX_s003
                                 + 0.785135 w_k1_IBM_s003
                                 + 1.23684 w_k1_INTC_s003
                                 + 0.891031 w_k1_KO_s003
                                 + 0.597411 w_k1_MAT_s003
                                 + 1.11177 w_k1_MSFT_s003
                                 + 0.865079 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.16556 w_k1_ADM_s003
                                 + 1.19215 w_k1_AXP_s003
                                 + 1.50882 w_k1_DIS_s003
                                 + 1.36718 w_k1_FDX_s003
                                 + 0.95728 w_k1_IBM_s003
                                 + 1.3027 w_k1_INTC_s003 + 1.14535 w_k1_KO_s003
                                 + 0.804982 w_k1_MAT_s003
                                 + 1.24471 w_k1_MSFT_s003
                                 + 1.05382 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.813172 w_k1_ADM_s004
                                 + 0.807858 w_k1_AXP_s004
                                 + 0.708388 w_k1_DIS_s004
                                 + 0.877983 w_k1_FDX_s004
                                 + 0.928398 w_k1_IBM_s004
                                 + 0.882725 w_k1_INTC_s004
                                 + 0.925984 w_k1_KO_s004
                                 + 1.09017 w_k1_MAT_s004
                                 + 0.893317 w_k1_MSFT_s004
                                 + 0.882538 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.972321 w_k1_ADM_s004
                                 + 0.948487 w_k1_AXP_s004
                                 + 1.31848 w_k1_DIS_s004
                                 + 1.20793 w_k1_FDX_s004
                                 + 0.962023 w_k1_IBM_s004
                                 + 1.29476 w_k1_INTC_s004
                                 + 1.08566 w_k1_KO_s004
                                 + 0.816064 w_k1_MAT_s004
                                 + 1.16396 w_k1_MSFT_s004
                                 + 1.01295 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.8445 w_k1_ADM_s004
                                 + 0.799539 w_k1_AXP_s004
                                 + 0.733299 w_k1_DIS_s004
                                 + 0.882357 w_k1_FDX_s004
                                 + 0.659512 w_k1_IBM_s004
                                 + 0.934227 w_k1_INTC_s004
                                 + 0.687713 w_k1_KO_s004
                                 + 0.668842 w_k1_MAT_s004
                                 + 0.91736 w_k1_MSFT_s004
                                 + 0.795905 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.00952 w_k1_ADM_s004
                                 + 0.775001 w_k1_AXP_s004
                                 + 0.666454 w_k1_DIS_s004
                                 + 0.798949 w_k1_FDX_s004
                                 + 0.89315 w_k1_IBM_s004
                                 + 0.857974 w_k1_INTC_s004
                                 + 0.878944 w_k1_KO_s004
                                 + 0.658854 w_k1_MAT_s004
                                 + 0.832766 w_k1_MSFT_s004
                                 + 0.819562 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.13356 w_k1_ADM_s004
                                 + 1.45404 w_k1_AXP_s004 + 1.1012 w_k1_DIS_s004
                                 + 1.21112 w_k1_FDX_s004
                                 + 1.10887 w_k1_IBM_s004
                                 + 1.1887 w_k1_INTC_s004 + 1.00658 w_k1_KO_s004
                                 + 1.27196 w_k1_MAT_s004
                                 + 1.10782 w_k1_MSFT_s004
                                 + 1.09175 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.20179 w_k1_ADM_s004
                                 + 1.35829 w_k1_AXP_s004
                                 + 1.40635 w_k1_DIS_s004
                                 + 1.11117 w_k1_FDX_s004
                                 + 1.62962 w_k1_IBM_s004
                                 + 1.24479 w_k1_INTC_s004 + 1.4175 w_k1_KO_s004
                                 + 1.41226 w_k1_MAT_s004
                                 + 1.06729 w_k1_MSFT_s004
                                 + 1.06782 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.884022 w_k1_ADM_s004
                                 + 0.785898 w_k1_AXP_s004
                                 + 0.849929 w_k1_DIS_s004
                                 + 0.822052 w_k1_FDX_s004
                                 + 1.23387 w_k1_IBM_s004
                                 + 0.871422 w_k1_INTC_s004
                                 + 1.12501 w_k1_KO_s004 + 1.03417 w_k1_MAT_s004
                                 + 0.926481 w_k1_MSFT_s004
                                 + 0.959613 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.856442 w_k1_ADM_s004
                                 + 0.975485 w_k1_AXP_s004
                                 + 0.821489 w_k1_DIS_s004
                                 + 0.86078 w_k1_FDX_s004
                                 + 1.57804 w_k1_IBM_s004
                                 + 0.847545 w_k1_INTC_s004
                                 + 0.962009 w_k1_KO_s004 + 1.6048 w_k1_MAT_s004
                                 + 0.76495 w_k1_MSFT_s004
                                 + 0.964834 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.11178 w_k1_ADM_s005
                                 + 0.726119 w_k1_AXP_s005
                                 + 0.829217 w_k1_DIS_s005
                                 + 0.908739 w_k1_FDX_s005
                                 + 0.613531 w_k1_IBM_s005
                                 + 1.01463 w_k1_INTC_s005
                                 + 0.862886 w_k1_KO_s005
                                 + 0.738156 w_k1_MAT_s005
                                 + 0.945561 w_k1_MSFT_s005
                                 + 0.960975 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.12785 w_k1_ADM_s005
                                 + 1.26255 w_k1_AXP_s005
                                 + 1.29743 w_k1_DIS_s005
                                 + 1.23168 w_k1_FDX_s005
                                 + 0.930097 w_k1_IBM_s005
                                 + 1.35237 w_k1_INTC_s005
                                 + 1.18647 w_k1_KO_s005 + 1.01766 w_k1_MAT_s005
                                 + 1.13586 w_k1_MSFT_s005
                                 + 0.953835 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.740289 w_k1_ADM_s005
                                 + 0.697385 w_k1_AXP_s005
                                 + 0.658578 w_k1_DIS_s005
                                 + 0.757178 w_k1_FDX_s005
                                 + 0.767032 w_k1_IBM_s005
                                 + 0.81902 w_k1_INTC_s005
                                 + 0.837232 w_k1_KO_s005
                                 + 0.953507 w_k1_MAT_s005
                                 + 0.748919 w_k1_MSFT_s005
                                 + 0.835814 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.27862 w_k1_ADM_s005
                                 + 1.01915 w_k1_AXP_s005
                                 + 0.994121 w_k1_DIS_s005
                                 + 0.998722 w_k1_FDX_s005
                                 + 0.804562 w_k1_IBM_s005
                                 + 1.09395 w_k1_INTC_s005
                                 + 0.915022 w_k1_KO_s005
                                 + 0.87801 w_k1_MAT_s005
                                 + 0.972023 w_k1_MSFT_s005
                                 + 0.981828 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.834939 w_k1_ADM_s005
                                 + 0.706596 w_k1_AXP_s005
                                 + 0.743656 w_k1_DIS_s005
                                 + 0.794888 w_k1_FDX_s005
                                 + 0.745647 w_k1_IBM_s005
                                 + 0.797354 w_k1_INTC_s005
                                 + 0.751988 w_k1_KO_s005
                                 + 0.757528 w_k1_MAT_s005
                                 + 0.865205 w_k1_MSFT_s005
                                 + 0.80754 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.41521 w_k1_ADM_s005
                                 + 1.54626 w_k1_AXP_s005
                                 + 1.55363 w_k1_DIS_s005 + 1.2925 w_k1_FDX_s005
                                 + 1.34667 w_k1_IBM_s005
                                 + 1.31832 w_k1_INTC_s005
                                 + 1.41874 w_k1_KO_s005 + 1.34042 w_k1_MAT_s005
                                 + 1.25363 w_k1_MSFT_s005
                                 + 1.20178 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.989429 w_k1_ADM_s005
                                 + 1.00126 w_k1_AXP_s005 + 1.0098 w_k1_DIS_s005
                                 + 0.928663 w_k1_FDX_s005
                                 + 1.13662 w_k1_IBM_s005
                                 + 0.954407 w_k1_INTC_s005
                                 + 1.07885 w_k1_KO_s005 + 1.26727 w_k1_MAT_s005
                                 + 1.00443 w_k1_MSFT_s005
                                 + 1.02024 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.02385 w_k1_ADM_s005
                                 + 1.41793 w_k1_AXP_s005
                                 + 1.16575 w_k1_DIS_s005 + 1.0009 w_k1_FDX_s005
                                 + 1.74249 w_k1_IBM_s005
                                 + 1.25126 w_k1_INTC_s005
                                 + 1.35201 w_k1_KO_s005 + 2.16406 w_k1_MAT_s005
                                 + 0.998075 w_k1_MSFT_s005
                                 + 1.10159 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.69082 w_k1_ADM_s006
                                 + 0.652725 w_k1_AXP_s006
                                 + 0.567303 w_k1_DIS_s006
                                 + 0.696326 w_k1_FDX_s006
                                 + 0.675922 w_k1_IBM_s006
                                 + 0.729265 w_k1_INTC_s006
                                 + 0.608378 w_k1_KO_s006
                                 + 0.651389 w_k1_MAT_s006
                                 + 0.720214 w_k1_MSFT_s006
                                 + 0.741374 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.03318 w_k1_ADM_s006
                                 + 1.5639 w_k1_AXP_s006 + 1.27346 w_k1_DIS_s006
                                 + 1.25949 w_k1_FDX_s006
                                 + 1.33587 w_k1_IBM_s006
                                 + 1.35727 w_k1_INTC_s006
                                 + 1.24144 w_k1_KO_s006 + 2.06668 w_k1_MAT_s006
                                 + 1.18401 w_k1_MSFT_s006
                                 + 1.26626 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.829746 w_k1_ADM_s006
                                 + 0.787019 w_k1_AXP_s006
                                 + 0.73239 w_k1_DIS_s006
                                 + 0.85533 w_k1_FDX_s006
                                 + 0.780149 w_k1_IBM_s006
                                 + 0.793823 w_k1_INTC_s006
                                 + 0.758836 w_k1_KO_s006
                                 + 0.532339 w_k1_MAT_s006
                                 + 0.698627 w_k1_MSFT_s006
                                 + 0.868576 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.0009 w_k1_ADM_s006
                                 + 1.95188 w_k1_AXP_s006
                                 + 1.89151 w_k1_DIS_s006
                                 + 1.50379 w_k1_FDX_s006
                                 + 1.11841 w_k1_IBM_s006
                                 + 1.35142 w_k1_INTC_s006
                                 + 1.30564 w_k1_KO_s006 + 2.00715 w_k1_MAT_s006
                                 + 1.39574 w_k1_MSFT_s006 + 1.4655 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.948074 w_k1_ADM_s006
                                 + 0.859744 w_k1_AXP_s006
                                 + 0.952959 w_k1_DIS_s006
                                 + 1.04797 w_k1_FDX_s006
                                 + 0.748383 w_k1_IBM_s006
                                 + 0.89849 w_k1_INTC_s006
                                 + 0.859603 w_k1_KO_s006
                                 + 0.91273 w_k1_MAT_s006
                                 + 1.05302 w_k1_MSFT_s006
                                 + 1.03867 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.888759 w_k1_ADM_s006
                                 + 1.10155 w_k1_AXP_s006
                                 + 0.771423 w_k1_DIS_s006
                                 + 0.833074 w_k1_FDX_s006
                                 + 1.30995 w_k1_IBM_s006
                                 + 1.04971 w_k1_INTC_s006
                                 + 1.04052 w_k1_KO_s006 + 1.69043 w_k1_MAT_s006
                                 + 0.895448 w_k1_MSFT_s006
                                 + 1.01451 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.855536 w_k1_ADM_s006
                                 + 1.13083 w_k1_AXP_s006
                                 + 1.10799 w_k1_DIS_s006
                                 + 1.03613 w_k1_FDX_s006
                                 + 1.07763 w_k1_IBM_s006
                                 + 1.02462 w_k1_INTC_s006
                                 + 1.03375 w_k1_KO_s006 + 1.41114 w_k1_MAT_s006
                                 + 0.993633 w_k1_MSFT_s006
                                 + 1.08458 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.867394 w_k1_ADM_s006
                                 + 0.988456 w_k1_AXP_s006
                                 + 1.25092 w_k1_DIS_s006
                                 + 1.10226 w_k1_FDX_s006
                                 + 0.782505 w_k1_IBM_s006
                                 + 1.03506 w_k1_INTC_s006
                                 + 1.02557 w_k1_KO_s006
                                 + 0.927635 w_k1_MAT_s006
                                 + 1.12243 w_k1_MSFT_s006
                                 + 1.04941 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.07924 w_k1_ADM_s007
                                 + 1.50844 w_k1_AXP_s007
                                 + 1.45992 w_k1_DIS_s007
                                 + 1.20671 w_k1_FDX_s007
                                 + 1.60269 w_k1_IBM_s007
                                 + 1.36744 w_k1_INTC_s007
                                 + 1.61694 w_k1_KO_s007 + 1.28903 w_k1_MAT_s007
                                 + 1.21022 w_k1_MSFT_s007
                                 + 1.20582 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.875443 w_k1_ADM_s007
                                 + 0.82532 w_k1_AXP_s007
                                 + 0.810398 w_k1_DIS_s007
                                 + 0.926688 w_k1_FDX_s007
                                 + 0.742096 w_k1_IBM_s007
                                 + 1.07643 w_k1_INTC_s007
                                 + 0.838887 w_k1_KO_s007
                                 + 0.637659 w_k1_MAT_s007
                                 + 0.962488 w_k1_MSFT_s007
                                 + 0.922511 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.712295 w_k1_ADM_s007
                                 + 0.760045 w_k1_AXP_s007
                                 + 0.637508 w_k1_DIS_s007
                                 + 0.715976 w_k1_FDX_s007
                                 + 0.865045 w_k1_IBM_s007
                                 + 1.02634 w_k1_INTC_s007
                                 + 0.823161 w_k1_KO_s007
                                 + 0.594194 w_k1_MAT_s007
                                 + 0.776455 w_k1_MSFT_s007
                                 + 0.898377 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.932313 w_k1_ADM_s007
                                 + 0.949007 w_k1_AXP_s007
                                 + 1.26546 w_k1_DIS_s007
                                 + 1.12217 w_k1_FDX_s007
                                 + 0.862322 w_k1_IBM_s007
                                 + 1.45171 w_k1_INTC_s007
                                 + 1.16422 w_k1_KO_s007
                                 + 0.728218 w_k1_MAT_s007
                                 + 1.19038 w_k1_MSFT_s007 + 1.0419 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.817838 w_k1_ADM_s007
                                 + 0.99418 w_k1_AXP_s007
                                 + 1.08415 w_k1_DIS_s007
                                 + 0.840692 w_k1_FDX_s007
                                 + 1.4697 w_k1_IBM_s007
                                 + 1.13119 w_k1_INTC_s007
                                 + 1.23466 w_k1_KO_s007 + 1.21293 w_k1_MAT_s007
                                 + 0.96416 w_k1_MSFT_s007
                                 + 1.14697 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.10278 w_k1_ADM_s007
                                 + 1.2833 w_k1_AXP_s007 + 1.49139 w_k1_DIS_s007
                                 + 1.34794 w_k1_FDX_s007
                                 + 1.11517 w_k1_IBM_s007
                                 + 1.29692 w_k1_INTC_s007
                                 + 1.20256 w_k1_KO_s007
                                 + 0.917239 w_k1_MAT_s007
                                 + 1.19303 w_k1_MSFT_s007
                                 + 1.23152 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.859276 w_k1_ADM_s007
                                 + 0.809168 w_k1_AXP_s007
                                 + 0.813966 w_k1_DIS_s007
                                 + 0.838918 w_k1_FDX_s007
                                 + 1.09968 w_k1_IBM_s007
                                 + 0.963942 w_k1_INTC_s007
                                 + 0.99589 w_k1_KO_s007
                                 + 0.768536 w_k1_MAT_s007
                                 + 0.764662 w_k1_MSFT_s007
                                 + 1.01892 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.16222 w_k1_ADM_s007
                                 + 1.17276 w_k1_AXP_s007
                                 + 1.06592 w_k1_DIS_s007
                                 + 1.04967 w_k1_FDX_s007
                                 + 1.08289 w_k1_IBM_s007
                                 + 1.20788 w_k1_INTC_s007
                                 + 1.08824 w_k1_KO_s007 + 1.02489 w_k1_MAT_s007
                                 + 0.947123 w_k1_MSFT_s007
                                 + 1.0712 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.927641 w_k1_ADM_s010
                                 + 0.765021 w_k1_AXP_s010
                                 + 0.768193 w_k1_DIS_s010
                                 + 0.934123 w_k1_FDX_s010
                                 + 0.928367 w_k1_IBM_s010
                                 + 1.13457 w_k1_INTC_s010
                                 + 0.846838 w_k1_KO_s010
                                 + 0.943869 w_k1_MAT_s010
                                 + 0.929147 w_k1_MSFT_s010
                                 + 0.994664 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.45553 w_k1_ADM_s010
                                 + 1.13198 w_k1_AXP_s010
                                 + 1.34972 w_k1_DIS_s010
                                 + 1.23577 w_k1_FDX_s010
                                 + 0.923313 w_k1_IBM_s010
                                 + 1.26706 w_k1_INTC_s010 + 1.057 w_k1_KO_s010
                                 + 0.947054 w_k1_MAT_s010
                                 + 1.2399 w_k1_MSFT_s010 + 1.11677 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.874032 w_k1_ADM_s010
                                 + 0.600799 w_k1_AXP_s010
                                 + 0.696946 w_k1_DIS_s010
                                 + 0.691411 w_k1_FDX_s010
                                 + 0.648568 w_k1_IBM_s010
                                 + 0.783046 w_k1_INTC_s010
                                 + 0.727561 w_k1_KO_s010
                                 + 0.682944 w_k1_MAT_s010
                                 + 0.734676 w_k1_MSFT_s010
                                 + 0.840714 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.979833 w_k1_ADM_s010
                                 + 0.840754 w_k1_AXP_s010
                                 + 0.895795 w_k1_DIS_s010
                                 + 0.841542 w_k1_FDX_s010
                                 + 1.14345 w_k1_IBM_s010
                                 + 0.940892 w_k1_INTC_s010
                                 + 0.970885 w_k1_KO_s010
                                 + 0.985263 w_k1_MAT_s010
                                 + 0.811243 w_k1_MSFT_s010
                                 + 1.00689 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.13705 w_k1_ADM_s010
                                 + 1.16972 w_k1_AXP_s010
                                 + 1.10253 w_k1_DIS_s010
                                 + 1.07405 w_k1_FDX_s010
                                 + 1.02185 w_k1_IBM_s010
                                 + 0.984982 w_k1_INTC_s010
                                 + 0.945222 w_k1_KO_s010
                                 + 0.94549 w_k1_MAT_s010
                                 + 0.873871 w_k1_MSFT_s010
                                 + 1.06969 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.834362 w_k1_ADM_s010
                                 + 0.610098 w_k1_AXP_s010
                                 + 0.772984 w_k1_DIS_s010
                                 + 0.722514 w_k1_FDX_s010
                                 + 0.852461 w_k1_IBM_s010
                                 + 0.890417 w_k1_INTC_s010
                                 + 0.813645 w_k1_KO_s010
                                 + 0.538248 w_k1_MAT_s010
                                 + 0.802755 w_k1_MSFT_s010
                                 + 0.869329 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.50937 w_k1_ADM_s010
                                 + 1.33047 w_k1_AXP_s010
                                 + 1.55684 w_k1_DIS_s010
                                 + 1.22691 w_k1_FDX_s010
                                 + 1.37684 w_k1_IBM_s010
                                 + 1.19214 w_k1_INTC_s010
                                 + 1.18374 w_k1_KO_s010 + 1.6415 w_k1_MAT_s010
                                 + 1.13429 w_k1_MSFT_s010
                                 + 1.24037 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.04333 w_k1_ADM_s010
                                 + 0.795719 w_k1_AXP_s010
                                 + 0.890731 w_k1_DIS_s010
                                 + 1.00202 w_k1_FDX_s010
                                 + 0.664586 w_k1_IBM_s010
                                 + 0.825076 w_k1_INTC_s010
                                 + 0.699765 w_k1_KO_s010
                                 + 0.516236 w_k1_MAT_s010
                                 + 0.906146 w_k1_MSFT_s010
                                 + 0.955453 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.895934 w_k1_ADM_s011
                                 + 0.831679 w_k1_AXP_s011
                                 + 0.925018 w_k1_DIS_s011
                                 + 0.889964 w_k1_FDX_s011
                                 + 1.05856 w_k1_IBM_s011
                                 + 0.991206 w_k1_INTC_s011
                                 + 0.931164 w_k1_KO_s011
                                 + 0.976907 w_k1_MAT_s011
                                 + 1.0401 w_k1_MSFT_s011
                                 + 0.919081 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.05999 w_k1_ADM_s011
                                 + 1.06697 w_k1_AXP_s011
                                 + 1.03914 w_k1_DIS_s011
                                 + 1.09958 w_k1_FDX_s011
                                 + 0.704862 w_k1_IBM_s011
                                 + 0.985754 w_k1_INTC_s011
                                 + 0.861986 w_k1_KO_s011
                                 + 0.684776 w_k1_MAT_s011
                                 + 1.11006 w_k1_MSFT_s011
                                 + 1.02866 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.927485 w_k1_ADM_s011
                                 + 0.970847 w_k1_AXP_s011
                                 + 1.24006 w_k1_DIS_s011
                                 + 1.10994 w_k1_FDX_s011
                                 + 1.12784 w_k1_IBM_s011
                                 + 1.20634 w_k1_INTC_s011
                                 + 1.18038 w_k1_KO_s011 + 1.40487 w_k1_MAT_s011
                                 + 1.26866 w_k1_MSFT_s011
                                 + 1.19663 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.670883 w_k1_ADM_s011
                                 + 0.59863 w_k1_AXP_s011
                                 + 0.775466 w_k1_DIS_s011
                                 + 0.830936 w_k1_FDX_s011
                                 + 0.751897 w_k1_IBM_s011
                                 + 1.05019 w_k1_INTC_s011
                                 + 0.871738 w_k1_KO_s011
                                 + 0.780332 w_k1_MAT_s011
                                 + 0.979601 w_k1_MSFT_s011
                                 + 0.893171 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.980864 w_k1_ADM_s011
                                 + 0.889247 w_k1_AXP_s011
                                 + 0.789455 w_k1_DIS_s011
                                 + 0.880674 w_k1_FDX_s011
                                 + 1.26607 w_k1_IBM_s011
                                 + 0.974433 w_k1_INTC_s011
                                 + 1.05477 w_k1_KO_s011 + 1.24625 w_k1_MAT_s011
                                 + 0.854888 w_k1_MSFT_s011
                                 + 1.08383 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.26653 w_k1_ADM_s011
                                 + 1.53455 w_k1_AXP_s011
                                 + 1.68902 w_k1_DIS_s011
                                 + 1.35997 w_k1_FDX_s011
                                 + 0.958969 w_k1_IBM_s011
                                 + 1.21395 w_k1_INTC_s011
                                 + 1.32294 w_k1_KO_s011 + 1.15817 w_k1_MAT_s011
                                 + 1.38193 w_k1_MSFT_s011
                                 + 1.30153 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.755752 w_k1_ADM_s011
                                 + 0.697692 w_k1_AXP_s011
                                 + 0.6291 w_k1_DIS_s011
                                 + 0.650928 w_k1_FDX_s011
                                 + 0.892379 w_k1_IBM_s011
                                 + 0.845153 w_k1_INTC_s011
                                 + 0.902522 w_k1_KO_s011
                                 + 0.786576 w_k1_MAT_s011
                                 + 0.799904 w_k1_MSFT_s011
                                 + 0.862662 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.15793 w_k1_ADM_s011
                                 + 1.60888 w_k1_AXP_s011 + 1.7869 w_k1_DIS_s011
                                 + 1.34798 w_k1_FDX_s011
                                 + 1.56483 w_k1_IBM_s011
                                 + 1.34404 w_k1_INTC_s011
                                 + 1.64269 w_k1_KO_s011 + 2.14358 w_k1_MAT_s011
                                 + 1.65533 w_k1_MSFT_s011 + 1.3466 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.99869 w_k1_ADM_s012
                                 + 1.05062 w_k1_AXP_s012
                                 + 0.939831 w_k1_DIS_s012
                                 + 1.10286 w_k1_FDX_s012
                                 + 0.80119 w_k1_IBM_s012
                                 + 1.00389 w_k1_INTC_s012
                                 + 0.865354 w_k1_KO_s012
                                 + 0.900969 w_k1_MAT_s012
                                 + 0.98467 w_k1_MSFT_s012
                                 + 0.908011 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.684136 w_k1_ADM_s012
                                 + 0.654818 w_k1_AXP_s012
                                 + 0.612894 w_k1_DIS_s012
                                 + 0.81163 w_k1_FDX_s012
                                 + 0.613237 w_k1_IBM_s012
                                 + 0.790932 w_k1_INTC_s012
                                 + 0.660437 w_k1_KO_s012
                                 + 0.619477 w_k1_MAT_s012
                                 + 0.821012 w_k1_MSFT_s012
                                 + 0.759253 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.44752 w_k1_ADM_s012
                                 + 1.38724 w_k1_AXP_s012
                                 + 1.40191 w_k1_DIS_s012
                                 + 1.37275 w_k1_FDX_s012
                                 + 0.994449 w_k1_IBM_s012
                                 + 1.3579 w_k1_INTC_s012 + 1.32719 w_k1_KO_s012
                                 + 1.12237 w_k1_MAT_s012
                                 + 1.29158 w_k1_MSFT_s012
                                 + 1.07382 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.38755 w_k1_ADM_s012
                                 + 1.25907 w_k1_AXP_s012
                                 + 1.35906 w_k1_DIS_s012
                                 + 1.19747 w_k1_FDX_s012
                                 + 1.47235 w_k1_IBM_s012
                                 + 1.11518 w_k1_INTC_s012
                                 + 1.49249 w_k1_KO_s012 + 1.71437 w_k1_MAT_s012
                                 + 1.21488 w_k1_MSFT_s012 + 1.035 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.23892 w_k1_ADM_s012
                                 + 1.09541 w_k1_AXP_s012
                                 + 1.06851 w_k1_DIS_s012
                                 + 1.07757 w_k1_FDX_s012
                                 + 0.990691 w_k1_IBM_s012
                                 + 0.957547 w_k1_INTC_s012
                                 + 1.04984 w_k1_KO_s012 + 1.32304 w_k1_MAT_s012
                                 + 1.06203 w_k1_MSFT_s012
                                 + 1.08454 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.849038 w_k1_ADM_s012
                                 + 0.721588 w_k1_AXP_s012
                                 + 0.69061 w_k1_DIS_s012
                                 + 0.860352 w_k1_FDX_s012
                                 + 0.684047 w_k1_IBM_s012
                                 + 0.964677 w_k1_INTC_s012
                                 + 0.698719 w_k1_KO_s012
                                 + 0.731109 w_k1_MAT_s012
                                 + 0.853271 w_k1_MSFT_s012
                                 + 0.873085 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.00244 w_k1_ADM_s012
                                 + 0.832132 w_k1_AXP_s012
                                 + 0.858142 w_k1_DIS_s012
                                 + 0.852532 w_k1_FDX_s012
                                 + 1.39339 w_k1_IBM_s012
                                 + 1.0384 w_k1_INTC_s012 + 1.13683 w_k1_KO_s012
                                 + 1.34634 w_k1_MAT_s012
                                 + 0.881429 w_k1_MSFT_s012
                                 + 0.9847 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.972241 w_k1_ADM_s012
                                 + 0.829992 w_k1_AXP_s012
                                 + 0.796504 w_k1_DIS_s012
                                 + 0.838807 w_k1_FDX_s012
                                 + 0.97407 w_k1_IBM_s012
                                 + 0.859881 w_k1_INTC_s012
                                 + 0.933507 w_k1_KO_s012
                                 + 0.97511 w_k1_MAT_s012
                                 + 0.742932 w_k1_MSFT_s012
                                 + 0.896025 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.04533 w_k1_ADM_s013
                                 + 0.988399 w_k1_AXP_s013
                                 + 1.06122 w_k1_DIS_s013
                                 + 1.01198 w_k1_FDX_s013
                                 + 0.951684 w_k1_IBM_s013
                                 + 1.05682 w_k1_INTC_s013
                                 + 1.05925 w_k1_KO_s013 + 1.07883 w_k1_MAT_s013
                                 + 1.10065 w_k1_MSFT_s013
                                 + 0.993037 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.35149 w_k1_ADM_s013
                                 + 1.05297 w_k1_AXP_s013
                                 + 0.819179 w_k1_DIS_s013
                                 + 0.959965 w_k1_FDX_s013
                                 + 0.795499 w_k1_IBM_s013
                                 + 0.998991 w_k1_INTC_s013
                                 + 0.915845 w_k1_KO_s013 + 1.0127 w_k1_MAT_s013
                                 + 0.96181 w_k1_MSFT_s013
                                 + 0.951042 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.943852 w_k1_ADM_s013
                                 + 0.792646 w_k1_AXP_s013
                                 + 0.751777 w_k1_DIS_s013
                                 + 0.734046 w_k1_FDX_s013
                                 + 0.908411 w_k1_IBM_s013
                                 + 1.00203 w_k1_INTC_s013
                                 + 1.04448 w_k1_KO_s013 + 1.18181 w_k1_MAT_s013
                                 + 0.872983 w_k1_MSFT_s013
                                 + 0.948859 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.942426 w_k1_ADM_s013
                                 + 0.742567 w_k1_AXP_s013
                                 + 0.688055 w_k1_DIS_s013
                                 + 0.795228 w_k1_FDX_s013
                                 + 0.619456 w_k1_IBM_s013
                                 + 0.767962 w_k1_INTC_s013
                                 + 0.716802 w_k1_KO_s013
                                 + 0.530426 w_k1_MAT_s013
                                 + 0.864678 w_k1_MSFT_s013
                                 + 0.814738 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.54234 w_k1_ADM_s013
                                 + 1.3729 w_k1_AXP_s013 + 1.45327 w_k1_DIS_s013
                                 + 1.33323 w_k1_FDX_s013
                                 + 1.09096 w_k1_IBM_s013
                                 + 1.19123 w_k1_INTC_s013
                                 + 1.31699 w_k1_KO_s013 + 1.23248 w_k1_MAT_s013
                                 + 1.1615 w_k1_MSFT_s013 + 1.24731 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.00234 w_k1_ADM_s013
                                 + 0.673186 w_k1_AXP_s013
                                 + 0.76575 w_k1_DIS_s013
                                 + 0.807423 w_k1_FDX_s013
                                 + 1.02427 w_k1_IBM_s013
                                 + 0.893585 w_k1_INTC_s013
                                 + 0.946093 w_k1_KO_s013
                                 + 1.06482 w_k1_MAT_s013
                                 + 0.94072 w_k1_MSFT_s013
                                 + 0.928741 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.35185 w_k1_ADM_s013
                                 + 1.04083 w_k1_AXP_s013
                                 + 1.16326 w_k1_DIS_s013
                                 + 1.04246 w_k1_FDX_s013
                                 + 1.37589 w_k1_IBM_s013
                                 + 1.10045 w_k1_INTC_s013
                                 + 1.46478 w_k1_KO_s013 + 1.39252 w_k1_MAT_s013
                                 + 1.07758 w_k1_MSFT_s013
                                 + 1.30374 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.19883 w_k1_ADM_s013
                                 + 1.02185 w_k1_AXP_s013
                                 + 1.03284 w_k1_DIS_s013
                                 + 1.13178 w_k1_FDX_s013 + 0.658 w_k1_IBM_s013
                                 + 1.10899 w_k1_INTC_s013
                                 + 0.947016 w_k1_KO_s013
                                 + 0.724694 w_k1_MAT_s013
                                 + 1.17083 w_k1_MSFT_s013
                                 + 0.956752 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.18644 w_k1_ADM_s014
                                 + 1.17558 w_k1_AXP_s014
                                 + 1.24424 w_k1_DIS_s014
                                 + 1.28975 w_k1_FDX_s014
                                 + 1.65706 w_k1_IBM_s014
                                 + 1.41401 w_k1_INTC_s014 + 1.3674 w_k1_KO_s014
                                 + 1.76466 w_k1_MAT_s014
                                 + 1.18588 w_k1_MSFT_s014
                                 + 1.16616 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.876472 w_k1_ADM_s014
                                 + 0.635532 w_k1_AXP_s014
                                 + 0.645354 w_k1_DIS_s014
                                 + 0.801525 w_k1_FDX_s014
                                 + 0.742461 w_k1_IBM_s014
                                 + 0.845313 w_k1_INTC_s014
                                 + 0.688791 w_k1_KO_s014
                                 + 0.707315 w_k1_MAT_s014
                                 + 0.690456 w_k1_MSFT_s014
                                 + 0.867641 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.39826 w_k1_ADM_s014
                                 + 1.24006 w_k1_AXP_s014
                                 + 1.47181 w_k1_DIS_s014 + 1.4241 w_k1_FDX_s014
                                 + 1.24331 w_k1_IBM_s014
                                 + 1.61842 w_k1_INTC_s014
                                 + 1.23616 w_k1_KO_s014 + 1.15262 w_k1_MAT_s014
                                 + 1.33273 w_k1_MSFT_s014 + 1.1913 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20971 w_k1_ADM_s014
                                 + 0.982246 w_k1_AXP_s014
                                 + 0.758993 w_k1_DIS_s014
                                 + 0.920147 w_k1_FDX_s014
                                 + 1.07723 w_k1_IBM_s014
                                 + 0.976434 w_k1_INTC_s014
                                 + 0.833754 w_k1_KO_s014
                                 + 1.19907 w_k1_MAT_s014
                                 + 0.830772 w_k1_MSFT_s014
                                 + 0.947904 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.17161 w_k1_ADM_s014
                                 + 0.680658 w_k1_AXP_s014
                                 + 0.562299 w_k1_DIS_s014
                                 + 0.748334 w_k1_FDX_s014
                                 + 1.04703 w_k1_IBM_s014
                                 + 0.721116 w_k1_INTC_s014
                                 + 0.832402 w_k1_KO_s014
                                 + 1.23578 w_k1_MAT_s014
                                 + 0.743193 w_k1_MSFT_s014
                                 + 0.822573 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.85361 w_k1_ADM_s014
                                 + 0.933973 w_k1_AXP_s014
                                 + 0.969111 w_k1_DIS_s014
                                 + 1.08543 w_k1_FDX_s014
                                 + 0.799561 w_k1_IBM_s014
                                 + 1.03566 w_k1_INTC_s014
                                 + 0.943928 w_k1_KO_s014
                                 + 0.954731 w_k1_MAT_s014
                                 + 0.969578 w_k1_MSFT_s014
                                 + 0.908567 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.25389 w_k1_ADM_s014
                                 + 1.23939 w_k1_AXP_s014
                                 + 1.37056 w_k1_DIS_s014
                                 + 1.33561 w_k1_FDX_s014
                                 + 0.850112 w_k1_IBM_s014
                                 + 1.20448 w_k1_INTC_s014
                                 + 1.02667 w_k1_KO_s014
                                 + 0.823592 w_k1_MAT_s014
                                 + 1.10364 w_k1_MSFT_s014
                                 + 1.22515 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.15949 w_k1_ADM_s014
                                 + 0.90526 w_k1_AXP_s014
                                 + 1.19221 w_k1_DIS_s014
                                 + 1.04573 w_k1_FDX_s014
                                 + 1.05828 w_k1_IBM_s014
                                 + 0.967608 w_k1_INTC_s014
                                 + 1.12067 w_k1_KO_s014 + 1.0972 w_k1_MAT_s014
                                 + 0.968013 w_k1_MSFT_s014
                                 + 1.07863 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.35928 w_k1_ADM_s015
                                 + 1.94106 w_k1_AXP_s015
                                 + 1.77337 w_k1_DIS_s015
                                 + 1.41399 w_k1_FDX_s015
                                 + 1.51469 w_k1_IBM_s015
                                 + 0.988453 w_k1_INTC_s015
                                 + 1.44246 w_k1_KO_s015 + 1.84918 w_k1_MAT_s015
                                 + 1.16087 w_k1_MSFT_s015
                                 + 1.28722 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.28661 w_k1_ADM_s015
                                 + 1.53829 w_k1_AXP_s015
                                 + 1.55724 w_k1_DIS_s015 + 1.1552 w_k1_FDX_s015
                                 + 1.82297 w_k1_IBM_s015
                                 + 0.993788 w_k1_INTC_s015
                                 + 1.23094 w_k1_KO_s015 + 1.7704 w_k1_MAT_s015
                                 + 0.907048 w_k1_MSFT_s015
                                 + 1.17976 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.727909 w_k1_ADM_s015
                                 + 0.819115 w_k1_AXP_s015
                                 + 0.682757 w_k1_DIS_s015
                                 + 0.753436 w_k1_FDX_s015
                                 + 0.833707 w_k1_IBM_s015
                                 + 0.707039 w_k1_INTC_s015
                                 + 0.677398 w_k1_KO_s015
                                 + 0.793017 w_k1_MAT_s015
                                 + 0.678153 w_k1_MSFT_s015
                                 + 0.786699 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.960479 w_k1_ADM_s015
                                 + 0.932798 w_k1_AXP_s015
                                 + 0.727496 w_k1_DIS_s015
                                 + 0.845802 w_k1_FDX_s015
                                 + 0.65176 w_k1_IBM_s015
                                 + 0.865522 w_k1_INTC_s015
                                 + 0.640565 w_k1_KO_s015
                                 + 0.52784 w_k1_MAT_s015
                                 + 0.768649 w_k1_MSFT_s015
                                 + 0.905642 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.32101 w_k1_ADM_s015
                                 + 1.6166 w_k1_AXP_s015 + 1.51741 w_k1_DIS_s015
                                 + 1.26778 w_k1_FDX_s015
                                 + 1.01778 w_k1_IBM_s015
                                 + 0.874507 w_k1_INTC_s015
                                 + 1.05422 w_k1_KO_s015 + 1.12401 w_k1_MAT_s015
                                 + 1.03517 w_k1_MSFT_s015
                                 + 1.28518 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.947103 w_k1_ADM_s015
                                 + 1.10764 w_k1_AXP_s015
                                 + 1.14372 w_k1_DIS_s015
                                 + 1.04901 w_k1_FDX_s015
                                 + 1.07368 w_k1_IBM_s015
                                 + 0.846123 w_k1_INTC_s015
                                 + 0.987475 w_k1_KO_s015
                                 + 1.03288 w_k1_MAT_s015
                                 + 0.835382 w_k1_MSFT_s015
                                 + 0.961358 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.07029 w_k1_ADM_s015
                                 + 1.20366 w_k1_AXP_s015
                                 + 1.01742 w_k1_DIS_s015
                                 + 1.13748 w_k1_FDX_s015
                                 + 0.793549 w_k1_IBM_s015
                                 + 1.05047 w_k1_INTC_s015
                                 + 0.786675 w_k1_KO_s015
                                 + 0.633118 w_k1_MAT_s015
                                 + 0.828318 w_k1_MSFT_s015
                                 + 0.909045 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.87476 w_k1_ADM_s015
                                 + 0.827534 w_k1_AXP_s015
                                 + 0.650772 w_k1_DIS_s015
                                 + 0.798177 w_k1_FDX_s015
                                 + 1.02546 w_k1_IBM_s015
                                 + 0.716227 w_k1_INTC_s015
                                 + 0.805228 w_k1_KO_s015
                                 + 1.03326 w_k1_MAT_s015
                                 + 0.706211 w_k1_MSFT_s015
                                 + 0.839089 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.83459 w_k1_ADM_s016
                                 + 0.814116 w_k1_AXP_s016
                                 + 0.608326 w_k1_DIS_s016
                                 + 0.833716 w_k1_FDX_s016
                                 + 0.700698 w_k1_IBM_s016
                                 + 0.787344 w_k1_INTC_s016
                                 + 0.745726 w_k1_KO_s016
                                 + 0.727048 w_k1_MAT_s016
                                 + 0.824182 w_k1_MSFT_s016
                                 + 0.824703 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.887908 w_k1_ADM_s016
                                 + 1.11065 w_k1_AXP_s016
                                 + 0.918437 w_k1_DIS_s016
                                 + 1.06811 w_k1_FDX_s016
                                 + 0.978633 w_k1_IBM_s016
                                 + 1.09994 w_k1_INTC_s016
                                 + 0.90492 w_k1_KO_s016 + 1.25219 w_k1_MAT_s016
                                 + 1.01994 w_k1_MSFT_s016
                                 + 0.996237 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.16738 w_k1_ADM_s016
                                 + 1.19691 w_k1_AXP_s016
                                 + 1.27543 w_k1_DIS_s016
                                 + 1.05502 w_k1_FDX_s016
                                 + 1.33569 w_k1_IBM_s016
                                 + 0.974164 w_k1_INTC_s016
                                 + 1.17526 w_k1_KO_s016 + 1.76322 w_k1_MAT_s016
                                 + 0.966428 w_k1_MSFT_s016
                                 + 1.04108 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.828927 w_k1_ADM_s016
                                 + 0.798075 w_k1_AXP_s016
                                 + 0.862221 w_k1_DIS_s016
                                 + 1.05507 w_k1_FDX_s016
                                 + 0.664817 w_k1_IBM_s016
                                 + 1.03193 w_k1_INTC_s016
                                 + 0.784382 w_k1_KO_s016
                                 + 0.563389 w_k1_MAT_s016
                                 + 0.875805 w_k1_MSFT_s016
                                 + 0.888637 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.10865 w_k1_ADM_s016
                                 + 0.94931 w_k1_AXP_s016
                                 + 1.11372 w_k1_DIS_s016
                                 + 1.00382 w_k1_FDX_s016
                                 + 0.870864 w_k1_IBM_s016
                                 + 1.01431 w_k1_INTC_s016
                                 + 0.915473 w_k1_KO_s016 + 0.8195 w_k1_MAT_s016
                                 + 1.06835 w_k1_MSFT_s016
                                 + 0.999244 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.923595 w_k1_ADM_s016
                                 + 1.21662 w_k1_AXP_s016
                                 + 1.07275 w_k1_DIS_s016
                                 + 0.851685 w_k1_FDX_s016
                                 + 1.62071 w_k1_IBM_s016
                                 + 0.783473 w_k1_INTC_s016
                                 + 1.08472 w_k1_KO_s016 + 1.68415 w_k1_MAT_s016
                                 + 0.770783 w_k1_MSFT_s016
                                 + 0.91349 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.803608 w_k1_ADM_s016
                                 + 0.740468 w_k1_AXP_s016
                                 + 0.737259 w_k1_DIS_s016
                                 + 0.814275 w_k1_FDX_s016
                                 + 0.955081 w_k1_IBM_s016
                                 + 0.85796 w_k1_INTC_s016
                                 + 0.911475 w_k1_KO_s016
                                 + 0.97797 w_k1_MAT_s016
                                 + 0.772913 w_k1_MSFT_s016
                                 + 0.88608 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.30723 w_k1_ADM_s016
                                 + 1.34367 w_k1_AXP_s016
                                 + 1.62059 w_k1_DIS_s016
                                 + 1.42677 w_k1_FDX_s016
                                 + 0.928632 w_k1_IBM_s016
                                 + 1.10632 w_k1_INTC_s016
                                 + 1.16784 w_k1_KO_s016 + 1.42724 w_k1_MAT_s016
                                 + 1.12104 w_k1_MSFT_s016
                                 + 1.12992 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.10643 w_k1_ADM_s017
                                 + 1.20264 w_k1_AXP_s017
                                 + 1.63032 w_k1_DIS_s017 + 1.253 w_k1_FDX_s017
                                 + 1.35576 w_k1_IBM_s017
                                 + 1.50436 w_k1_INTC_s017
                                 + 1.44115 w_k1_KO_s017 + 1.09273 w_k1_MAT_s017
                                 + 1.36729 w_k1_MSFT_s017
                                 + 1.18285 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.789307 w_k1_ADM_s017
                                 + 0.996746 w_k1_AXP_s017
                                 + 0.902114 w_k1_DIS_s017
                                 + 0.942299 w_k1_FDX_s017
                                 + 1.26893 w_k1_IBM_s017
                                 + 0.930445 w_k1_INTC_s017
                                 + 0.939264 w_k1_KO_s017
                                 + 1.03257 w_k1_MAT_s017
                                 + 0.86774 w_k1_MSFT_s017
                                 + 0.948145 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.923939 w_k1_ADM_s017
                                 + 0.769002 w_k1_AXP_s017
                                 + 0.994876 w_k1_DIS_s017
                                 + 0.843813 w_k1_FDX_s017
                                 + 0.874317 w_k1_IBM_s017
                                 + 0.945684 w_k1_INTC_s017
                                 + 1.00761 w_k1_KO_s017
                                 + 0.568895 w_k1_MAT_s017
                                 + 1.00184 w_k1_MSFT_s017
                                 + 0.938875 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.706764 w_k1_ADM_s017
                                 + 0.614008 w_k1_AXP_s017
                                 + 0.622596 w_k1_DIS_s017
                                 + 0.777708 w_k1_FDX_s017
                                 + 0.695286 w_k1_IBM_s017
                                 + 0.826526 w_k1_INTC_s017
                                 + 0.678949 w_k1_KO_s017
                                 + 0.454738 w_k1_MAT_s017
                                 + 0.73592 w_k1_MSFT_s017
                                 + 0.736865 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.583867 w_k1_ADM_s017
                                 + 0.695359 w_k1_AXP_s017
                                 + 0.827405 w_k1_DIS_s017
                                 + 0.785013 w_k1_FDX_s017
                                 + 1.21733 w_k1_IBM_s017
                                 + 0.768521 w_k1_INTC_s017
                                 + 0.878315 w_k1_KO_s017
                                 + 1.01419 w_k1_MAT_s017
                                 + 0.914027 w_k1_MSFT_s017
                                 + 0.915764 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.807759 w_k1_ADM_s017
                                 + 1.09732 w_k1_AXP_s017
                                 + 1.03764 w_k1_DIS_s017
                                 + 1.03245 w_k1_FDX_s017
                                 + 0.840724 w_k1_IBM_s017
                                 + 1.22545 w_k1_INTC_s017
                                 + 1.05458 w_k1_KO_s017 + 1.09496 w_k1_MAT_s017
                                 + 1.1181 w_k1_MSFT_s017
                                 + 0.982203 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.88807 w_k1_ADM_s017
                                 + 1.04504 w_k1_AXP_s017
                                 + 0.879438 w_k1_DIS_s017
                                 + 0.96223 w_k1_FDX_s017
                                 + 1.09614 w_k1_IBM_s017
                                 + 0.880976 w_k1_INTC_s017
                                 + 1.01018 w_k1_KO_s017 + 1.37394 w_k1_MAT_s017
                                 + 0.883472 w_k1_MSFT_s017
                                 + 0.970117 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.05682 w_k1_ADM_s017
                                 + 1.08151 w_k1_AXP_s017
                                 + 1.16383 w_k1_DIS_s017
                                 + 0.979959 w_k1_FDX_s017
                                 + 1.48693 w_k1_IBM_s017
                                 + 1.04313 w_k1_INTC_s017
                                 + 1.23883 w_k1_KO_s017 + 1.12426 w_k1_MAT_s017
                                 + 1.19093 w_k1_MSFT_s017 + 1.0489 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.866544 w_k1_ADM_s100
                                 + 0.830806 w_k1_AXP_s100
                                 + 0.742064 w_k1_DIS_s100
                                 + 0.830998 w_k1_FDX_s100
                                 + 0.857636 w_k1_IBM_s100
                                 + 0.730128 w_k1_INTC_s100
                                 + 0.773442 w_k1_KO_s100
                                 + 0.959145 w_k1_MAT_s100
                                 + 0.723575 w_k1_MSFT_s100
                                 + 0.872367 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.884671 w_k1_ADM_s100
                                 + 0.893638 w_k1_AXP_s100
                                 + 0.726612 w_k1_DIS_s100
                                 + 0.800619 w_k1_FDX_s100
                                 + 1.18533 w_k1_IBM_s100
                                 + 0.801033 w_k1_INTC_s100
                                 + 0.999535 w_k1_KO_s100
                                 + 1.25048 w_k1_MAT_s100
                                 + 0.94189 w_k1_MSFT_s100
                                 + 1.04021 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.861225 w_k1_ADM_s100
                                 + 0.879459 w_k1_AXP_s100
                                 + 0.641153 w_k1_DIS_s100
                                 + 0.788118 w_k1_FDX_s100
                                 + 0.702209 w_k1_IBM_s100
                                 + 0.656025 w_k1_INTC_s100
                                 + 0.623862 w_k1_KO_s100
                                 + 0.958078 w_k1_MAT_s100
                                 + 0.710056 w_k1_MSFT_s100
                                 + 0.88705 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.854354 w_k1_ADM_s100
                                 + 0.790656 w_k1_AXP_s100
                                 + 0.649612 w_k1_DIS_s100
                                 + 0.863274 w_k1_FDX_s100
                                 + 0.620033 w_k1_IBM_s100
                                 + 0.81009 w_k1_INTC_s100
                                 + 0.725624 w_k1_KO_s100
                                 + 0.755842 w_k1_MAT_s100
                                 + 0.872792 w_k1_MSFT_s100
                                 + 0.881512 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.05657 w_k1_ADM_s100
                                 + 0.906114 w_k1_AXP_s100
                                 + 0.954131 w_k1_DIS_s100
                                 + 1.04345 w_k1_FDX_s100
                                 + 0.712502 w_k1_IBM_s100
                                 + 1.07328 w_k1_INTC_s100
                                 + 0.94587 w_k1_KO_s100 + 0.55152 w_k1_MAT_s100
                                 + 1.00183 w_k1_MSFT_s100
                                 + 1.11376 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.07768 w_k1_ADM_s100
                                 + 1.07787 w_k1_AXP_s100
                                 + 1.23602 w_k1_DIS_s100
                                 + 1.11899 w_k1_FDX_s100
                                 + 0.871784 w_k1_IBM_s100
                                 + 1.25125 w_k1_INTC_s100
                                 + 1.10791 w_k1_KO_s100
                                 + 0.786112 w_k1_MAT_s100
                                 + 1.09434 w_k1_MSFT_s100
                                 + 1.14513 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.30753 w_k1_ADM_s100
                                 + 1.45439 w_k1_AXP_s100
                                 + 1.43717 w_k1_DIS_s100
                                 + 1.22844 w_k1_FDX_s100
                                 + 1.20749 w_k1_IBM_s100
                                 + 1.04613 w_k1_INTC_s100
                                 + 1.22207 w_k1_KO_s100 + 1.55018 w_k1_MAT_s100
                                 + 1.20912 w_k1_MSFT_s100 + 1.2028 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.19896 w_k1_ADM_s100
                                 + 1.20511 w_k1_AXP_s100
                                 + 1.09901 w_k1_DIS_s100
                                 + 1.04312 w_k1_FDX_s100
                                 + 1.64098 w_k1_IBM_s100
                                 + 0.90467 w_k1_INTC_s100
                                 + 1.23418 w_k1_KO_s100 + 1.75875 w_k1_MAT_s100
                                 + 0.979365 w_k1_MSFT_s100
                                 + 1.06286 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.582811 w_k1_ADM_s101
                                 + 0.813689 w_k1_AXP_s101
                                 + 0.780544 w_k1_DIS_s101
                                 + 0.782267 w_k1_FDX_s101
                                 + 1.2445 w_k1_IBM_s101
                                 + 1.00332 w_k1_INTC_s101
                                 + 0.936278 w_k1_KO_s101
                                 + 1.01633 w_k1_MAT_s101
                                 + 0.915244 w_k1_MSFT_s101
                                 + 1.0005 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.921503 w_k1_ADM_s101
                                 + 0.961048 w_k1_AXP_s101
                                 + 1.26941 w_k1_DIS_s101
                                 + 1.12722 w_k1_FDX_s101
                                 + 1.02132 w_k1_IBM_s101
                                 + 1.29989 w_k1_INTC_s101
                                 + 1.23885 w_k1_KO_s101
                                 + 0.821904 w_k1_MAT_s101
                                 + 1.32319 w_k1_MSFT_s101
                                 + 1.03841 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.09428 w_k1_ADM_s101
                                 + 1.64311 w_k1_AXP_s101
                                 + 1.52271 w_k1_DIS_s101
                                 + 1.44548 w_k1_FDX_s101
                                 + 1.23226 w_k1_IBM_s101
                                 + 1.57354 w_k1_INTC_s101
                                 + 1.23384 w_k1_KO_s101 + 1.19293 w_k1_MAT_s101
                                 + 1.344 w_k1_MSFT_s101 + 1.15122 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.745338 w_k1_ADM_s101
                                 + 1.22093 w_k1_AXP_s101
                                 + 1.06631 w_k1_DIS_s101
                                 + 1.11069 w_k1_FDX_s101
                                 + 1.07309 w_k1_IBM_s101
                                 + 1.16319 w_k1_INTC_s101
                                 + 1.03101 w_k1_KO_s101
                                 + 0.940638 w_k1_MAT_s101
                                 + 1.11591 w_k1_MSFT_s101
                                 + 1.00274 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.69084 w_k1_ADM_s101
                                 + 1.10882 w_k1_AXP_s101
                                 + 0.823477 w_k1_DIS_s101
                                 + 0.924445 w_k1_FDX_s101
                                 + 1.0745 w_k1_IBM_s101
                                 + 1.12678 w_k1_INTC_s101
                                 + 0.91613 w_k1_KO_s101 + 1.17672 w_k1_MAT_s101
                                 + 0.983724 w_k1_MSFT_s101
                                 + 0.844218 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.541042 w_k1_ADM_s101
                                 + 0.591853 w_k1_AXP_s101
                                 + 0.552259 w_k1_DIS_s101
                                 + 0.661025 w_k1_FDX_s101
                                 + 0.972556 w_k1_IBM_s101
                                 + 0.806607 w_k1_INTC_s101
                                 + 0.806945 w_k1_KO_s101
                                 + 0.809266 w_k1_MAT_s101
                                 + 0.869849 w_k1_MSFT_s101
                                 + 0.786507 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.581337 w_k1_ADM_s101
                                 + 0.66728 w_k1_AXP_s101
                                 + 0.541081 w_k1_DIS_s101
                                 + 0.691687 w_k1_FDX_s101
                                 + 0.599322 w_k1_IBM_s101
                                 + 0.917028 w_k1_INTC_s101
                                 + 0.660768 w_k1_KO_s101
                                 + 0.725231 w_k1_MAT_s101
                                 + 0.835086 w_k1_MSFT_s101
                                 + 0.714342 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.791361 w_k1_ADM_s101
                                 + 0.73851 w_k1_AXP_s101
                                 + 0.873012 w_k1_DIS_s101
                                 + 0.856627 w_k1_FDX_s101
                                 + 0.733958 w_k1_IBM_s101
                                 + 0.990552 w_k1_INTC_s101
                                 + 0.856094 w_k1_KO_s101
                                 + 0.729219 w_k1_MAT_s101
                                 + 0.910186 w_k1_MSFT_s101
                                 + 0.920859 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.917867 w_k1_ADM_s102
                                 + 0.801397 w_k1_AXP_s102
                                 + 1.16632 w_k1_DIS_s102
                                 + 1.00134 w_k1_FDX_s102
                                 + 0.963329 w_k1_IBM_s102
                                 + 1.03996 w_k1_INTC_s102
                                 + 1.05281 w_k1_KO_s102 + 1.05452 w_k1_MAT_s102
                                 + 0.981979 w_k1_MSFT_s102
                                 + 0.932934 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.16281 w_k1_ADM_s102
                                 + 1.09211 w_k1_AXP_s102
                                 + 1.36566 w_k1_DIS_s102
                                 + 1.11493 w_k1_FDX_s102
                                 + 1.66274 w_k1_IBM_s102
                                 + 0.998931 w_k1_INTC_s102
                                 + 1.28006 w_k1_KO_s102 + 1.94887 w_k1_MAT_s102
                                 + 1.08366 w_k1_MSFT_s102
                                 + 1.20097 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.14862 w_k1_ADM_s102
                                 + 1.14626 w_k1_AXP_s102
                                 + 1.41356 w_k1_DIS_s102
                                 + 1.28813 w_k1_FDX_s102
                                 + 0.95754 w_k1_IBM_s102
                                 + 1.18274 w_k1_INTC_s102
                                 + 1.08061 w_k1_KO_s102 + 1.24268 w_k1_MAT_s102
                                 + 1.29346 w_k1_MSFT_s102
                                 + 1.20819 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.662895 w_k1_ADM_s102
                                 + 0.671266 w_k1_AXP_s102
                                 + 0.845753 w_k1_DIS_s102
                                 + 0.774649 w_k1_FDX_s102
                                 + 1.14603 w_k1_IBM_s102
                                 + 0.858786 w_k1_INTC_s102
                                 + 0.92077 w_k1_KO_s102
                                 + 0.966984 w_k1_MAT_s102
                                 + 0.867251 w_k1_MSFT_s102
                                 + 0.817012 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.902431 w_k1_ADM_s102
                                 + 0.848105 w_k1_AXP_s102
                                 + 0.79719 w_k1_DIS_s102
                                 + 0.973542 w_k1_FDX_s102
                                 + 0.636801 w_k1_IBM_s102
                                 + 0.831633 w_k1_INTC_s102
                                 + 0.665497 w_k1_KO_s102
                                 + 0.63118 w_k1_MAT_s102
                                 + 0.920282 w_k1_MSFT_s102
                                 + 0.767792 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.730575 w_k1_ADM_s102
                                 + 0.935757 w_k1_AXP_s102
                                 + 0.858417 w_k1_DIS_s102
                                 + 0.977659 w_k1_FDX_s102
                                 + 0.852971 w_k1_IBM_s102
                                 + 1.08961 w_k1_INTC_s102
                                 + 0.802938 w_k1_KO_s102
                                 + 1.00403 w_k1_MAT_s102
                                 + 0.998135 w_k1_MSFT_s102
                                 + 0.866198 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.967829 w_k1_ADM_s102
                                 + 1.2605 w_k1_AXP_s102 + 1.38051 w_k1_DIS_s102
                                 + 1.23787 w_k1_FDX_s102
                                 + 1.30879 w_k1_IBM_s102
                                 + 1.17242 w_k1_INTC_s102
                                 + 1.03163 w_k1_KO_s102 + 1.37266 w_k1_MAT_s102
                                 + 1.14699 w_k1_MSFT_s102
                                 + 1.02376 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.812992 w_k1_ADM_s102
                                 + 1.28869 w_k1_AXP_s102
                                 + 0.964362 w_k1_DIS_s102
                                 + 1.01473 w_k1_FDX_s102
                                 + 1.28169 w_k1_IBM_s102
                                 + 0.908437 w_k1_INTC_s102
                                 + 0.911502 w_k1_KO_s102
                                 + 1.76122 w_k1_MAT_s102
                                 + 0.903661 w_k1_MSFT_s102
                                 + 1.03826 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.891709 w_k1_ADM_s103
                                 + 1.20326 w_k1_AXP_s103
                                 + 0.970458 w_k1_DIS_s103
                                 + 1.04287 w_k1_FDX_s103
                                 + 1.08695 w_k1_IBM_s103
                                 + 0.920807 w_k1_INTC_s103
                                 + 1.01976 w_k1_KO_s103 + 1.20369 w_k1_MAT_s103
                                 + 1.00547 w_k1_MSFT_s103
                                 + 1.13234 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.92859 w_k1_ADM_s103
                                 + 1.17807 w_k1_AXP_s103
                                 + 0.725588 w_k1_DIS_s103
                                 + 0.892611 w_k1_FDX_s103
                                 + 0.932907 w_k1_IBM_s103
                                 + 0.957469 w_k1_INTC_s103
                                 + 0.887278 w_k1_KO_s103
                                 + 1.62653 w_k1_MAT_s103
                                 + 0.938465 w_k1_MSFT_s103
                                 + 0.985428 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.21714 w_k1_ADM_s103
                                 + 1.19741 w_k1_AXP_s103
                                 + 1.05181 w_k1_DIS_s103
                                 + 1.14082 w_k1_FDX_s103
                                 + 0.596577 w_k1_IBM_s103
                                 + 1.13185 w_k1_INTC_s103
                                 + 0.877324 w_k1_KO_s103 + 0.921 w_k1_MAT_s103
                                 + 1.11382 w_k1_MSFT_s103
                                 + 1.10892 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.23952 w_k1_ADM_s103
                                 + 1.53027 w_k1_AXP_s103
                                 + 1.00711 w_k1_DIS_s103
                                 + 0.955134 w_k1_FDX_s103
                                 + 1.52006 w_k1_IBM_s103
                                 + 0.919514 w_k1_INTC_s103
                                 + 1.26555 w_k1_KO_s103 + 2.13937 w_k1_MAT_s103
                                 + 1.01292 w_k1_MSFT_s103
                                 + 1.13773 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.37297 w_k1_ADM_s103
                                 + 1.12618 w_k1_AXP_s103
                                 + 1.14526 w_k1_DIS_s103
                                 + 1.15796 w_k1_FDX_s103
                                 + 1.12885 w_k1_IBM_s103
                                 + 0.767384 w_k1_INTC_s103
                                 + 1.04447 w_k1_KO_s103 + 1.19319 w_k1_MAT_s103
                                 + 0.976319 w_k1_MSFT_s103
                                 + 1.11504 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.728698 w_k1_ADM_s103
                                 + 0.70428 w_k1_AXP_s103
                                 + 0.733222 w_k1_DIS_s103
                                 + 0.838661 w_k1_FDX_s103
                                 + 0.83526 w_k1_IBM_s103
                                 + 0.765814 w_k1_INTC_s103
                                 + 0.784941 w_k1_KO_s103
                                 + 0.845652 w_k1_MAT_s103
                                 + 0.840832 w_k1_MSFT_s103
                                 + 0.909638 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.642162 w_k1_ADM_s103
                                 + 0.876619 w_k1_AXP_s103
                                 + 0.587342 w_k1_DIS_s103
                                 + 0.862161 w_k1_FDX_s103
                                 + 0.636969 w_k1_IBM_s103
                                 + 0.635906 w_k1_INTC_s103
                                 + 0.598049 w_k1_KO_s103
                                 + 0.839221 w_k1_MAT_s103
                                 + 0.742561 w_k1_MSFT_s103
                                 + 0.751319 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.15094 w_k1_ADM_s103
                                 + 1.55677 w_k1_AXP_s103 + 1.5558 w_k1_DIS_s103
                                 + 1.32497 w_k1_FDX_s103
                                 + 1.07617 w_k1_IBM_s103
                                 + 1.34447 w_k1_INTC_s103
                                 + 1.39019 w_k1_KO_s103 + 1.80534 w_k1_MAT_s103
                                 + 1.4545 w_k1_MSFT_s103 + 1.25518 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.25888 w_k1_ADM_s104
                                 + 1.33062 w_k1_AXP_s104
                                 + 1.59805 w_k1_DIS_s104 + 1.3598 w_k1_FDX_s104
                                 + 1.11811 w_k1_IBM_s104
                                 + 1.22597 w_k1_INTC_s104
                                 + 1.08375 w_k1_KO_s104 + 1.20513 w_k1_MAT_s104
                                 + 1.28323 w_k1_MSFT_s104
                                 + 1.15311 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.08657 w_k1_ADM_s104
                                 + 1.30163 w_k1_AXP_s104
                                 + 1.07192 w_k1_DIS_s104
                                 + 1.10682 w_k1_FDX_s104
                                 + 0.75595 w_k1_IBM_s104
                                 + 1.02371 w_k1_INTC_s104
                                 + 0.887221 w_k1_KO_s104
                                 + 0.986003 w_k1_MAT_s104
                                 + 1.08927 w_k1_MSFT_s104
                                 + 1.04502 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.932939 w_k1_ADM_s104
                                 + 0.804198 w_k1_AXP_s104
                                 + 0.955218 w_k1_DIS_s104
                                 + 0.890424 w_k1_FDX_s104
                                 + 1.33858 w_k1_IBM_s104
                                 + 0.854541 w_k1_INTC_s104
                                 + 1.07938 w_k1_KO_s104 + 1.42772 w_k1_MAT_s104
                                 + 0.941931 w_k1_MSFT_s104
                                 + 1.00342 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.06102 w_k1_ADM_s104
                                 + 1.31657 w_k1_AXP_s104
                                 + 1.32528 w_k1_DIS_s104
                                 + 0.976046 w_k1_FDX_s104
                                 + 1.20432 w_k1_IBM_s104
                                 + 0.924809 w_k1_INTC_s104
                                 + 1.05783 w_k1_KO_s104 + 1.58451 w_k1_MAT_s104
                                 + 0.973582 w_k1_MSFT_s104
                                 + 1.12095 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.09193 w_k1_ADM_s104
                                 + 1.2848 w_k1_AXP_s104 + 1.16261 w_k1_DIS_s104
                                 + 1.09272 w_k1_FDX_s104
                                 + 1.09832 w_k1_IBM_s104
                                 + 1.13262 w_k1_INTC_s104
                                 + 1.13401 w_k1_KO_s104 + 1.51761 w_k1_MAT_s104
                                 + 1.11072 w_k1_MSFT_s104
                                 + 1.07485 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.00613 w_k1_ADM_s104
                                 + 0.882717 w_k1_AXP_s104
                                 + 0.928426 w_k1_DIS_s104
                                 + 0.993752 w_k1_FDX_s104
                                 + 0.680507 w_k1_IBM_s104
                                 + 0.877423 w_k1_INTC_s104
                                 + 0.737407 w_k1_KO_s104
                                 + 0.562577 w_k1_MAT_s104
                                 + 0.930348 w_k1_MSFT_s104
                                 + 0.878595 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.841793 w_k1_ADM_s104
                                 + 0.934075 w_k1_AXP_s104
                                 + 0.66724 w_k1_DIS_s104
                                 + 0.783148 w_k1_FDX_s104
                                 + 0.892124 w_k1_IBM_s104
                                 + 0.835446 w_k1_INTC_s104
                                 + 0.746454 w_k1_KO_s104
                                 + 0.848663 w_k1_MAT_s104
                                 + 0.785887 w_k1_MSFT_s104
                                 + 0.856138 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.735515 w_k1_ADM_s104
                                 + 0.584206 w_k1_AXP_s104
                                 + 0.6609 w_k1_DIS_s104
                                 + 0.715954 w_k1_FDX_s104
                                 + 0.585174 w_k1_IBM_s104
                                 + 0.741069 w_k1_INTC_s104
                                 + 0.525456 w_k1_KO_s104
                                 + 0.545592 w_k1_MAT_s104
                                 + 0.697781 w_k1_MSFT_s104
                                 + 0.754372 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.05466 w_k1_ADM_s105
                                 + 0.996686 w_k1_AXP_s105
                                 + 0.965614 w_k1_DIS_s105
                                 + 0.983103 w_k1_FDX_s105
                                 + 1.02756 w_k1_IBM_s105
                                 + 0.810246 w_k1_INTC_s105
                                 + 0.949198 w_k1_KO_s105
                                 + 0.732132 w_k1_MAT_s105
                                 + 0.9294 w_k1_MSFT_s105
                                 + 0.950509 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.698204 w_k1_ADM_s105
                                 + 0.878895 w_k1_AXP_s105
                                 + 0.762924 w_k1_DIS_s105
                                 + 0.867638 w_k1_FDX_s105
                                 + 0.771427 w_k1_IBM_s105
                                 + 0.940413 w_k1_INTC_s105
                                 + 0.731422 w_k1_KO_s105
                                 + 0.72018 w_k1_MAT_s105
                                 + 0.788402 w_k1_MSFT_s105
                                 + 0.939559 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.863289 w_k1_ADM_s105
                                 + 0.979703 w_k1_AXP_s105
                                 + 0.906819 w_k1_DIS_s105
                                 + 0.877584 w_k1_FDX_s105
                                 + 1.31593 w_k1_IBM_s105
                                 + 1.07608 w_k1_INTC_s105
                                 + 0.998352 w_k1_KO_s105 + 1.0193 w_k1_MAT_s105
                                 + 0.943398 w_k1_MSFT_s105
                                 + 0.963164 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.20831 w_k1_ADM_s105
                                 + 1.69401 w_k1_AXP_s105
                                 + 1.52233 w_k1_DIS_s105
                                 + 1.33628 w_k1_FDX_s105
                                 + 1.26055 w_k1_IBM_s105
                                 + 1.45276 w_k1_INTC_s105
                                 + 1.34071 w_k1_KO_s105
                                 + 0.993132 w_k1_MAT_s105
                                 + 1.24595 w_k1_MSFT_s105
                                 + 1.21079 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.12658 w_k1_ADM_s105
                                 + 1.28366 w_k1_AXP_s105
                                 + 1.07548 w_k1_DIS_s105
                                 + 1.13738 w_k1_FDX_s105
                                 + 1.06013 w_k1_IBM_s105
                                 + 1.17749 w_k1_INTC_s105
                                 + 1.12538 w_k1_KO_s105 + 1.1201 w_k1_MAT_s105
                                 + 1.14959 w_k1_MSFT_s105
                                 + 1.15171 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.788803 w_k1_ADM_s105
                                 + 1.08389 w_k1_AXP_s105
                                 + 0.796225 w_k1_DIS_s105
                                 + 1.04019 w_k1_FDX_s105
                                 + 0.740525 w_k1_IBM_s105
                                 + 1.14261 w_k1_INTC_s105
                                 + 0.814474 w_k1_KO_s105
                                 + 0.85752 w_k1_MAT_s105
                                 + 0.930246 w_k1_MSFT_s105
                                 + 0.977113 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.643111 w_k1_ADM_s105
                                 + 0.82795 w_k1_AXP_s105
                                 + 0.597217 w_k1_DIS_s105
                                 + 0.667641 w_k1_FDX_s105
                                 + 0.846024 w_k1_IBM_s105
                                 + 0.665612 w_k1_INTC_s105
                                 + 0.628316 w_k1_KO_s105
                                 + 0.798464 w_k1_MAT_s105
                                 + 0.682229 w_k1_MSFT_s105
                                 + 0.859919 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.09586 w_k1_ADM_s105
                                 + 1.58083 w_k1_AXP_s105
                                 + 1.09459 w_k1_DIS_s105 + 1.141 w_k1_FDX_s105
                                 + 1.38619 w_k1_IBM_s105
                                 + 1.22803 w_k1_INTC_s105
                                 + 1.10402 w_k1_KO_s105 + 1.44182 w_k1_MAT_s105
                                 + 1.09679 w_k1_MSFT_s105
                                 + 1.10253 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.980339 w_k1_ADM_s106
                                 + 1.02382 w_k1_AXP_s106
                                 + 0.981545 w_k1_DIS_s106
                                 + 0.997915 w_k1_FDX_s106
                                 + 0.950447 w_k1_IBM_s106
                                 + 0.892706 w_k1_INTC_s106
                                 + 0.874882 w_k1_KO_s106
                                 + 1.05143 w_k1_MAT_s106
                                 + 0.92043 w_k1_MSFT_s106
                                 + 0.925973 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.857576 w_k1_ADM_s106
                                 + 0.669742 w_k1_AXP_s106
                                 + 0.536014 w_k1_DIS_s106
                                 + 0.774765 w_k1_FDX_s106
                                 + 0.693017 w_k1_IBM_s106
                                 + 0.811962 w_k1_INTC_s106
                                 + 0.678784 w_k1_KO_s106
                                 + 0.693685 w_k1_MAT_s106
                                 + 0.791733 w_k1_MSFT_s106
                                 + 0.927072 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.22707 w_k1_ADM_s106
                                 + 1.03603 w_k1_AXP_s106
                                 + 0.986224 w_k1_DIS_s106
                                 + 1.10775 w_k1_FDX_s106
                                 + 0.705393 w_k1_IBM_s106
                                 + 0.935518 w_k1_INTC_s106
                                 + 0.93625 w_k1_KO_s106
                                 + 0.743216 w_k1_MAT_s106
                                 + 0.947246 w_k1_MSFT_s106
                                 + 1.02867 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.839674 w_k1_ADM_s106
                                 + 0.846124 w_k1_AXP_s106
                                 + 0.782259 w_k1_DIS_s106
                                 + 0.905328 w_k1_FDX_s106
                                 + 0.704506 w_k1_IBM_s106
                                 + 0.912367 w_k1_INTC_s106
                                 + 0.803903 w_k1_KO_s106
                                 + 0.834017 w_k1_MAT_s106
                                 + 0.880911 w_k1_MSFT_s106
                                 + 0.875269 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.15359 w_k1_ADM_s106
                                 + 1.53089 w_k1_AXP_s106
                                 + 1.16301 w_k1_DIS_s106
                                 + 1.12706 w_k1_FDX_s106
                                 + 1.49718 w_k1_IBM_s106
                                 + 0.957977 w_k1_INTC_s106
                                 + 1.21522 w_k1_KO_s106 + 2.40105 w_k1_MAT_s106
                                 + 1.03629 w_k1_MSFT_s106
                                 + 1.23448 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.10069 w_k1_ADM_s106
                                 + 1.33255 w_k1_AXP_s106
                                 + 0.985207 w_k1_DIS_s106
                                 + 1.02075 w_k1_FDX_s106
                                 + 1.04058 w_k1_IBM_s106
                                 + 0.901023 w_k1_INTC_s106
                                 + 1.00479 w_k1_KO_s106 + 1.36717 w_k1_MAT_s106
                                 + 0.902739 w_k1_MSFT_s106
                                 + 0.912077 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.03644 w_k1_ADM_s106
                                 + 0.937263 w_k1_AXP_s106
                                 + 0.891128 w_k1_DIS_s106
                                 + 0.86127 w_k1_FDX_s106
                                 + 1.22419 w_k1_IBM_s106
                                 + 0.80036 w_k1_INTC_s106
                                 + 1.13279 w_k1_KO_s106 + 1.19525 w_k1_MAT_s106
                                 + 0.819204 w_k1_MSFT_s106
                                 + 1.01281 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.45414 w_k1_ADM_s106
                                 + 1.34394 w_k1_AXP_s106
                                 + 1.36729 w_k1_DIS_s106
                                 + 1.31231 w_k1_FDX_s106
                                 + 1.12122 w_k1_IBM_s106
                                 + 1.09713 w_k1_INTC_s106
                                 + 1.09924 w_k1_KO_s106
                                 + 0.937805 w_k1_MAT_s106
                                 + 1.04729 w_k1_MSFT_s106
                                 + 1.05537 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.856057 w_k1_ADM_s107
                                 + 1.60118 w_k1_AXP_s107
                                 + 1.19478 w_k1_DIS_s107
                                 + 1.13836 w_k1_FDX_s107
                                 + 1.36105 w_k1_IBM_s107
                                 + 1.01389 w_k1_INTC_s107
                                 + 1.20044 w_k1_KO_s107 + 1.71635 w_k1_MAT_s107
                                 + 0.949583 w_k1_MSFT_s107
                                 + 1.17929 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.893585 w_k1_ADM_s107
                                 + 0.978469 w_k1_AXP_s107
                                 + 0.931138 w_k1_DIS_s107
                                 + 0.938833 w_k1_FDX_s107
                                 + 0.824247 w_k1_IBM_s107
                                 + 0.761398 w_k1_INTC_s107
                                 + 0.947659 w_k1_KO_s107
                                 + 0.797081 w_k1_MAT_s107
                                 + 0.808951 w_k1_MSFT_s107
                                 + 0.946825 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.784441 w_k1_ADM_s107
                                 + 0.718331 w_k1_AXP_s107
                                 + 0.73335 w_k1_DIS_s107
                                 + 0.733415 w_k1_FDX_s107
                                 + 0.822014 w_k1_IBM_s107
                                 + 0.688193 w_k1_INTC_s107
                                 + 0.766183 w_k1_KO_s107
                                 + 0.760869 w_k1_MAT_s107
                                 + 0.701083 w_k1_MSFT_s107
                                 + 0.813546 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.27148 w_k1_ADM_s107
                                 + 1.03207 w_k1_AXP_s107
                                 + 1.23393 w_k1_DIS_s107
                                 + 1.18783 w_k1_FDX_s107
                                 + 0.707422 w_k1_IBM_s107
                                 + 0.852588 w_k1_INTC_s107
                                 + 0.865758 w_k1_KO_s107
                                 + 0.526061 w_k1_MAT_s107
                                 + 0.988446 w_k1_MSFT_s107
                                 + 0.994188 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.906472 w_k1_ADM_s107
                                 + 1.10629 w_k1_AXP_s107
                                 + 1.02818 w_k1_DIS_s107
                                 + 1.04951 w_k1_FDX_s107
                                 + 1.08146 w_k1_IBM_s107
                                 + 0.890955 w_k1_INTC_s107
                                 + 1.15201 w_k1_KO_s107 + 1.06616 w_k1_MAT_s107
                                 + 1.04485 w_k1_MSFT_s107
                                 + 1.09271 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.995522 w_k1_ADM_s107
                                 + 0.875552 w_k1_AXP_s107
                                 + 1.33756 w_k1_DIS_s107
                                 + 1.05475 w_k1_FDX_s107
                                 + 1.26419 w_k1_IBM_s107
                                 + 0.989086 w_k1_INTC_s107
                                 + 1.46691 w_k1_KO_s107 + 1.067 w_k1_MAT_s107
                                 + 1.14081 w_k1_MSFT_s107
                                 + 1.13853 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.956004 w_k1_ADM_s107
                                 + 1.23326 w_k1_AXP_s107
                                 + 1.35185 w_k1_DIS_s107
                                 + 1.30945 w_k1_FDX_s107
                                 + 0.860277 w_k1_IBM_s107
                                 + 0.905936 w_k1_INTC_s107
                                 + 1.06824 w_k1_KO_s107 + 1.13848 w_k1_MAT_s107
                                 + 1.1289 w_k1_MSFT_s107 + 1.16621 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.861585 w_k1_ADM_s107
                                 + 0.970819 w_k1_AXP_s107
                                 + 1.02171 w_k1_DIS_s107
                                 + 0.916869 w_k1_FDX_s107
                                 + 1.25713 w_k1_IBM_s107
                                 + 0.978056 w_k1_INTC_s107
                                 + 1.11247 w_k1_KO_s107 + 1.26222 w_k1_MAT_s107
                                 + 0.844387 w_k1_MSFT_s107
                                 + 0.968494 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.08786 w_k1_ADM_s110
                                 + 1.34217 w_k1_AXP_s110
                                 + 1.72986 w_k1_DIS_s110
                                 + 1.21043 w_k1_FDX_s110 + 1.2007 w_k1_IBM_s110
                                 + 1.17923 w_k1_INTC_s110
                                 + 1.41997 w_k1_KO_s110 + 1.04977 w_k1_MAT_s110
                                 + 1.1807 w_k1_MSFT_s110 + 1.01082 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.3529 w_k1_ADM_s110
                                 + 1.69362 w_k1_AXP_s110
                                 + 1.54827 w_k1_DIS_s110
                                 + 1.53385 w_k1_FDX_s110
                                 + 1.36527 w_k1_IBM_s110
                                 + 1.21438 w_k1_INTC_s110
                                 + 1.43415 w_k1_KO_s110 + 1.69723 w_k1_MAT_s110
                                 + 1.29222 w_k1_MSFT_s110
                                 + 1.08862 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.972305 w_k1_ADM_s110
                                 + 0.731109 w_k1_AXP_s110
                                 + 0.915529 w_k1_DIS_s110
                                 + 0.983885 w_k1_FDX_s110
                                 + 0.627162 w_k1_IBM_s110
                                 + 0.795661 w_k1_INTC_s110
                                 + 0.837953 w_k1_KO_s110
                                 + 0.503929 w_k1_MAT_s110
                                 + 0.864311 w_k1_MSFT_s110
                                 + 0.936184 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.06242 w_k1_ADM_s110
                                 + 0.726827 w_k1_AXP_s110
                                 + 0.7189 w_k1_DIS_s110
                                 + 0.842919 w_k1_FDX_s110
                                 + 0.722169 w_k1_IBM_s110
                                 + 0.86752 w_k1_INTC_s110
                                 + 0.803737 w_k1_KO_s110
                                 + 0.548767 w_k1_MAT_s110
                                 + 0.722836 w_k1_MSFT_s110
                                 + 0.7428 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.00469 w_k1_ADM_s110
                                 + 1.28652 w_k1_AXP_s110
                                 + 1.14642 w_k1_DIS_s110
                                 + 1.25413 w_k1_FDX_s110
                                 + 0.94931 w_k1_IBM_s110
                                 + 1.01794 w_k1_INTC_s110
                                 + 0.993485 w_k1_KO_s110
                                 + 0.987279 w_k1_MAT_s110
                                 + 1.03924 w_k1_MSFT_s110
                                 + 1.09331 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.92908 w_k1_ADM_s110
                                 + 0.858354 w_k1_AXP_s110
                                 + 0.990601 w_k1_DIS_s110
                                 + 0.947471 w_k1_FDX_s110
                                 + 1.04762 w_k1_IBM_s110
                                 + 0.94992 w_k1_INTC_s110
                                 + 1.09629 w_k1_KO_s110
                                 + 0.890614 w_k1_MAT_s110
                                 + 0.893649 w_k1_MSFT_s110
                                 + 1.04421 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.756536 w_k1_ADM_s110
                                 + 0.758292 w_k1_AXP_s110
                                 + 0.594622 w_k1_DIS_s110
                                 + 0.796693 w_k1_FDX_s110
                                 + 0.829607 w_k1_IBM_s110
                                 + 0.757152 w_k1_INTC_s110
                                 + 0.803756 w_k1_KO_s110
                                 + 0.816657 w_k1_MAT_s110
                                 + 0.736114 w_k1_MSFT_s110
                                 + 0.761936 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.18878 w_k1_ADM_s110
                                 + 1.07327 w_k1_AXP_s110
                                 + 1.32126 w_k1_DIS_s110
                                 + 1.02705 w_k1_FDX_s110
                                 + 1.59842 w_k1_IBM_s110
                                 + 0.843772 w_k1_INTC_s110
                                 + 1.33397 w_k1_KO_s110 + 1.2989 w_k1_MAT_s110
                                 + 1.08357 w_k1_MSFT_s110
                                 + 1.10617 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.01949 w_k1_ADM_s111
                                 + 0.727762 w_k1_AXP_s111
                                 + 0.889959 w_k1_DIS_s111
                                 + 0.900153 w_k1_FDX_s111
                                 + 0.74131 w_k1_IBM_s111
                                 + 1.01411 w_k1_INTC_s111
                                 + 0.874188 w_k1_KO_s111
                                 + 0.737844 w_k1_MAT_s111
                                 + 0.981297 w_k1_MSFT_s111
                                 + 0.936171 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.44347 w_k1_ADM_s111
                                 + 1.32916 w_k1_AXP_s111
                                 + 1.12849 w_k1_DIS_s111
                                 + 0.983523 w_k1_FDX_s111
                                 + 1.67874 w_k1_IBM_s111
                                 + 0.96451 w_k1_INTC_s111
                                 + 1.24067 w_k1_KO_s111 + 2.23155 w_k1_MAT_s111
                                 + 0.860042 w_k1_MSFT_s111
                                 + 1.18341 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.26034 w_k1_ADM_s111
                                 + 1.29137 w_k1_AXP_s111
                                 + 1.19595 w_k1_DIS_s111
                                 + 1.23836 w_k1_FDX_s111
                                 + 0.882132 w_k1_IBM_s111
                                 + 0.993313 w_k1_INTC_s111
                                 + 1.04454 w_k1_KO_s111 + 1.23112 w_k1_MAT_s111
                                 + 1.08314 w_k1_MSFT_s111
                                 + 1.18111 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.830369 w_k1_ADM_s111
                                 + 0.736822 w_k1_AXP_s111
                                 + 0.719252 w_k1_DIS_s111
                                 + 0.770813 w_k1_FDX_s111
                                 + 1.00041 w_k1_IBM_s111
                                 + 0.786335 w_k1_INTC_s111
                                 + 0.82547 w_k1_KO_s111 + 1.13932 w_k1_MAT_s111
                                 + 0.751948 w_k1_MSFT_s111
                                 + 0.912107 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04309 w_k1_ADM_s111
                                 + 1.08936 w_k1_AXP_s111
                                 + 1.24091 w_k1_DIS_s111
                                 + 1.05538 w_k1_FDX_s111
                                 + 1.28645 w_k1_IBM_s111
                                 + 0.977518 w_k1_INTC_s111
                                 + 1.30908 w_k1_KO_s111 + 1.53895 w_k1_MAT_s111
                                 + 0.916435 w_k1_MSFT_s111
                                 + 1.17157 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.02369 w_k1_ADM_s111
                                 + 1.01121 w_k1_AXP_s111
                                 + 0.822312 w_k1_DIS_s111
                                 + 0.963541 w_k1_FDX_s111
                                 + 0.842698 w_k1_IBM_s111
                                 + 0.679094 w_k1_INTC_s111
                                 + 0.784867 w_k1_KO_s111
                                 + 1.44866 w_k1_MAT_s111
                                 + 0.865293 w_k1_MSFT_s111
                                 + 1.07325 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.812245 w_k1_ADM_s111
                                 + 0.724596 w_k1_AXP_s111
                                 + 0.600915 w_k1_DIS_s111
                                 + 0.766751 w_k1_FDX_s111
                                 + 0.558301 w_k1_IBM_s111
                                 + 0.724192 w_k1_INTC_s111
                                 + 0.697339 w_k1_KO_s111
                                 + 0.837787 w_k1_MAT_s111
                                 + 0.779481 w_k1_MSFT_s111
                                 + 0.901319 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.52189 w_k1_ADM_s111
                                 + 1.55955 w_k1_AXP_s111
                                 + 1.86405 w_k1_DIS_s111
                                 + 1.19005 w_k1_FDX_s111
                                 + 1.96488 w_k1_IBM_s111
                                 + 1.11316 w_k1_INTC_s111
                                 + 1.74784 w_k1_KO_s111 + 3.37536 w_k1_MAT_s111
                                 + 1.21017 w_k1_MSFT_s111
                                 + 1.49606 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.872056 w_k1_ADM_s112
                                 + 0.737815 w_k1_AXP_s112
                                 + 0.793529 w_k1_DIS_s112
                                 + 0.867877 w_k1_FDX_s112
                                 + 1.09947 w_k1_IBM_s112
                                 + 0.835055 w_k1_INTC_s112
                                 + 0.82169 w_k1_KO_s112
                                 + 0.872573 w_k1_MAT_s112
                                 + 0.826238 w_k1_MSFT_s112
                                 + 0.926006 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.24772 w_k1_ADM_s112
                                 + 1.19325 w_k1_AXP_s112
                                 + 1.19713 w_k1_DIS_s112
                                 + 1.08229 w_k1_FDX_s112 + 1.3945 w_k1_IBM_s112
                                 + 1.36598 w_k1_INTC_s112
                                 + 1.30484 w_k1_KO_s112 + 1.33919 w_k1_MAT_s112
                                 + 1.16357 w_k1_MSFT_s112
                                 + 1.08217 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.845848 w_k1_ADM_s112
                                 + 0.819594 w_k1_AXP_s112
                                 + 0.881293 w_k1_DIS_s112
                                 + 0.958526 w_k1_FDX_s112
                                 + 0.736428 w_k1_IBM_s112
                                 + 1.0588 w_k1_INTC_s112
                                 + 0.863228 w_k1_KO_s112
                                 + 0.393738 w_k1_MAT_s112
                                 + 1.02769 w_k1_MSFT_s112
                                 + 0.930289 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.77975 w_k1_ADM_s112
                                 + 0.603848 w_k1_AXP_s112
                                 + 0.553327 w_k1_DIS_s112
                                 + 0.694375 w_k1_FDX_s112
                                 + 0.831022 w_k1_IBM_s112
                                 + 0.925873 w_k1_INTC_s112
                                 + 0.767102 w_k1_KO_s112 + 0.8636 w_k1_MAT_s112
                                 + 0.8103 w_k1_MSFT_s112
                                 + 0.791789 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.01641 w_k1_ADM_s112
                                 + 0.834604 w_k1_AXP_s112
                                 + 1.11039 w_k1_DIS_s112
                                 + 0.962136 w_k1_FDX_s112
                                 + 1.1467 w_k1_IBM_s112
                                 + 1.04499 w_k1_INTC_s112
                                 + 1.05366 w_k1_KO_s112
                                 + 0.862558 w_k1_MAT_s112
                                 + 1.04018 w_k1_MSFT_s112
                                 + 1.05926 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.97937 w_k1_ADM_s112
                                 + 1.24773 w_k1_AXP_s112
                                 + 1.19245 w_k1_DIS_s112
                                 + 1.28606 w_k1_FDX_s112
                                 + 0.953416 w_k1_IBM_s112
                                 + 1.48886 w_k1_INTC_s112
                                 + 1.12317 w_k1_KO_s112 + 1.16412 w_k1_MAT_s112
                                 + 1.20206 w_k1_MSFT_s112 + 1.256 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.751611 w_k1_ADM_s112
                                 + 1.03686 w_k1_AXP_s112
                                 + 0.610961 w_k1_DIS_s112
                                 + 0.744356 w_k1_FDX_s112
                                 + 1.24039 w_k1_IBM_s112
                                 + 0.934062 w_k1_INTC_s112
                                 + 0.90557 w_k1_KO_s112 + 1.526 w_k1_MAT_s112
                                 + 0.790737 w_k1_MSFT_s112
                                 + 0.877521 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.14196 w_k1_ADM_s112
                                 + 1.33529 w_k1_AXP_s112
                                 + 1.63899 w_k1_DIS_s112
                                 + 1.40421 w_k1_FDX_s112
                                 + 1.19757 w_k1_IBM_s112
                                 + 1.45606 w_k1_INTC_s112
                                 + 1.15623 w_k1_KO_s112 + 1.65824 w_k1_MAT_s112
                                 + 1.41363 w_k1_MSFT_s112
                                 + 1.07465 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.22538 w_k1_ADM_s113
                                 + 1.31767 w_k1_AXP_s113
                                 + 1.07973 w_k1_DIS_s113
                                 + 1.02264 w_k1_FDX_s113
                                 + 1.07113 w_k1_IBM_s113
                                 + 1.10666 w_k1_INTC_s113
                                 + 1.04089 w_k1_KO_s113 + 1.34672 w_k1_MAT_s113
                                 + 1.10305 w_k1_MSFT_s113
                                 + 1.27213 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.982833 w_k1_ADM_s113
                                 + 0.836623 w_k1_AXP_s113
                                 + 1.02407 w_k1_DIS_s113
                                 + 0.917988 w_k1_FDX_s113
                                 + 0.949017 w_k1_IBM_s113
                                 + 0.793112 w_k1_INTC_s113
                                 + 0.990742 w_k1_KO_s113
                                 + 1.02552 w_k1_MAT_s113
                                 + 1.07589 w_k1_MSFT_s113
                                 + 1.00161 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.33718 w_k1_ADM_s113
                                 + 1.14219 w_k1_AXP_s113
                                 + 0.902116 w_k1_DIS_s113
                                 + 1.08222 w_k1_FDX_s113
                                 + 0.92295 w_k1_IBM_s113
                                 + 0.793743 w_k1_INTC_s113
                                 + 0.961196 w_k1_KO_s113
                                 + 1.03634 w_k1_MAT_s113
                                 + 1.06134 w_k1_MSFT_s113
                                 + 0.913906 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.822155 w_k1_ADM_s113
                                 + 0.669493 w_k1_AXP_s113
                                 + 0.593306 w_k1_DIS_s113
                                 + 0.753823 w_k1_FDX_s113
                                 + 0.785948 w_k1_IBM_s113
                                 + 0.751027 w_k1_INTC_s113
                                 + 0.729299 w_k1_KO_s113
                                 + 0.661945 w_k1_MAT_s113
                                 + 0.858354 w_k1_MSFT_s113
                                 + 0.841792 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.02077 w_k1_ADM_s113
                                 + 1.24952 w_k1_AXP_s113
                                 + 1.46632 w_k1_DIS_s113 + 1.1214 w_k1_FDX_s113
                                 + 1.26752 w_k1_IBM_s113
                                 + 1.10196 w_k1_INTC_s113
                                 + 1.36818 w_k1_KO_s113 + 1.19675 w_k1_MAT_s113
                                 + 1.18746 w_k1_MSFT_s113
                                 + 1.03839 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.945977 w_k1_ADM_s113
                                 + 0.795347 w_k1_AXP_s113
                                 + 0.780816 w_k1_DIS_s113
                                 + 0.883851 w_k1_FDX_s113
                                 + 0.804296 w_k1_IBM_s113
                                 + 0.964379 w_k1_INTC_s113
                                 + 0.804935 w_k1_KO_s113
                                 + 0.553535 w_k1_MAT_s113
                                 + 0.93766 w_k1_MSFT_s113
                                 + 0.836711 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.87017 w_k1_ADM_s113
                                 + 1.45192 w_k1_AXP_s113
                                 + 1.55158 w_k1_DIS_s113
                                 + 1.40141 w_k1_FDX_s113
                                 + 0.988589 w_k1_IBM_s113
                                 + 1.13187 w_k1_INTC_s113
                                 + 1.27769 w_k1_KO_s113 + 1.11317 w_k1_MAT_s113
                                 + 1.47408 w_k1_MSFT_s113
                                 + 1.25133 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.961498 w_k1_ADM_s113
                                 + 0.881821 w_k1_AXP_s113
                                 + 0.676442 w_k1_DIS_s113
                                 + 0.749218 w_k1_FDX_s113
                                 + 1.21392 w_k1_IBM_s113
                                 + 0.767412 w_k1_INTC_s113
                                 + 0.970568 w_k1_KO_s113
                                 + 1.21204 w_k1_MAT_s113
                                 + 0.85831 w_k1_MSFT_s113
                                 + 0.958641 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.881758 w_k1_ADM_s114
                                 + 1.1144 w_k1_AXP_s114
                                 + 0.767549 w_k1_DIS_s114
                                 + 0.999966 w_k1_FDX_s114
                                 + 1.34567 w_k1_IBM_s114
                                 + 1.0993 w_k1_INTC_s114 + 1.08015 w_k1_KO_s114
                                 + 1.11031 w_k1_MAT_s114
                                 + 0.986643 w_k1_MSFT_s114
                                 + 0.975983 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.835683 w_k1_ADM_s114
                                 + 0.704647 w_k1_AXP_s114
                                 + 0.500605 w_k1_DIS_s114
                                 + 0.785715 w_k1_FDX_s114
                                 + 0.648114 w_k1_IBM_s114
                                 + 0.83717 w_k1_INTC_s114
                                 + 0.610935 w_k1_KO_s114
                                 + 0.470513 w_k1_MAT_s114
                                 + 0.765076 w_k1_MSFT_s114
                                 + 0.704595 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.650082 w_k1_ADM_s114
                                 + 0.853195 w_k1_AXP_s114
                                 + 0.7803 w_k1_DIS_s114
                                 + 0.783406 w_k1_FDX_s114
                                 + 1.23255 w_k1_IBM_s114
                                 + 0.710401 w_k1_INTC_s114
                                 + 1.09815 w_k1_KO_s114
                                 + 0.953992 w_k1_MAT_s114
                                 + 0.847583 w_k1_MSFT_s114
                                 + 0.8616 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.02232 w_k1_ADM_s114
                                 + 1.15434 w_k1_AXP_s114
                                 + 0.890308 w_k1_DIS_s114
                                 + 1.22158 w_k1_FDX_s114
                                 + 0.83095 w_k1_IBM_s114
                                 + 0.89465 w_k1_INTC_s114
                                 + 0.925737 w_k1_KO_s114
                                 + 0.874237 w_k1_MAT_s114
                                 + 1.13399 w_k1_MSFT_s114
                                 + 1.04633 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.20595 w_k1_ADM_s114
                                 + 1.20788 w_k1_AXP_s114
                                 + 1.18348 w_k1_DIS_s114
                                 + 1.26691 w_k1_FDX_s114
                                 + 1.00126 w_k1_IBM_s114
                                 + 0.944412 w_k1_INTC_s114
                                 + 1.17413 w_k1_KO_s114
                                 + 0.820069 w_k1_MAT_s114
                                 + 1.19011 w_k1_MSFT_s114 + 1.0262 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.81413 w_k1_ADM_s114
                                 + 0.880884 w_k1_AXP_s114
                                 + 0.74459 w_k1_DIS_s114
                                 + 0.903658 w_k1_FDX_s114
                                 + 0.991911 w_k1_IBM_s114
                                 + 0.847837 w_k1_INTC_s114
                                 + 0.85489 w_k1_KO_s114
                                 + 0.752883 w_k1_MAT_s114
                                 + 0.851161 w_k1_MSFT_s114
                                 + 0.808644 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04066 w_k1_ADM_s114
                                 + 0.977227 w_k1_AXP_s114
                                 + 0.954915 w_k1_DIS_s114
                                 + 1.04005 w_k1_FDX_s114
                                 + 0.862787 w_k1_IBM_s114
                                 + 0.923043 w_k1_INTC_s114
                                 + 0.964861 w_k1_KO_s114
                                 + 0.531376 w_k1_MAT_s114
                                 + 1.01738 w_k1_MSFT_s114
                                 + 0.944722 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.14719 w_k1_ADM_s114
                                 + 1.58914 w_k1_AXP_s114
                                 + 1.56898 w_k1_DIS_s114
                                 + 1.37411 w_k1_FDX_s114
                                 + 1.35922 w_k1_IBM_s114
                                 + 1.16029 w_k1_INTC_s114
                                 + 1.47315 w_k1_KO_s114 + 1.56057 w_k1_MAT_s114
                                 + 1.29596 w_k1_MSFT_s114 + 1.1106 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.941945 w_k1_ADM_s115
                                 + 1.02496 w_k1_AXP_s115
                                 + 0.905504 w_k1_DIS_s115
                                 + 1.0301 w_k1_FDX_s115
                                 + 0.726864 w_k1_IBM_s115
                                 + 0.943419 w_k1_INTC_s115
                                 + 0.802783 w_k1_KO_s115
                                 + 0.740217 w_k1_MAT_s115
                                 + 0.869801 w_k1_MSFT_s115
                                 + 0.872494 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.952031 w_k1_ADM_s115
                                 + 1.42781 w_k1_AXP_s115
                                 + 1.26758 w_k1_DIS_s115
                                 + 1.12352 w_k1_FDX_s115
                                 + 1.24842 w_k1_IBM_s115
                                 + 1.28761 w_k1_INTC_s115
                                 + 1.12705 w_k1_KO_s115 + 1.17063 w_k1_MAT_s115
                                 + 0.974112 w_k1_MSFT_s115
                                 + 1.09279 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.07289 w_k1_ADM_s115
                                 + 0.913419 w_k1_AXP_s115
                                 + 0.738987 w_k1_DIS_s115
                                 + 0.863588 w_k1_FDX_s115
                                 + 0.854341 w_k1_IBM_s115
                                 + 0.891461 w_k1_INTC_s115
                                 + 0.844924 w_k1_KO_s115
                                 + 0.777125 w_k1_MAT_s115
                                 + 0.758853 w_k1_MSFT_s115
                                 + 0.850837 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.33187 w_k1_ADM_s115
                                 + 1.27728 w_k1_AXP_s115
                                 + 1.40022 w_k1_DIS_s115
                                 + 1.28206 w_k1_FDX_s115 + 1.0236 w_k1_IBM_s115
                                 + 1.06129 w_k1_INTC_s115
                                 + 1.06134 w_k1_KO_s115
                                 + 0.960519 w_k1_MAT_s115
                                 + 1.10608 w_k1_MSFT_s115
                                 + 1.09403 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.0032 w_k1_ADM_s115
                                 + 0.896986 w_k1_AXP_s115
                                 + 1.21998 w_k1_DIS_s115
                                 + 1.10738 w_k1_FDX_s115
                                 + 0.899476 w_k1_IBM_s115
                                 + 1.13637 w_k1_INTC_s115
                                 + 1.08886 w_k1_KO_s115
                                 + 0.708062 w_k1_MAT_s115
                                 + 1.08648 w_k1_MSFT_s115 + 1.0758 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.62076 w_k1_ADM_s115
                                 + 1.28355 w_k1_AXP_s115
                                 + 1.81528 w_k1_DIS_s115
                                 + 1.43888 w_k1_FDX_s115
                                 + 1.30812 w_k1_IBM_s115
                                 + 1.10419 w_k1_INTC_s115
                                 + 1.47016 w_k1_KO_s115 + 0.95647 w_k1_MAT_s115
                                 + 1.30315 w_k1_MSFT_s115
                                 + 1.20135 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.752168 w_k1_ADM_s115
                                 + 0.959615 w_k1_AXP_s115
                                 + 0.881343 w_k1_DIS_s115
                                 + 0.871315 w_k1_FDX_s115
                                 + 1.21067 w_k1_IBM_s115
                                 + 0.787374 w_k1_INTC_s115
                                 + 1.23147 w_k1_KO_s115 + 1.0468 w_k1_MAT_s115
                                 + 0.792466 w_k1_MSFT_s115
                                 + 0.952843 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.844719 w_k1_ADM_s115
                                 + 0.813087 w_k1_AXP_s115
                                 + 0.873339 w_k1_DIS_s115
                                 + 0.795261 w_k1_FDX_s115
                                 + 0.767538 w_k1_IBM_s115
                                 + 0.851868 w_k1_INTC_s115
                                 + 0.903204 w_k1_KO_s115
                                 + 0.77069 w_k1_MAT_s115
                                 + 0.745944 w_k1_MSFT_s115
                                 + 0.847211 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.926148 w_k1_ADM_s116
                                 + 1.06087 w_k1_AXP_s116
                                 + 1.13765 w_k1_DIS_s116
                                 + 1.03825 w_k1_FDX_s116
                                 + 1.07633 w_k1_IBM_s116
                                 + 0.92609 w_k1_INTC_s116
                                 + 0.937005 w_k1_KO_s116
                                 + 1.90969 w_k1_MAT_s116
                                 + 0.985037 w_k1_MSFT_s116
                                 + 0.962587 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.813455 w_k1_ADM_s116
                                 + 1.04386 w_k1_AXP_s116
                                 + 0.771252 w_k1_DIS_s116
                                 + 0.921802 w_k1_FDX_s116
                                 + 1.12282 w_k1_IBM_s116
                                 + 0.903315 w_k1_INTC_s116
                                 + 0.865607 w_k1_KO_s116
                                 + 1.37209 w_k1_MAT_s116
                                 + 0.837262 w_k1_MSFT_s116
                                 + 0.945327 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.780826 w_k1_ADM_s116
                                 + 0.648186 w_k1_AXP_s116
                                 + 0.762227 w_k1_DIS_s116
                                 + 0.822251 w_k1_FDX_s116
                                 + 0.905165 w_k1_IBM_s116
                                 + 0.850009 w_k1_INTC_s116
                                 + 0.844313 w_k1_KO_s116
                                 + 1.31257 w_k1_MAT_s116
                                 + 0.854648 w_k1_MSFT_s116
                                 + 0.865231 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.15073 w_k1_ADM_s116
                                 + 1.6598 w_k1_AXP_s116 + 1.61092 w_k1_DIS_s116
                                 + 1.22334 w_k1_FDX_s116
                                 + 1.70521 w_k1_IBM_s116
                                 + 1.25359 w_k1_INTC_s116
                                 + 1.44448 w_k1_KO_s116 + 2.83728 w_k1_MAT_s116
                                 + 1.16407 w_k1_MSFT_s116
                                 + 1.20001 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.07444 w_k1_ADM_s116
                                 + 1.2082 w_k1_AXP_s116 + 1.29143 w_k1_DIS_s116
                                 + 1.15267 w_k1_FDX_s116
                                 + 1.49379 w_k1_IBM_s116
                                 + 1.14292 w_k1_INTC_s116
                                 + 1.21193 w_k1_KO_s116 + 1.77176 w_k1_MAT_s116
                                 + 1.09542 w_k1_MSFT_s116
                                 + 1.06202 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.02836 w_k1_ADM_s116
                                 + 1.21045 w_k1_AXP_s116 + 1.0243 w_k1_DIS_s116
                                 + 1.14169 w_k1_FDX_s116
                                 + 0.755483 w_k1_IBM_s116
                                 + 1.1176 w_k1_INTC_s116
                                 + 0.837375 w_k1_KO_s116
                                 + 1.04562 w_k1_MAT_s116
                                 + 1.06777 w_k1_MSFT_s116
                                 + 0.898177 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.702842 w_k1_ADM_s116
                                 + 0.640502 w_k1_AXP_s116
                                 + 0.509646 w_k1_DIS_s116
                                 + 0.651861 w_k1_FDX_s116
                                 + 0.695624 w_k1_IBM_s116
                                 + 0.816116 w_k1_INTC_s116
                                 + 0.69506 w_k1_KO_s116
                                 + 0.747168 w_k1_MAT_s116
                                 + 0.63697 w_k1_MSFT_s116
                                 + 0.729975 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.22903 w_k1_ADM_s116
                                 + 1.59751 w_k1_AXP_s116
                                 + 1.75788 w_k1_DIS_s116 + 1.5087 w_k1_FDX_s116
                                 + 1.12139 w_k1_IBM_s116
                                 + 1.07317 w_k1_INTC_s116 + 1.1568 w_k1_KO_s116
                                 + 1.61455 w_k1_MAT_s116
                                 + 1.33577 w_k1_MSFT_s116
                                 + 1.18032 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.899846 w_k1_ADM_s117
                                 + 0.612733 w_k1_AXP_s117
                                 + 0.621914 w_k1_DIS_s117
                                 + 0.848866 w_k1_FDX_s117
                                 + 0.567614 w_k1_IBM_s117
                                 + 0.917598 w_k1_INTC_s117
                                 + 0.804687 w_k1_KO_s117
                                 + 0.433562 w_k1_MAT_s117
                                 + 0.938687 w_k1_MSFT_s117
                                 + 0.92459 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.10154 w_k1_ADM_s117
                                 + 1.11668 w_k1_AXP_s117
                                 + 1.25297 w_k1_DIS_s117
                                 + 1.26941 w_k1_FDX_s117
                                 + 0.88294 w_k1_IBM_s117
                                 + 0.976737 w_k1_INTC_s117
                                 + 1.16139 w_k1_KO_s117
                                 + 0.756736 w_k1_MAT_s117
                                 + 1.04489 w_k1_MSFT_s117
                                 + 1.09976 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.961007 w_k1_ADM_s117
                                 + 0.858963 w_k1_AXP_s117
                                 + 0.926396 w_k1_DIS_s117
                                 + 1.02655 w_k1_FDX_s117
                                 + 0.644892 w_k1_IBM_s117
                                 + 0.89352 w_k1_INTC_s117 + 0.7908 w_k1_KO_s117
                                 + 0.519302 w_k1_MAT_s117
                                 + 0.953119 w_k1_MSFT_s117
                                 + 0.826925 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.31586 w_k1_ADM_s117
                                 + 1.4615 w_k1_AXP_s117 + 1.59281 w_k1_DIS_s117
                                 + 1.30266 w_k1_FDX_s117
                                 + 1.69124 w_k1_IBM_s117
                                 + 1.42453 w_k1_INTC_s117
                                 + 1.62192 w_k1_KO_s117 + 1.72604 w_k1_MAT_s117
                                 + 1.31284 w_k1_MSFT_s117 + 1.2031 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.11923 w_k1_ADM_s117
                                 + 1.04911 w_k1_AXP_s117
                                 + 0.903652 w_k1_DIS_s117
                                 + 0.926348 w_k1_FDX_s117
                                 + 1.0717 w_k1_IBM_s117
                                 + 0.854666 w_k1_INTC_s117
                                 + 1.0737 w_k1_KO_s117 + 0.864211 w_k1_MAT_s117
                                 + 0.909301 w_k1_MSFT_s117
                                 + 0.958591 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.10538 w_k1_ADM_s117
                                 + 1.09254 w_k1_AXP_s117
                                 + 1.26654 w_k1_DIS_s117
                                 + 1.12067 w_k1_FDX_s117
                                 + 1.49897 w_k1_IBM_s117
                                 + 1.13549 w_k1_INTC_s117
                                 + 1.38352 w_k1_KO_s117 + 1.28542 w_k1_MAT_s117
                                 + 1.17313 w_k1_MSFT_s117
                                 + 1.24755 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.15533 w_k1_ADM_s117
                                 + 0.912582 w_k1_AXP_s117
                                 + 0.780103 w_k1_DIS_s117
                                 + 1.00843 w_k1_FDX_s117
                                 + 0.724582 w_k1_IBM_s117
                                 + 0.86123 w_k1_INTC_s117
                                 + 0.856369 w_k1_KO_s117
                                 + 0.953111 w_k1_MAT_s117
                                 + 0.857377 w_k1_MSFT_s117
                                 + 0.951808 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.743728 w_k1_ADM_s117
                                 + 0.681765 w_k1_AXP_s117
                                 + 0.73368 w_k1_DIS_s117
                                 + 0.732291 w_k1_FDX_s117
                                 + 0.763256 w_k1_IBM_s117
                                 + 0.680216 w_k1_INTC_s117
                                 + 0.839936 w_k1_KO_s117
                                 + 0.708009 w_k1_MAT_s117
                                 + 0.805237 w_k1_MSFT_s117
                                 + 0.81978 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0610683585596261 w_k1_ADM_s000
                                 + 0.0425995025594077 w_k1_ADM_s001
                                 + 0.117757963526122 w_k1_ADM_s002
                                 + 0.0243768988994564 w_k1_ADM_s003
                                 + 0.115942355706922 w_k1_ADM_s004
                                 + 0.055906215988318 w_k1_ADM_s005
                                 + 0.0321327819083154 w_k1_ADM_s006
                                 + 0.0529130129330999 w_k1_ADM_s007
                                 + 0.0302906121663368 w_k1_ADM_s010
                                 + 0.0142749115602095 w_k1_ADM_s011
                                 + 0.00984064018561448 w_k1_ADM_s012
                                 + 0.0110708760286218 w_k1_ADM_s013
                                 + 0.0117211296222334 w_k1_ADM_s014
                                 + 0.00764981608244784 w_k1_ADM_s015
                                 + 0.0272789287385777 w_k1_ADM_s016
                                 + 0.0194546211016912 w_k1_ADM_s017
                                 + 0.0227005563000098 w_k1_ADM_s100
                                 + 0.0249722517666322 w_k1_ADM_s101
                                 + 0.0234703636243802 w_k1_ADM_s102
                                 + 0.0110332285545087 w_k1_ADM_s103
                                 + 0.027988881794651 w_k1_ADM_s104
                                 + 0.00866399945749428 w_k1_ADM_s105
                                 + 0.00812289442790931 w_k1_ADM_s106
                                 + 0.0244950719527322 w_k1_ADM_s107
                                 + 0.0164026019865379 w_k1_ADM_s110
                                 + 0.0334119984529677 w_k1_ADM_s111
                                 + 0.0162966645070772 w_k1_ADM_s112
                                 + 0.0162855519009592 w_k1_ADM_s113
                                 + 0.0152195410208912 w_k1_ADM_s114
                                 + 0.015265871966775 w_k1_ADM_s115
                                 + 0.0614657381026901 w_k1_ADM_s116
                                 + 0.0181172649722228 w_k1_ADM_s117
                                 + 0.0619898474009494 w_k1_AXP_s000
                                 + 0.0473006158020613 w_k1_AXP_s001
                                 + 0.120971778195607 w_k1_AXP_s002
                                 + 0.0286478583549252 w_k1_AXP_s003
                                 + 0.120025172934645 w_k1_AXP_s004
                                 + 0.0543673601970044 w_k1_AXP_s005
                                 + 0.0387734643531022 w_k1_AXP_s006
                                 + 0.0566449088549345 w_k1_AXP_s007
                                 + 0.0250845933568049 w_k1_AXP_s010
                                 + 0.0149971462568434 w_k1_AXP_s011
                                 + 0.00895682720333961 w_k1_AXP_s012
                                 + 0.00899700488412749 w_k1_AXP_s013
                                 + 0.0100840029347549 w_k1_AXP_s014
                                 + 0.00889427066939977 w_k1_AXP_s015
                                 + 0.0280408538553943 w_k1_AXP_s016
                                 + 0.0206696294967763 w_k1_AXP_s017
                                 + 0.0228208763141307 w_k1_AXP_s100
                                 + 0.0325216885102609 w_k1_AXP_s101
                                 + 0.0247660559151543 w_k1_AXP_s102
                                 + 0.0132232866398847 w_k1_AXP_s103
                                 + 0.0299677631625076 w_k1_AXP_s104
                                 + 0.0103662322436771 w_k1_AXP_s105
                                 + 0.00796098909879281 w_k1_AXP_s106
                                 + 0.0282041813505609 w_k1_AXP_s107
                                 + 0.0163011444587855 w_k1_AXP_s110
                                 + 0.0314773687085205 w_k1_AXP_s111
                                 + 0.0162020587305259 w_k1_AXP_s112
                                 + 0.0146051668909332 w_k1_AXP_s113
                                 + 0.0167469613902957 w_k1_AXP_s114
                                 + 0.0164497547218492 w_k1_AXP_s115
                                 + 0.070899066348092 w_k1_AXP_s116
                                 + 0.0170113317820187 w_k1_AXP_s117
                                 + 0.0660497576317347 w_k1_DIS_s000
                                 + 0.0433448628950839 w_k1_DIS_s001
                                 + 0.125162479542171 w_k1_DIS_s002
                                 + 0.0278456842055429 w_k1_DIS_s003
                                 + 0.117857660807172 w_k1_DIS_s004
                                 + 0.0541277071589022 w_k1_DIS_s005
                                 + 0.0372461239284159 w_k1_DIS_s006
                                 + 0.0596123929210359 w_k1_DIS_s007
                                 + 0.0281322761518633 w_k1_DIS_s010
                                 + 0.016360360275762 w_k1_DIS_s011
                                 + 0.00879303045184183 w_k1_DIS_s012
                                 + 0.00935726746430714 w_k1_DIS_s013
                                 + 0.0107551172023587 w_k1_DIS_s014
                                 + 0.0082374134382627 w_k1_DIS_s015
                                 + 0.0282421062727488 w_k1_DIS_s016
                                 + 0.0232940455011378 w_k1_DIS_s017
                                 + 0.0219465781610819 w_k1_DIS_s100
                                 + 0.0323219487915614 w_k1_DIS_s101
                                 + 0.0275894594832535 w_k1_DIS_s102
                                 + 0.0110925353745974 w_k1_DIS_s103
                                 + 0.0290142443409515 w_k1_DIS_s104
                                 + 0.00885739030016667 w_k1_DIS_s105
                                 + 0.00731602089333013 w_k1_DIS_s106
                                 + 0.0287763578828909 w_k1_DIS_s107
                                 + 0.0176571194293576 w_k1_DIS_s110
                                 + 0.0318699903285257 w_k1_DIS_s111
                                 + 0.0175165906193837 w_k1_DIS_s112
                                 + 0.0145622046823815 w_k1_DIS_s113
                                 + 0.0148227895120276 w_k1_DIS_s114
                                 + 0.0163596084932378 w_k1_DIS_s115
                                 + 0.069783259549685 w_k1_DIS_s116
                                 + 0.0171957618146646 w_k1_DIS_s117
                                 + 0.0651952485694424 w_k1_FDX_s000
                                 + 0.0432887882834064 w_k1_FDX_s001
                                 + 0.125322097064082 w_k1_FDX_s002
                                 + 0.0275073722141764 w_k1_FDX_s003
                                 + 0.121191473539729 w_k1_FDX_s004
                                 + 0.0515396337105507 w_k1_FDX_s005
                                 + 0.0371105089475375 w_k1_FDX_s006
                                 + 0.0563038482218175 w_k1_FDX_s007
                                 + 0.0263771646507179 w_k1_FDX_s010
                                 + 0.0151927990611275 w_k1_FDX_s011
                                 + 0.00920622129676822 w_k1_FDX_s012
                                 + 0.00912873055989197 w_k1_FDX_s013
                                 + 0.0113759921252313 w_k1_FDX_s014
                                 + 0.00767986037994634 w_k1_FDX_s015
                                 + 0.0276721261570967 w_k1_FDX_s016
                                 + 0.0211342886174172 w_k1_FDX_s017
                                 + 0.0217386462226669 w_k1_FDX_s100
                                 + 0.0321705909430784 w_k1_FDX_s101
                                 + 0.0269879409488138 w_k1_FDX_s102
                                 + 0.0113901857674095 w_k1_FDX_s103
                                 + 0.0276866376682481 w_k1_FDX_s104
                                 + 0.00916382724391906 w_k1_FDX_s105
                                 + 0.00745811848950305 w_k1_FDX_s106
                                 + 0.027000198415131 w_k1_FDX_s107
                                 + 0.0168521414941663 w_k1_FDX_s110
                                 + 0.0309282102333664 w_k1_FDX_s111
                                 + 0.0172199453654209 w_k1_FDX_s112
                                 + 0.0141725872293625 w_k1_FDX_s113
                                 + 0.0165988269575999 w_k1_FDX_s114
                                 + 0.0156889149207975 w_k1_FDX_s115
                                 + 0.0684844737189021 w_k1_FDX_s116
                                 + 0.0173565866038921 w_k1_FDX_s117
                                 + 0.0686376627439001 w_k1_IBM_s000
                                 + 0.0467302003122795 w_k1_IBM_s001
                                 + 0.125037615775739 w_k1_IBM_s002
                                 + 0.0280669145664953 w_k1_IBM_s003
                                 + 0.127877318662657 w_k1_IBM_s004
                                 + 0.0528972027279668 w_k1_IBM_s005
                                 + 0.0342877243724751 w_k1_IBM_s006
                                 + 0.0604790216331147 w_k1_IBM_s007
                                 + 0.0268250812531984 w_k1_IBM_s010
                                 + 0.0156527382982725 w_k1_IBM_s011
                                 + 0.00920495462243628 w_k1_IBM_s012
                                 + 0.00932519793743551 w_k1_IBM_s013
                                 + 0.0109722418627636 w_k1_IBM_s014
                                 + 0.00800531480962546 w_k1_IBM_s015
                                 + 0.028171740281362 w_k1_IBM_s016
                                 + 0.023085121045338 w_k1_IBM_s017
                                 + 0.0217690022725535 w_k1_IBM_s100
                                 + 0.0317884107269533 w_k1_IBM_s101
                                 + 0.0265964203013235 w_k1_IBM_s102
                                 + 0.0111987948987447 w_k1_IBM_s103
                                 + 0.0275297054075997 w_k1_IBM_s104
                                 + 0.00941471459490658 w_k1_IBM_s105
                                 + 0.00775966058497171 w_k1_IBM_s106
                                 + 0.0283632277575124 w_k1_IBM_s107
                                 + 0.0163460191049516 w_k1_IBM_s110
                                 + 0.0315827556734684 w_k1_IBM_s111
                                 + 0.017581670839644 w_k1_IBM_s112
                                 + 0.0135106968563387 w_k1_IBM_s113
                                 + 0.0161320939694853 w_k1_IBM_s114
                                 + 0.0155038148501442 w_k1_IBM_s115
                                 + 0.0665566414924817 w_k1_IBM_s116
                                 + 0.0173173521432532 w_k1_IBM_s117
                                 + 0.0605150282529917 w_k1_INTC_s000
                                 + 0.0419977972120307 w_k1_INTC_s001
                                 + 0.11609284806301 w_k1_INTC_s002
                                 + 0.0304932971371002 w_k1_INTC_s003
                                 + 0.126147633269682 w_k1_INTC_s004
                                 + 0.0551667089673938 w_k1_INTC_s005
                                 + 0.0359950542177711 w_k1_INTC_s006
                                 + 0.066763887664668 w_k1_INTC_s007
                                 + 0.02758559232915 w_k1_INTC_s010
                                 + 0.0161275823013739 w_k1_INTC_s011
                                 + 0.00912976575286862 w_k1_INTC_s012
                                 + 0.00939565729045544 w_k1_INTC_s013
                                 + 0.0116852329866387 w_k1_INTC_s014
                                 + 0.0063728390087105 w_k1_INTC_s015
                                 + 0.0264815519142005 w_k1_INTC_s016
                                 + 0.0234688059886522 w_k1_INTC_s017
                                 + 0.0206316931412124 w_k1_INTC_s100
                                 + 0.0366857016994828 w_k1_INTC_s101
                                 + 0.0265743638189935 w_k1_INTC_s102
                                 + 0.0107807550752928 w_k1_INTC_s103
                                 + 0.0268591152892734 w_k1_INTC_s104
                                 + 0.00941624465681317 w_k1_INTC_s105
                                 + 0.00669040368896108 w_k1_INTC_s106
                                 + 0.0238354293291624 w_k1_INTC_s107
                                 + 0.0152668474029192 w_k1_INTC_s110
                                 + 0.0283334129444934 w_k1_INTC_s111
                                 + 0.019296836435141 w_k1_INTC_s112
                                 + 0.0131840901840485 w_k1_INTC_s113
                                 + 0.0147098343513138 w_k1_INTC_s114
                                 + 0.0151727643553488 w_k1_INTC_s115
                                 + 0.0637267759364699 w_k1_INTC_s116
                                 + 0.0160888511367406 w_k1_INTC_s117
                                 + 0.0694463126923999 w_k1_KO_s000
                                 + 0.0455333812997226 w_k1_KO_s001
                                 + 0.132771816102201 w_k1_KO_s002
                                 + 0.0261018074961297 w_k1_KO_s003
                                 + 0.1201749727375 w_k1_KO_s004
                                 + 0.054752162523586 w_k1_KO_s005
                                 + 0.0345902672648448 w_k1_KO_s006
                                 + 0.0619234930775293 w_k1_KO_s007
                                 + 0.0252504455467459 w_k1_KO_s010
                                 + 0.0162651767976749 w_k1_KO_s011
                                 + 0.00936521306735684 w_k1_KO_s012
                                 + 0.0103665714172056 w_k1_KO_s013
                                 + 0.0104628224264494 w_k1_KO_s014
                                 + 0.00703275518630551 w_k1_KO_s015
                                 + 0.0268274811084531 w_k1_KO_s016
                                 + 0.0233361525241242 w_k1_KO_s017
                                 + 0.0217095114324993 w_k1_KO_s100
                                 + 0.0320247086600683 w_k1_KO_s101
                                 + 0.0246294969211655 w_k1_KO_s102
                                 + 0.0113751427248077 w_k1_KO_s103
                                 + 0.0259630066715423 w_k1_KO_s104
                                 + 0.00877936682441861 w_k1_KO_s105
                                 + 0.00744432185090116 w_k1_KO_s106
                                 + 0.0286826304405611 w_k1_KO_s107
                                 + 0.0172620518433704 w_k1_KO_s110
                                 + 0.0318527704648547 w_k1_KO_s111
                                 + 0.016927895465426 w_k1_KO_s112
                                 + 0.0141873943535413 w_k1_KO_s113
                                 + 0.0161013354412175 w_k1_KO_s114
                                 + 0.0161548277928875 w_k1_KO_s115
                                 + 0.0612471492693741 w_k1_KO_s116
                                 + 0.0183803540924655 w_k1_KO_s117
                                 + 0.07825633249104 w_k1_MAT_s000
                                 + 0.0596852709831176 w_k1_MAT_s001
                                 + 0.121179397794471 w_k1_MAT_s002
                                 + 0.0233103138649561 w_k1_MAT_s003
                                 + 0.123288619332792 w_k1_MAT_s004
                                 + 0.0588694261268241 w_k1_MAT_s005
                                 + 0.0416148545030523 w_k1_MAT_s006
                                 + 0.0492554038225868 w_k1_MAT_s007
                                 + 0.0260299990639904 w_k1_MAT_s010
                                 + 0.0172667912128854 w_k1_MAT_s011
                                 + 0.0101060146050355 w_k1_MAT_s012
                                 + 0.0100818706867651 w_k1_MAT_s013
                                 + 0.011344265618979 w_k1_MAT_s014
                                 + 0.00799179834318752 w_k1_MAT_s015
                                 + 0.0331368580545588 w_k1_MAT_s016
                                 + 0.0200341517021664 w_k1_MAT_s017
                                 + 0.0239099249567184 w_k1_MAT_s100
                                 + 0.0290759621357737 w_k1_MAT_s101
                                 + 0.0291229353884208 w_k1_MAT_s102
                                 + 0.0153770065150871 w_k1_MAT_s103
                                 + 0.0313582816286435 w_k1_MAT_s104
                                 + 0.00833192948655717 w_k1_MAT_s105
                                 + 0.00838103367372142 w_k1_MAT_s106
                                 + 0.0291085774552838 w_k1_MAT_s107
                                 + 0.014728423148847 w_k1_MAT_s110
                                 + 0.0417027722020085 w_k1_MAT_s111
                                 + 0.0165339019133415 w_k1_MAT_s112
                                 + 0.013610354541707 w_k1_MAT_s113
                                 + 0.0136143042829046 w_k1_MAT_s114
                                 + 0.0141012698411817 w_k1_MAT_s115
                                 + 0.0953580327868684 w_k1_MAT_s116
                                 + 0.0158540997095444 w_k1_MAT_s117
                                 + 0.0639427494899629 w_k1_MSFT_s000
                                 + 0.0438227138814944 w_k1_MSFT_s001
                                 + 0.121546696246128 w_k1_MSFT_s002
                                 + 0.0269418617904064 w_k1_MSFT_s003
                                 + 0.119414814143927 w_k1_MSFT_s004
                                 + 0.0522411450848189 w_k1_MSFT_s005
                                 + 0.0351042411724364 w_k1_MSFT_s006
                                 + 0.0562407941285947 w_k1_MSFT_s007
                                 + 0.0254852807301317 w_k1_MSFT_s010
                                 + 0.0170066146575764 w_k1_MSFT_s011
                                 + 0.00881419370390798 w_k1_MSFT_s012
                                 + 0.00938976746425694 w_k1_MSFT_s013
                                 + 0.0102205337862902 w_k1_MSFT_s014
                                 + 0.0063386451699235 w_k1_MSFT_s015
                                 + 0.0256517677757519 w_k1_MSFT_s016
                                 + 0.0230895994111404 w_k1_MSFT_s017
                                 + 0.0212383724494664 w_k1_MSFT_s100
                                 + 0.0344815603973518 w_k1_MSFT_s101
                                 + 0.0267742484474408 w_k1_MSFT_s102
                                 + 0.0115726766017103 w_k1_MSFT_s103
                                 + 0.0274513671442782 w_k1_MSFT_s104
                                 + 0.00881456487560709 w_k1_MSFT_s105
                                 + 0.00671462802493363 w_k1_MSFT_s106
                                 + 0.0247057982778533 w_k1_MSFT_s107
                                 + 0.0153846778609286 w_k1_MSFT_s110
                                 + 0.0289955262790096 w_k1_MSFT_s111
                                 + 0.0177932178318555 w_k1_MSFT_s112
                                 + 0.0152397209805014 w_k1_MSFT_s113
                                 + 0.0159614080613989 w_k1_MSFT_s114
                                 + 0.0139804712789483 w_k1_MSFT_s115
                                 + 0.0645159677427643 w_k1_MSFT_s116
                                 + 0.0166275404026613 w_k1_MSFT_s117
                                 + 0.0609865191649308 w_k1_PG_s000
                                 + 0.0443368253040193 w_k1_PG_s001
                                 + 0.128428709689219 w_k1_PG_s002
                                 + 0.0257833873338984 w_k1_PG_s003
                                 + 0.115288335538924 w_k1_PG_s004
                                 + 0.0516795079351597 w_k1_PG_s005
                                 + 0.0378344445587172 w_k1_PG_s006
                                 + 0.0596290534234439 w_k1_PG_s007
                                 + 0.027624744748972 w_k1_PG_s010
                                 + 0.0160972659656826 w_k1_PG_s011
                                 + 0.00874256605080523 w_k1_PG_s012
                                 + 0.00981380916456013 w_k1_PG_s013
                                 + 0.0108349094889021 w_k1_PG_s014
                                 + 0.00746514346610413 w_k1_PG_s015
                                 + 0.0265902235905514 w_k1_PG_s016
                                 + 0.0214553127313592 w_k1_PG_s017
                                 + 0.0227951948360223 w_k1_PG_s100
                                 + 0.0306599808059253 w_k1_PG_s101
                                 + 0.0248666261172678 w_k1_PG_s102
                                 + 0.0118781016463151 w_k1_PG_s103
                                 + 0.0275427957820279 w_k1_PG_s104
                                 + 0.00919516109050505 w_k1_PG_s105
                                 + 0.00737734498031073 w_k1_PG_s106
                                 + 0.0274424019604884 w_k1_PG_s107
                                 + 0.015798526720228 w_k1_PG_s110
                                 + 0.0336150097464031 w_k1_PG_s111
                                 + 0.0170540419376472 w_k1_PG_s112
                                 + 0.014148953968953 w_k1_PG_s113
                                 + 0.0146630102729731 w_k1_PG_s114
                                 + 0.0150476777820927 w_k1_PG_s115
                                 + 0.0620513417621422 w_k1_PG_s116
                                 + 0.0170500098269389 w_k1_PG_s117 >= 1.08
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
