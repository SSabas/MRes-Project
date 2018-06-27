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
 w_k1_ADM_s0_return_constraint:  1.21508 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.88438 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.06576 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.880143 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.813521 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.18089 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.609205 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.20099 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.816323 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.6766 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.932489 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.967763 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.16593 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.691038 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.729579 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.945709 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.04193 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.02961 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.829042 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.914895 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.17739 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.5937 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.33341 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.995648 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.875234 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.967971 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.03998 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.53909 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.26066 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.16506 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.00853 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.831501 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.938218 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.816427 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.984978 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.00079 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.22487 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.769489 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.32071 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.915855 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.964023 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.97693 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.781016 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.15462 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.904686 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.95798 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.901764 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.620015 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.952829 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.1827 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.4079 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.795606 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.64607 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.713187 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.986188 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.960801 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.897082 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.1599 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.18122 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.72345 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.963355 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.37767 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.751033 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.659983 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.883144 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.38832 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.34612 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.18039 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.37681 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.638889 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.891481 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.7394 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.00116 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.05692 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.22474 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.681803 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30892 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.886913 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.03166 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.1408 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.689093 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.18496 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.609174 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 2.06574 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.904952 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.639951 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.04326 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.19922 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.32527 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.69331 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.575877 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.649209 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.909775 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.04487 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.573481 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.0097 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.2428 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.77521 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.31069 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.27581 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.849802 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.678733 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.00856 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.51179 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.70044 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.999463 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.08628 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.58365 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.842119 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.769391 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.36454 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.08509 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.32997 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.729841 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17831 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.917144 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.06576 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.951129 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.777903 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.1209 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.726381 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.65364 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.893093 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.717452 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.07565 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.10477 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.2907 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.83541 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.748713 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.724243 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.993763 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.00899 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.73644 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.92668 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.29911 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.45966 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.0945 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.02754 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.804403 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.875991 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.1318 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.37153 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.23013 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.01031 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.14002 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.794463 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.0119 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.790901 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.04355 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.91124 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.30603 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.767086 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.22139 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.926178 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.870193 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.34834 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.91229 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.10719 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.12708 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.42786 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.11828 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.854573 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.733496 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.27916 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.01018 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.643593 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.655173 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.868369 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.03914 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.769537 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.06169 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.51925 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.889612 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.43075 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.06268 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.80395 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.05675 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.567876 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.72543 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.906056 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.4473 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.013 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.15964 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.571742 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.715346 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.959944 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.12061 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.57521 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.774735 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.72927 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18102 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.896723 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.921349 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.959764 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.944418 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.04768 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.774231 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.53619 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.04353 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.917352 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.22565 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.13424 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.30672 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.869749 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.850381 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.775523 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.09141 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.05405 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.739704 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.19395 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.27426 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.37467 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.831137 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.05749 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.777237 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.793325 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00558 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.954664 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.00364 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.01998 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.24173 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.00259 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.960779 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.855459 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.31701 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.00712 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.21908 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.854754 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.22814 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.919739 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.953591 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.19148 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.808397 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.09415 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.830271 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.56681 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.04089 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.782616 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.886492 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.31297 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.19249 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.68073 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.612951 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.764137 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.985246 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.945658 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.803572 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.16452 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.04829 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.38035 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.23683 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.37201 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.00961 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.685747 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.867815 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.12266 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.50434 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.06744 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.06867 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.585791 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.844541 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.931726 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.23415 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.17561 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.07145 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.823764 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.25132 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.90493 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.914208 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.4211 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.82466 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.25157 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.1999 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 2.01032 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.21 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.748837 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.648734 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.52593 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.13443 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.675063 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.530256 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.814617 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.05043 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.59778 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.948912 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.42995 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.855376 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.50017 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.15855 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.92425 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.751528 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.630046 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.797357 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.888307 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.46156 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.29588 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.299 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.377001 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.710187 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.770063 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.974998 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.43989 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.854194 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.718086 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16644 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.916305 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.03011 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.0159 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.851796 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.08233 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.759535 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.55094 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.974174 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.778147 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.10012 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.13295 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.12137 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.849 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.759361 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.732968 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.935788 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.03241 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.715829 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.01909 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.24323 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.31065 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.07982 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.08221 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.822153 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.857962 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.0108 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.25592 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.11356 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.95647 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.08943 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.827537 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.0179 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.802468 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.12268 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.921599 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.26841 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.866742 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12084 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.90512 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.959936 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.0593 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.881708 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.1184 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.914168 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.55366 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.0141 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.799128 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05873 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.18297 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.23346 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.822873 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.835946 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.82972 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.974721 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.951782 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.844625 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.08097 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.00495 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.22784 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.13027 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.16582 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.899383 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.847558 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.02453 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.16375 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.26315 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.04249 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07643 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.740673 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.898401 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.869255 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.09491 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.06333 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03014 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.876074 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.208999746981095 z_k1_s0000
                                 - 0.184999729099124 z_k1_s0001
                                 - 0.6759998748431 z_k1_s0002
                                 - 0.900998627613647 z_k1_s0003
                                 - 0.243999552967624 z_k1_s0004
                                 - 0.628999682616137 z_k1_s0005
                                 - 0.429000791264539 z_k1_s0006
                                 - 0.500001001373473 z_k1_s0007
                                 - 0.102000098643438 z_k1_s0010
                                 - 0.102000098643438 z_k1_s0011
                                 - 0.111000063553142 z_k1_s0012
                                 - 0.464999924168484 z_k1_s0013
                                 - 0.115999977880739 z_k1_s0014
                                 - 0.196000096122661 z_k1_s0015
                                 - 0.170000660739185 z_k1_s0016
                                 - 0.226999594733772 z_k1_s0017
                                 - 0.708998587577317 z_k1_s0020
                                 - 0.188999799744155 z_k1_s0021
                                 - 0.301000160547689 z_k1_s0022
                                 - 0.307998812561582 z_k1_s0023
                                 - 0.262999778650283 z_k1_s0024
                                 - 0.15699992311438 z_k1_s0025
                                 - 0.149999950101617 z_k1_s0026
                                 - 0.56499914274333 z_k1_s0027
                                 - 0.28100004529201 z_k1_s0030
                                 - 0.248999341966978 z_k1_s0031
                                 - 0.263000636234128 z_k1_s0032
                                 - 0.205999876717638 z_k1_s0033
                                 - 0.496999718986243 z_k1_s0034
                                 - 0.305000009035814 z_k1_s0035
                                 - 0.186999924212126 z_k1_s0036
                                 - 0.267001006024742 z_k1_s0037
                                 - 0.325999686747009 z_k1_s0040
                                 - 0.139999982171141 z_k1_s0041
                                 - 0.193999418002711 z_k1_s0042
                                 - 0.17399932843538 z_k1_s0043
                                 - 0.18300073894106 z_k1_s0044
                                 - 0.178000313549115 z_k1_s0045
                                 - 0.0860000627394355 z_k1_s0046
                                 - 0.331000112138954 z_k1_s0047
                                 - 0.318000063687574 z_k1_s0050
                                 - 1.43299918254183 z_k1_s0051
                                 - 0.393999843700963 z_k1_s0052
                                 - 0.409000235308857 z_k1_s0053
                                 - 1.09199989145545 z_k1_s0054
                                 - 0.671998682434157 z_k1_s0055
                                 - 0.477000095531361 z_k1_s0056
                                 - 0.62500186366264 z_k1_s0057
                                 - 0.267000121816458 z_k1_s0060
                                 - 0.405999253279203 z_k1_s0061
                                 - 0.267000121816458 z_k1_s0062
                                 - 0.233000002753715 z_k1_s0063
                                 - 0.393000856695956 z_k1_s0064
                                 - 1.03800164715131 z_k1_s0065
                                 - 0.436999565130474 z_k1_s0066
                                 - 0.79900010997516 z_k1_s0067
                                 - 0.522999969218481 z_k1_s0070
                                 - 0.335999475942951 z_k1_s0071
                                 - 0.387999343367525 z_k1_s0072
                                 - 0.371998441838151 z_k1_s0073
                                 - 0.943999145305558 z_k1_s0074
                                 - 0.31699944123793 z_k1_s0075
                                 - 0.413998556480778 z_k1_s0076
                                 - 0.308999711072277 z_k1_s0077
                                 - 0.457000983188967 z_k1_s0100
                                 - 0.221000482662508 z_k1_s0101
                                 - 0.131000210878582 z_k1_s0102
                                 - 0.176000429070654 z_k1_s0103
                                 - 0.140000221596952 z_k1_s0104
                                 - 0.0980001715778884 z_k1_s0105
                                 - 0.0960001143292889 z_k1_s0106
                                 - 0.326999731033269 z_k1_s0107
                                 - 0.0529999230593824 z_k1_s0110
                                 - 0.0559999605834638 z_k1_s0111
                                 - 0.0879999073740399 z_k1_s0112
                                 - 0.144999904332178 z_k1_s0113
                                 - 0.0519999582846492 z_k1_s0114
                                 - 0.177000065922577 z_k1_s0115
                                 - 0.0800000459762925 z_k1_s0116
                                 - 0.0649999299558263 z_k1_s0117
                                 - 0.172000131950115 z_k1_s0120
                                 - 0.337999718107478 z_k1_s0121
                                 - 0.206001008474456 z_k1_s0122
                                 - 0.470000224741517 z_k1_s0123
                                 - 0.0960001070791529 z_k1_s0124
                                 - 0.146000043049477 z_k1_s0125
                                 - 0.1510000186765 z_k1_s0126
                                 - 0.217999584258537 z_k1_s0127
                                 - 0.111999484325577 z_k1_s0130
                                 - 0.14499938667141 z_k1_s0131
                                 - 0.0659998806915351 z_k1_s0132
                                 - 0.0909999610138671 z_k1_s0133
                                 - 0.0509999388979501 z_k1_s0134
                                 - 0.170999701246232 z_k1_s0135
                                 - 0.0569999004154107 z_k1_s0136
                                 - 0.0669998869444231 z_k1_s0137
                                 - 0.0609999258619707 z_k1_s0140
                                 - 0.123999519077296 z_k1_s0141
                                 - 0.100000153642156 z_k1_s0142
                                 - 0.162999441657946 z_k1_s0143
                                 - 0.0479999516684207 z_k1_s0144
                                 - 0.16999995599213 z_k1_s0145
                                 - 0.0499999242213103 z_k1_s0146
                                 - 0.0469999653919759 z_k1_s0147
                                 - 0.197999840616947 z_k1_s0150
                                 - 0.152000211200768 z_k1_s0151
                                 - 0.246999967951383 z_k1_s0152
                                 - 0.0830000165767708 z_k1_s0153
                                 - 0.157999706037826 z_k1_s0154
                                 - 0.176999357188057 z_k1_s0155
                                 - 0.29699976092542 z_k1_s0156
                                 - 0.188999847861631 z_k1_s0157
                                 - 0.307999383838363 z_k1_s0160
                                 - 0.0699997160638708 z_k1_s0161
                                 - 0.188998902403846 z_k1_s0162
                                 - 0.239998286450676 z_k1_s0163
                                 - 0.0869997026120055 z_k1_s0164
                                 - 0.116999560368037 z_k1_s0165
                                 - 0.183999837474 z_k1_s0166
                                 - 0.243998689589701 z_k1_s0167
                                 - 0.234998823121639 z_k1_s0170
                                 - 0.121999702922065 z_k1_s0171
                                 - 0.131999669148483 z_k1_s0172
                                 - 0.158999692919572 z_k1_s0173
                                 - 0.223999090192097 z_k1_s0174
                                 - 0.262999188394647 z_k1_s0175
                                 - 0.31499931933138 z_k1_s0176
                                 - 0.465998962629978 z_k1_s0177
                                 - 0.378001914628029 z_k1_s1000
                                 - 0.172000825703541 z_k1_s1001
                                 - 0.235000428138105 z_k1_s1002
                                 - 0.10900057826588 z_k1_s1003
                                 - 0.124000555274454 z_k1_s1004
                                 - 0.443003176671719 z_k1_s1005
                                 - 0.396001801037905 z_k1_s1006
                                 - 0.293001256575661 z_k1_s1007
                                 - 0.32899853026108 z_k1_s1010
                                 - 0.212999658092251 z_k1_s1011
                                 - 0.407999129650331 z_k1_s1012
                                 - 0.178999771737983 z_k1_s1013
                                 - 0.270999464226002 z_k1_s1014
                                 - 0.321999663806741 z_k1_s1015
                                 - 0.388998329709788 z_k1_s1016
                                 - 0.355999796860566 z_k1_s1017
                                 - 0.453000608098165 z_k1_s1020
                                 - 0.214000488287439 z_k1_s1021
                                 - 1.05900180400517 z_k1_s1022
                                 - 0.307000627837426 z_k1_s1023
                                 - 0.880002132453295 z_k1_s1024
                                 - 0.459001763426517 z_k1_s1025
                                 - 0.462999572305508 z_k1_s1026
                                 - 0.607000648126752 z_k1_s1027
                                 - 0.304999067434745 z_k1_s1030
                                 - 0.405000341215709 z_k1_s1031
                                 - 0.311999638359132 z_k1_s1032
                                 - 0.399999073127362 z_k1_s1033
                                 - 0.196000027592127 z_k1_s1034
                                 - 0.695998266801679 z_k1_s1035
                                 - 0.43599856816907 z_k1_s1036
                                 - 0.260999952250279 z_k1_s1037
                                 - 0.430000173156682 z_k1_s1040
                                 - 0.567003023293875 z_k1_s1041
                                 - 0.215000737980253 z_k1_s1042
                                 - 0.853001032768651 z_k1_s1043
                                 - 0.296000937235838 z_k1_s1044
                                 - 0.186000650557836 z_k1_s1045
                                 - 0.350001504340836 z_k1_s1046
                                 - 0.360000523690427 z_k1_s1047
                                 - 0.0960000691894036 z_k1_s1050
                                 - 0.247000180177946 z_k1_s1051
                                 - 0.400999302174902 z_k1_s1052
                                 - 0.463000698629335 z_k1_s1053
                                 - 0.156999998706556 z_k1_s1054
                                 - 0.168000172906489 z_k1_s1055
                                 - 0.138000112416026 z_k1_s1056
                                 - 0.402999748447881 z_k1_s1057
                                 - 0.266000204133952 z_k1_s1060
                                 - 0.118000102736126 z_k1_s1061
                                 - 0.174000316238173 z_k1_s1062
                                 - 0.2750006627329 z_k1_s1063
                                 - 0.170000262416561 z_k1_s1064
                                 - 0.156999749996001 z_k1_s1065
                                 - 0.0950001307621809 z_k1_s1066
                                 - 0.0950001307621809 z_k1_s1067
                                 - 0.343000059345673 z_k1_s1070
                                 - 0.0829999841646716 z_k1_s1071
                                 - 0.268999722215435 z_k1_s1072
                                 - 0.329999844606449 z_k1_s1073
                                 - 0.404000181736687 z_k1_s1074
                                 - 0.196000581912266 z_k1_s1075
                                 - 0.101000045434172 z_k1_s1076
                                 - 0.191999630607158 z_k1_s1077
                                 - 1.20400055930834 z_k1_s1100
                                 - 0.901000731768386 z_k1_s1101
                                 - 0.854999644365327 z_k1_s1102
                                 - 0.908000465764134 z_k1_s1103
                                 - 0.476000436526737 z_k1_s1104
                                 - 0.294000075031053 z_k1_s1105
                                 - 1.28599933354586 z_k1_s1106
                                 - 0.692001112746002 z_k1_s1107
                                 - 0.309000338585298 z_k1_s1110
                                 - 1.0810015804623 z_k1_s1111
                                 - 0.6470032332026 z_k1_s1112
                                 - 0.513002145963859 z_k1_s1113
                                 - 1.14500194756122 z_k1_s1114
                                 - 0.458000779024489 z_k1_s1115
                                 - 0.579000931407752 z_k1_s1116
                                 - 0.366000633670238 z_k1_s1117
                                 - 1.0299984431603 z_k1_s1120
                                 - 0.715999607523959 z_k1_s1121
                                 - 0.292999520829151 z_k1_s1122
                                 - 1.08599910877562 z_k1_s1123
                                 - 0.385999513370603 z_k1_s1124
                                 - 0.908000999349303 z_k1_s1125
                                 - 1.3109988896988 z_k1_s1126
                                 - 0.501999289633391 z_k1_s1127
                                 - 0.493999159347451 z_k1_s1130
                                 - 0.233999601796161 z_k1_s1131
                                 - 0.279999523516774 z_k1_s1132
                                 - 0.273999533727129 z_k1_s1133
                                 - 0.275999530323677 z_k1_s1134
                                 - 0.216999863825336 z_k1_s1135
                                 - 0.262999552446113 z_k1_s1136
                                 - 0.292999501394339 z_k1_s1137
                                 - 0.567997885374796 z_k1_s1140
                                 - 0.342999958314086 z_k1_s1141
                                 - 0.547999713466348 z_k1_s1142
                                 - 0.212000030819756 z_k1_s1143
                                 - 0.626000274105415 z_k1_s1144
                                 - 0.408001524112754 z_k1_s1145
                                 - 0.659998264949223 z_k1_s1146
                                 - 0.296999768526873 z_k1_s1147
                                 - 0.681999530200072 z_k1_s1150
                                 - 0.382999214955332 z_k1_s1151
                                 - 1.71899649335381 z_k1_s1152
                                 - 1.50899463958133 z_k1_s1153
                                 - 1.85700244760019 z_k1_s1154
                                 - 0.927999333735606 z_k1_s1155
                                 - 1.38400099104362 z_k1_s1156
                                 - 0.746999358496964 z_k1_s1157
                                 - 1.14599534161364 z_k1_s1160
                                 - 0.643999489526411 z_k1_s1161
                                 - 1.29599647271352 z_k1_s1162
                                 - 1.49899631180336 z_k1_s1163
                                 - 0.53799944854856 z_k1_s1164
                                 - 1.42899636728962 z_k1_s1165
                                 - 1.63399995479118 z_k1_s1166
                                 - 0.563999177939577 z_k1_s1167
                                 - 0.180999907658842 z_k1_s1170
                                 - 0.264999896047453 z_k1_s1171
                                 - 0.116000027050909 z_k1_s1172
                                 - 0.374998615200113 z_k1_s1173
                                 - 0.131999820182219 z_k1_s1174
                                 - 0.437999794040721 z_k1_s1175
                                 - 0.195999897506814 z_k1_s1176
                                 - 0.558998611308485 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.59673 w_k1_ADM_s000
                                 + 1.5535 w_k1_AXP_s000 + 1.64377 w_k1_DIS_s000
                                 + 1.41833 w_k1_FDX_s000
                                 + 0.946113 w_k1_IBM_s000
                                 + 1.42668 w_k1_INTC_s000
                                 + 1.17185 w_k1_KO_s000 + 1.0816 w_k1_MAT_s000
                                 + 1.23208 w_k1_MSFT_s000
                                 + 1.26743 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.12799 w_k1_ADM_s000
                                 + 1.07897 w_k1_AXP_s000
                                 + 1.30936 w_k1_DIS_s000
                                 + 1.09205 w_k1_FDX_s000
                                 + 1.89051 w_k1_IBM_s000
                                 + 1.20728 w_k1_INTC_s000
                                 + 1.49122 w_k1_KO_s000 + 1.91836 w_k1_MAT_s000
                                 + 1.08933 w_k1_MSFT_s000
                                 + 1.19797 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.3728 w_k1_ADM_s000
                                 + 1.02261 w_k1_AXP_s000
                                 + 0.983932 w_k1_DIS_s000
                                 + 1.06858 w_k1_FDX_s000
                                 + 0.702442 w_k1_IBM_s000
                                 + 1.10542 w_k1_INTC_s000
                                 + 0.87142 w_k1_KO_s000 + 0.67275 w_k1_MAT_s000
                                 + 0.955765 w_k1_MSFT_s000
                                 + 0.934732 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.22354 w_k1_ADM_s000
                                 + 0.786662 w_k1_AXP_s000
                                 + 0.92669 w_k1_DIS_s000
                                 + 0.933688 w_k1_FDX_s000
                                 + 1.00336 w_k1_IBM_s000
                                 + 0.973933 w_k1_INTC_s000
                                 + 0.95431 w_k1_KO_s000 + 1.01871 w_k1_MAT_s000
                                 + 1.01468 w_k1_MSFT_s000
                                 + 0.960791 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.99527 w_k1_ADM_s000
                                 + 1.76435 w_k1_AXP_s000
                                 + 1.74523 w_k1_DIS_s000
                                 + 1.41964 w_k1_FDX_s000
                                 + 1.61503 w_k1_IBM_s000
                                 + 1.36789 w_k1_INTC_s000
                                 + 1.51357 w_k1_KO_s000 + 1.58665 w_k1_MAT_s000
                                 + 1.23224 w_k1_MSFT_s000
                                 + 1.32278 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.789928 w_k1_ADM_s000
                                 + 0.574763 w_k1_AXP_s000
                                 + 0.599957 w_k1_DIS_s000
                                 + 0.768928 w_k1_FDX_s000
                                 + 0.669312 w_k1_IBM_s000
                                 + 1.01461 w_k1_INTC_s000
                                 + 0.705638 w_k1_KO_s000
                                 + 0.418274 w_k1_MAT_s000
                                 + 0.784897 w_k1_MSFT_s000
                                 + 0.766371 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.18121 w_k1_ADM_s000
                                 + 1.14038 w_k1_AXP_s000
                                 + 0.966051 w_k1_DIS_s000
                                 + 0.983523 w_k1_FDX_s000
                                 + 1.22337 w_k1_IBM_s000
                                 + 0.999449 w_k1_INTC_s000
                                 + 1.00577 w_k1_KO_s000 + 1.2272 w_k1_MAT_s000
                                 + 0.865095 w_k1_MSFT_s000
                                 + 0.987154 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.21289 w_k1_ADM_s000
                                 + 1.1882 w_k1_AXP_s000 + 1.37324 w_k1_DIS_s000
                                 + 1.18333 w_k1_FDX_s000
                                 + 1.35339 w_k1_IBM_s000
                                 + 1.17453 w_k1_INTC_s000
                                 + 1.40256 w_k1_KO_s000 + 1.5093 w_k1_MAT_s000
                                 + 1.16479 w_k1_MSFT_s000
                                 + 1.20333 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.06607 w_k1_ADM_s001
                                 + 0.781232 w_k1_AXP_s001
                                 + 1.19375 w_k1_DIS_s001
                                 + 1.07986 w_k1_FDX_s001
                                 + 0.679748 w_k1_IBM_s001
                                 + 1.1288 w_k1_INTC_s001
                                 + 0.812285 w_k1_KO_s001
                                 + 0.478052 w_k1_MAT_s001
                                 + 0.97946 w_k1_MSFT_s001
                                 + 0.900554 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.3038 w_k1_ADM_s001
                                 + 0.786619 w_k1_AXP_s001
                                 + 1.10405 w_k1_DIS_s001
                                 + 0.907196 w_k1_FDX_s001
                                 + 1.45039 w_k1_IBM_s001
                                 + 0.965955 w_k1_INTC_s001 + 1.377 w_k1_KO_s001
                                 + 1.2042 w_k1_MAT_s001
                                 + 0.902753 w_k1_MSFT_s001
                                 + 1.05806 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.40418 w_k1_ADM_s001
                                 + 1.43026 w_k1_AXP_s001
                                 + 1.65398 w_k1_DIS_s001
                                 + 1.37722 w_k1_FDX_s001
                                 + 0.922403 w_k1_IBM_s001
                                 + 1.18991 w_k1_INTC_s001
                                 + 1.15682 w_k1_KO_s001
                                 + 0.996079 w_k1_MAT_s001
                                 + 1.24193 w_k1_MSFT_s001
                                 + 1.16596 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.954118 w_k1_ADM_s001
                                 + 0.742657 w_k1_AXP_s001
                                 + 0.877021 w_k1_DIS_s001
                                 + 0.881922 w_k1_FDX_s001
                                 + 0.706338 w_k1_IBM_s001
                                 + 0.907842 w_k1_INTC_s001
                                 + 0.752028 w_k1_KO_s001
                                 + 0.637281 w_k1_MAT_s001
                                 + 0.911963 w_k1_MSFT_s001
                                 + 0.754536 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.08538 w_k1_ADM_s001
                                 + 0.97212 w_k1_AXP_s001
                                 + 1.20592 w_k1_DIS_s001 + 1.0444 w_k1_FDX_s001
                                 + 1.11304 w_k1_IBM_s001
                                 + 0.959882 w_k1_INTC_s001
                                 + 1.10475 w_k1_KO_s001
                                 + 0.960154 w_k1_MAT_s001
                                 + 1.0012 w_k1_MSFT_s001
                                 + 0.999626 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.00773 w_k1_ADM_s001
                                 + 1.24101 w_k1_AXP_s001
                                 + 1.14625 w_k1_DIS_s001
                                 + 1.18977 w_k1_FDX_s001
                                 + 0.890329 w_k1_IBM_s001
                                 + 1.26601 w_k1_INTC_s001
                                 + 1.01467 w_k1_KO_s001
                                 + 0.879926 w_k1_MAT_s001
                                 + 1.11897 w_k1_MSFT_s001
                                 + 0.903539 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.09388 w_k1_ADM_s001
                                 + 0.936156 w_k1_AXP_s001
                                 + 0.796802 w_k1_DIS_s001
                                 + 0.793703 w_k1_FDX_s001
                                 + 1.23963 w_k1_IBM_s001
                                 + 1.06042 w_k1_INTC_s001
                                 + 0.995407 w_k1_KO_s001
                                 + 1.16726 w_k1_MAT_s001
                                 + 0.814757 w_k1_MSFT_s001
                                 + 0.793262 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.3643 w_k1_ADM_s001
                                 + 1.5172 w_k1_AXP_s001 + 1.40926 w_k1_DIS_s001
                                 + 1.13236 w_k1_FDX_s001
                                 + 1.64031 w_k1_IBM_s001
                                 + 1.31599 w_k1_INTC_s001
                                 + 1.39372 w_k1_KO_s001 + 1.59097 w_k1_MAT_s001
                                 + 1.10866 w_k1_MSFT_s001
                                 + 1.21413 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16083 w_k1_ADM_s002
                                 + 1.12857 w_k1_AXP_s002
                                 + 1.07986 w_k1_DIS_s002
                                 + 1.10834 w_k1_FDX_s002
                                 + 0.906817 w_k1_IBM_s002
                                 + 1.16129 w_k1_INTC_s002
                                 + 0.93097 w_k1_KO_s002
                                 + 0.759345 w_k1_MAT_s002
                                 + 1.01609 w_k1_MSFT_s002
                                 + 1.04647 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.800824 w_k1_ADM_s002
                                 + 0.684492 w_k1_AXP_s002
                                 + 0.609897 w_k1_DIS_s002
                                 + 0.766063 w_k1_FDX_s002
                                 + 0.805102 w_k1_IBM_s002
                                 + 0.769206 w_k1_INTC_s002
                                 + 0.760879 w_k1_KO_s002
                                 + 0.520862 w_k1_MAT_s002
                                 + 0.675393 w_k1_MSFT_s002
                                 + 0.779329 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.971046 w_k1_ADM_s002
                                 + 0.917678 w_k1_AXP_s002
                                 + 0.898992 w_k1_DIS_s002
                                 + 0.869368 w_k1_FDX_s002
                                 + 1.09937 w_k1_IBM_s002
                                 + 0.879168 w_k1_INTC_s002
                                 + 0.920688 w_k1_KO_s002
                                 + 0.784569 w_k1_MAT_s002
                                 + 0.799899 w_k1_MSFT_s002
                                 + 0.935971 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.814107 w_k1_ADM_s002
                                 + 0.800584 w_k1_AXP_s002
                                 + 1.07992 w_k1_DIS_s002
                                 + 0.980667 w_k1_FDX_s002
                                 + 0.713649 w_k1_IBM_s002
                                 + 1.01565 w_k1_INTC_s002
                                 + 0.852598 w_k1_KO_s002
                                 + 0.552014 w_k1_MAT_s002
                                 + 1.11226 w_k1_MSFT_s002
                                 + 0.975137 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.04016 w_k1_ADM_s002
                                 + 1.04857 w_k1_AXP_s002
                                 + 0.883571 w_k1_DIS_s002
                                 + 0.948683 w_k1_FDX_s002
                                 + 1.24812 w_k1_IBM_s002
                                 + 0.893397 w_k1_INTC_s002
                                 + 1.03525 w_k1_KO_s002 + 1.13973 w_k1_MAT_s002
                                 + 0.947811 w_k1_MSFT_s002
                                 + 1.07991 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.998411 w_k1_ADM_s002
                                 + 1.19419 w_k1_AXP_s002
                                 + 0.895077 w_k1_DIS_s002
                                 + 0.945034 w_k1_FDX_s002
                                 + 0.966503 w_k1_IBM_s002
                                 + 0.765276 w_k1_INTC_s002
                                 + 0.835884 w_k1_KO_s002
                                 + 1.04067 w_k1_MAT_s002
                                 + 0.785069 w_k1_MSFT_s002
                                 + 0.973269 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.944844 w_k1_ADM_s002
                                 + 0.87004 w_k1_AXP_s002
                                 + 0.761777 w_k1_DIS_s002
                                 + 0.860255 w_k1_FDX_s002
                                 + 0.625898 w_k1_IBM_s002
                                 + 0.951082 w_k1_INTC_s002
                                 + 0.698782 w_k1_KO_s002
                                 + 0.674095 w_k1_MAT_s002
                                 + 0.860335 w_k1_MSFT_s002
                                 + 0.952332 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.21966 w_k1_ADM_s002
                                 + 1.32587 w_k1_AXP_s002
                                 + 1.39045 w_k1_DIS_s002 + 1.1498 w_k1_FDX_s002
                                 + 1.49129 w_k1_IBM_s002
                                 + 1.06001 w_k1_INTC_s002
                                 + 1.26261 w_k1_KO_s002 + 1.24189 w_k1_MAT_s002
                                 + 1.11154 w_k1_MSFT_s002
                                 + 1.16641 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.710247 w_k1_ADM_s003
                                 + 0.682877 w_k1_AXP_s003
                                 + 0.661764 w_k1_DIS_s003
                                 + 0.69417 w_k1_FDX_s003
                                 + 0.720675 w_k1_IBM_s003
                                 + 0.897515 w_k1_INTC_s003
                                 + 0.722865 w_k1_KO_s003
                                 + 0.660921 w_k1_MAT_s003
                                 + 0.820244 w_k1_MSFT_s003
                                 + 0.874936 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.737112 w_k1_ADM_s003
                                 + 0.779848 w_k1_AXP_s003
                                 + 0.730227 w_k1_DIS_s003
                                 + 0.768934 w_k1_FDX_s003
                                 + 0.974525 w_k1_IBM_s003
                                 + 0.623826 w_k1_INTC_s003
                                 + 0.884335 w_k1_KO_s003
                                 + 0.835941 w_k1_MAT_s003
                                 + 0.765687 w_k1_MSFT_s003
                                 + 1.03369 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.13828 w_k1_ADM_s003
                                 + 1.59389 w_k1_AXP_s003
                                 + 1.56919 w_k1_DIS_s003
                                 + 1.17549 w_k1_FDX_s003
                                 + 1.64858 w_k1_IBM_s003
                                 + 1.39558 w_k1_INTC_s003
                                 + 1.53641 w_k1_KO_s003 + 1.62728 w_k1_MAT_s003
                                 + 1.1775 w_k1_MSFT_s003 + 1.4076 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.873097 w_k1_ADM_s003
                                 + 1.07607 w_k1_AXP_s003
                                 + 0.854971 w_k1_DIS_s003
                                 + 0.982655 w_k1_FDX_s003
                                 + 0.867228 w_k1_IBM_s003
                                 + 0.909174 w_k1_INTC_s003
                                 + 0.855828 w_k1_KO_s003
                                 + 0.975662 w_k1_MAT_s003
                                 + 0.851602 w_k1_MSFT_s003
                                 + 0.9652 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.17307 w_k1_ADM_s003
                                 + 1.27138 w_k1_AXP_s003
                                 + 1.04384 w_k1_DIS_s003
                                 + 1.04204 w_k1_FDX_s003 + 1.1265 w_k1_IBM_s003
                                 + 0.928524 w_k1_INTC_s003
                                 + 0.963982 w_k1_KO_s003
                                 + 1.39198 w_k1_MAT_s003
                                 + 0.982503 w_k1_MSFT_s003
                                 + 1.11937 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.2157 w_k1_ADM_s003
                                 + 1.35996 w_k1_AXP_s003 + 1.5942 w_k1_DIS_s003
                                 + 1.23173 w_k1_FDX_s003
                                 + 0.981041 w_k1_IBM_s003
                                 + 1.08651 w_k1_INTC_s003
                                 + 1.13191 w_k1_KO_s003
                                 + 0.764673 w_k1_MAT_s003
                                 + 1.37774 w_k1_MSFT_s003
                                 + 1.11024 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.09905 w_k1_ADM_s003
                                 + 0.982082 w_k1_AXP_s003
                                 + 0.975605 w_k1_DIS_s003
                                 + 0.899796 w_k1_FDX_s003
                                 + 1.36198 w_k1_IBM_s003
                                 + 0.951087 w_k1_INTC_s003
                                 + 1.17837 w_k1_KO_s003 + 1.12851 w_k1_MAT_s003
                                 + 0.991107 w_k1_MSFT_s003
                                 + 1.13748 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.02894 w_k1_ADM_s003
                                 + 0.825317 w_k1_AXP_s003
                                 + 1.00347 w_k1_DIS_s003
                                 + 0.939272 w_k1_FDX_s003
                                 + 0.65448 w_k1_IBM_s003
                                 + 0.83448 w_k1_INTC_s003
                                 + 0.894446 w_k1_KO_s003
                                 + 0.773007 w_k1_MAT_s003
                                 + 1.00389 w_k1_MSFT_s003
                                 + 0.99191 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.20834 w_k1_ADM_s004
                                 + 0.914481 w_k1_AXP_s004
                                 + 1.08156 w_k1_DIS_s004
                                 + 1.11328 w_k1_FDX_s004
                                 + 0.748828 w_k1_IBM_s004
                                 + 0.977685 w_k1_INTC_s004
                                 + 0.898138 w_k1_KO_s004
                                 + 0.813906 w_k1_MAT_s004
                                 + 1.07116 w_k1_MSFT_s004
                                 + 1.06781 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.877753 w_k1_ADM_s004
                                 + 0.800547 w_k1_AXP_s004
                                 + 0.761187 w_k1_DIS_s004
                                 + 0.763168 w_k1_FDX_s004
                                 + 1.25694 w_k1_IBM_s004
                                 + 0.781423 w_k1_INTC_s004
                                 + 1.04248 w_k1_KO_s004 + 1.38192 w_k1_MAT_s004
                                 + 0.780112 w_k1_MSFT_s004
                                 + 0.900428 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.14354 w_k1_ADM_s004
                                 + 1.28171 w_k1_AXP_s004 + 1.2582 w_k1_DIS_s004
                                 + 1.12639 w_k1_FDX_s004
                                 + 1.18709 w_k1_IBM_s004
                                 + 0.813657 w_k1_INTC_s004
                                 + 1.1727 w_k1_KO_s004 + 2.26523 w_k1_MAT_s004
                                 + 1.08029 w_k1_MSFT_s004
                                 + 1.15712 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.17487 w_k1_ADM_s004
                                 + 1.44124 w_k1_AXP_s004
                                 + 1.49984 w_k1_DIS_s004
                                 + 1.32183 w_k1_FDX_s004
                                 + 1.06561 w_k1_IBM_s004
                                 + 1.16584 w_k1_INTC_s004
                                 + 1.05945 w_k1_KO_s004 + 1.32781 w_k1_MAT_s004
                                 + 1.04131 w_k1_MSFT_s004
                                 + 1.03813 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.857362 w_k1_ADM_s004
                                 + 0.705956 w_k1_AXP_s004
                                 + 0.632392 w_k1_DIS_s004
                                 + 0.747219 w_k1_FDX_s004
                                 + 0.694054 w_k1_IBM_s004
                                 + 0.6818 w_k1_INTC_s004
                                 + 0.677991 w_k1_KO_s004
                                 + 0.940109 w_k1_MAT_s004
                                 + 0.742072 w_k1_MSFT_s004
                                 + 0.800592 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.893469 w_k1_ADM_s004
                                 + 0.818919 w_k1_AXP_s004
                                 + 1.02843 w_k1_DIS_s004
                                 + 0.889493 w_k1_FDX_s004
                                 + 1.01032 w_k1_IBM_s004
                                 + 0.870629 w_k1_INTC_s004
                                 + 1.00942 w_k1_KO_s004 + 1.30441 w_k1_MAT_s004
                                 + 1.00828 w_k1_MSFT_s004
                                 + 0.871864 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.952895 w_k1_ADM_s004
                                 + 1.13718 w_k1_AXP_s004
                                 + 1.37253 w_k1_DIS_s004
                                 + 1.04104 w_k1_FDX_s004
                                 + 1.48753 w_k1_IBM_s004
                                 + 0.922919 w_k1_INTC_s004
                                 + 1.38221 w_k1_KO_s004 + 2.25103 w_k1_MAT_s004
                                 + 1.24066 w_k1_MSFT_s004
                                 + 1.12803 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.90637 w_k1_ADM_s004
                                 + 1.29012 w_k1_AXP_s004
                                 + 0.795482 w_k1_DIS_s004
                                 + 0.952668 w_k1_FDX_s004
                                 + 1.0623 w_k1_IBM_s004
                                 + 0.846543 w_k1_INTC_s004
                                 + 0.870078 w_k1_KO_s004
                                 + 1.56776 w_k1_MAT_s004
                                 + 0.939243 w_k1_MSFT_s004
                                 + 0.898633 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.7655 w_k1_ADM_s005
                                 + 0.777485 w_k1_AXP_s005
                                 + 0.755455 w_k1_DIS_s005
                                 + 0.770697 w_k1_FDX_s005
                                 + 1.12265 w_k1_IBM_s005
                                 + 0.951112 w_k1_INTC_s005
                                 + 0.868594 w_k1_KO_s005
                                 + 0.90325 w_k1_MAT_s005
                                 + 0.764029 w_k1_MSFT_s005
                                 + 0.932718 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.932877 w_k1_ADM_s005
                                 + 0.732215 w_k1_AXP_s005
                                 + 0.716385 w_k1_DIS_s005
                                 + 0.810304 w_k1_FDX_s005
                                 + 0.671396 w_k1_IBM_s005
                                 + 0.863284 w_k1_INTC_s005
                                 + 0.723894 w_k1_KO_s005
                                 + 0.721819 w_k1_MAT_s005
                                 + 0.867756 w_k1_MSFT_s005
                                 + 0.849361 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06349 w_k1_ADM_s005
                                 + 1.2326 w_k1_AXP_s005
                                 + 0.756311 w_k1_DIS_s005
                                 + 0.925858 w_k1_FDX_s005
                                 + 1.10796 w_k1_IBM_s005
                                 + 1.02702 w_k1_INTC_s005
                                 + 0.837783 w_k1_KO_s005 + 1.2635 w_k1_MAT_s005
                                 + 0.83736 w_k1_MSFT_s005
                                 + 0.957144 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.85662 w_k1_ADM_s005
                                 + 1.00638 w_k1_AXP_s005
                                 + 0.992729 w_k1_DIS_s005
                                 + 0.944713 w_k1_FDX_s005
                                 + 0.947096 w_k1_IBM_s005
                                 + 1.00118 w_k1_INTC_s005
                                 + 0.83856 w_k1_KO_s005 + 0.74969 w_k1_MAT_s005
                                 + 0.972237 w_k1_MSFT_s005
                                 + 0.901244 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.06449 w_k1_ADM_s005
                                 + 1.0911 w_k1_AXP_s005 + 1.40277 w_k1_DIS_s005
                                 + 1.15601 w_k1_FDX_s005
                                 + 0.972385 w_k1_IBM_s005
                                 + 1.08955 w_k1_INTC_s005
                                 + 1.00374 w_k1_KO_s005
                                 + 0.967278 w_k1_MAT_s005
                                 + 1.18634 w_k1_MSFT_s005 + 1.0612 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.06114 w_k1_ADM_s005
                                 + 0.776934 w_k1_AXP_s005
                                 + 1.07842 w_k1_DIS_s005
                                 + 1.00114 w_k1_FDX_s005
                                 + 0.956564 w_k1_IBM_s005
                                 + 1.02428 w_k1_INTC_s005
                                 + 0.794797 w_k1_KO_s005
                                 + 0.707495 w_k1_MAT_s005
                                 + 1.06701 w_k1_MSFT_s005
                                 + 0.961379 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.16785 w_k1_ADM_s005
                                 + 1.11109 w_k1_AXP_s005
                                 + 1.02142 w_k1_DIS_s005
                                 + 0.896114 w_k1_FDX_s005
                                 + 1.69278 w_k1_IBM_s005
                                 + 0.941686 w_k1_INTC_s005
                                 + 1.24934 w_k1_KO_s005 + 1.6558 w_k1_MAT_s005
                                 + 1.05993 w_k1_MSFT_s005
                                 + 1.06845 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.25884 w_k1_ADM_s005
                                 + 1.59061 w_k1_AXP_s005 + 1.3558 w_k1_DIS_s005
                                 + 1.2504 w_k1_FDX_s005 + 1.49481 w_k1_IBM_s005
                                 + 1.18957 w_k1_INTC_s005
                                 + 1.19703 w_k1_KO_s005 + 1.85558 w_k1_MAT_s005
                                 + 1.18884 w_k1_MSFT_s005
                                 + 1.14792 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.876572 w_k1_ADM_s006
                                 + 1.00267 w_k1_AXP_s006
                                 + 0.934654 w_k1_DIS_s006
                                 + 0.887357 w_k1_FDX_s006
                                 + 1.36176 w_k1_IBM_s006
                                 + 1.03766 w_k1_INTC_s006
                                 + 1.06316 w_k1_KO_s006 + 1.23531 w_k1_MAT_s006
                                 + 1.05544 w_k1_MSFT_s006
                                 + 0.88285 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.78042 w_k1_ADM_s006
                                 + 0.644122 w_k1_AXP_s006
                                 + 0.627095 w_k1_DIS_s006
                                 + 0.743519 w_k1_FDX_s006
                                 + 0.813962 w_k1_IBM_s006
                                 + 0.725477 w_k1_INTC_s006
                                 + 0.670648 w_k1_KO_s006
                                 + 0.803081 w_k1_MAT_s006
                                 + 0.871956 w_k1_MSFT_s006
                                 + 0.860474 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.92056 w_k1_ADM_s006
                                 + 0.70624 w_k1_AXP_s006
                                 + 1.16307 w_k1_DIS_s006
                                 + 0.938571 w_k1_FDX_s006
                                 + 0.95154 w_k1_IBM_s006
                                 + 1.07597 w_k1_INTC_s006
                                 + 0.922566 w_k1_KO_s006
                                 + 0.752075 w_k1_MAT_s006
                                 + 1.279 w_k1_MSFT_s006 + 0.885582 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.887948 w_k1_ADM_s006
                                 + 1.15117 w_k1_AXP_s006
                                 + 0.83178 w_k1_DIS_s006
                                 + 0.876514 w_k1_FDX_s006
                                 + 1.19159 w_k1_IBM_s006
                                 + 0.779061 w_k1_INTC_s006
                                 + 0.881662 w_k1_KO_s006
                                 + 1.44274 w_k1_MAT_s006
                                 + 0.741816 w_k1_MSFT_s006
                                 + 1.04644 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.827469 w_k1_ADM_s006
                                 + 0.797138 w_k1_AXP_s006
                                 + 0.808781 w_k1_DIS_s006
                                 + 0.95385 w_k1_FDX_s006
                                 + 0.644257 w_k1_IBM_s006
                                 + 0.890485 w_k1_INTC_s006
                                 + 0.686784 w_k1_KO_s006
                                 + 0.769831 w_k1_MAT_s006
                                 + 0.92683 w_k1_MSFT_s006
                                 + 0.788209 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.14797 w_k1_ADM_s006
                                 + 1.24686 w_k1_AXP_s006
                                 + 1.38709 w_k1_DIS_s006
                                 + 1.14415 w_k1_FDX_s006
                                 + 1.16489 w_k1_IBM_s006
                                 + 1.17906 w_k1_INTC_s006
                                 + 1.10939 w_k1_KO_s006 + 1.35184 w_k1_MAT_s006
                                 + 1.3239 w_k1_MSFT_s006 + 1.06223 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.00541 w_k1_ADM_s006
                                 + 1.43046 w_k1_AXP_s006
                                 + 1.44442 w_k1_DIS_s006
                                 + 1.22376 w_k1_FDX_s006
                                 + 1.66078 w_k1_IBM_s006
                                 + 1.48842 w_k1_INTC_s006
                                 + 1.43133 w_k1_KO_s006 + 2.05693 w_k1_MAT_s006
                                 + 1.30621 w_k1_MSFT_s006
                                 + 1.06581 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.04522 w_k1_ADM_s006
                                 + 0.902487 w_k1_AXP_s006
                                 + 0.965558 w_k1_DIS_s006
                                 + 1.04771 w_k1_FDX_s006
                                 + 0.811197 w_k1_IBM_s006
                                 + 1.04631 w_k1_INTC_s006
                                 + 0.888046 w_k1_KO_s006
                                 + 0.859966 w_k1_MAT_s006
                                 + 1.15091 w_k1_MSFT_s006
                                 + 0.910802 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.25381 w_k1_ADM_s007
                                 + 0.95415 w_k1_AXP_s007
                                 + 0.95197 w_k1_DIS_s007
                                 + 1.11544 w_k1_FDX_s007
                                 + 0.705832 w_k1_IBM_s007
                                 + 1.09714 w_k1_INTC_s007
                                 + 0.91027 w_k1_KO_s007
                                 + 0.864511 w_k1_MAT_s007
                                 + 1.0482 w_k1_MSFT_s007 + 1.10229 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.50184 w_k1_ADM_s007
                                 + 1.32792 w_k1_AXP_s007
                                 + 1.49176 w_k1_DIS_s007
                                 + 1.33319 w_k1_FDX_s007
                                 + 1.32077 w_k1_IBM_s007
                                 + 1.22264 w_k1_INTC_s007
                                 + 1.27763 w_k1_KO_s007 + 1.43377 w_k1_MAT_s007
                                 + 1.37581 w_k1_MSFT_s007
                                 + 1.27505 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.893589 w_k1_ADM_s007
                                 + 1.15008 w_k1_AXP_s007
                                 + 0.75551 w_k1_DIS_s007
                                 + 0.87889 w_k1_FDX_s007 + 1.017 w_k1_IBM_s007
                                 + 0.926132 w_k1_INTC_s007
                                 + 0.999538 w_k1_KO_s007
                                 + 1.06873 w_k1_MAT_s007
                                 + 0.844465 w_k1_MSFT_s007
                                 + 0.977443 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.24462 w_k1_ADM_s007
                                 + 1.32902 w_k1_AXP_s007
                                 + 1.30541 w_k1_DIS_s007
                                 + 1.10146 w_k1_FDX_s007
                                 + 1.26188 w_k1_IBM_s007
                                 + 1.19382 w_k1_INTC_s007
                                 + 1.17759 w_k1_KO_s007 + 1.8064 w_k1_MAT_s007
                                 + 1.16563 w_k1_MSFT_s007
                                 + 1.23066 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.0646 w_k1_ADM_s007
                                 + 0.761001 w_k1_AXP_s007
                                 + 0.825846 w_k1_DIS_s007
                                 + 0.807659 w_k1_FDX_s007
                                 + 1.04425 w_k1_IBM_s007
                                 + 0.770872 w_k1_INTC_s007
                                 + 0.969685 w_k1_KO_s007
                                 + 0.916002 w_k1_MAT_s007
                                 + 0.805733 w_k1_MSFT_s007
                                 + 1.02838 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.61876 w_k1_ADM_s007
                                 + 0.926271 w_k1_AXP_s007
                                 + 1.65495 w_k1_DIS_s007
                                 + 1.31105 w_k1_FDX_s007
                                 + 1.17598 w_k1_IBM_s007
                                 + 1.31242 w_k1_INTC_s007
                                 + 1.54022 w_k1_KO_s007 + 1.09938 w_k1_MAT_s007
                                 + 1.39323 w_k1_MSFT_s007
                                 + 1.31673 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.06896 w_k1_ADM_s007
                                 + 0.716934 w_k1_AXP_s007
                                 + 0.815401 w_k1_DIS_s007
                                 + 0.870445 w_k1_FDX_s007
                                 + 0.641317 w_k1_IBM_s007
                                 + 0.919921 w_k1_INTC_s007
                                 + 0.876113 w_k1_KO_s007
                                 + 0.60301 w_k1_MAT_s007
                                 + 0.90584 w_k1_MSFT_s007
                                 + 0.99487 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.15276 w_k1_ADM_s007
                                 + 0.785533 w_k1_AXP_s007
                                 + 1.12177 w_k1_DIS_s007
                                 + 1.03968 w_k1_FDX_s007
                                 + 0.879931 w_k1_IBM_s007
                                 + 1.21406 w_k1_INTC_s007
                                 + 0.990722 w_k1_KO_s007
                                 + 0.757728 w_k1_MAT_s007
                                 + 1.13752 w_k1_MSFT_s007
                                 + 1.06042 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.08578 w_k1_ADM_s010
                                 + 0.836461 w_k1_AXP_s010
                                 + 0.698872 w_k1_DIS_s010
                                 + 0.802093 w_k1_FDX_s010
                                 + 0.856469 w_k1_IBM_s010
                                 + 0.743919 w_k1_INTC_s010
                                 + 0.880263 w_k1_KO_s010
                                 + 0.784172 w_k1_MAT_s010
                                 + 0.805712 w_k1_MSFT_s010
                                 + 0.813039 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.16074 w_k1_ADM_s010
                                 + 1.22329 w_k1_AXP_s010
                                 + 1.15426 w_k1_DIS_s010
                                 + 0.957511 w_k1_FDX_s010
                                 + 1.41367 w_k1_IBM_s010
                                 + 0.852186 w_k1_INTC_s010
                                 + 1.17418 w_k1_KO_s010 + 1.59052 w_k1_MAT_s010
                                 + 0.871851 w_k1_MSFT_s010
                                 + 0.887876 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.41514 w_k1_ADM_s010
                                 + 1.72093 w_k1_AXP_s010
                                 + 1.32894 w_k1_DIS_s010
                                 + 1.35434 w_k1_FDX_s010
                                 + 1.36847 w_k1_IBM_s010
                                 + 1.24331 w_k1_INTC_s010
                                 + 1.42093 w_k1_KO_s010 + 1.59249 w_k1_MAT_s010
                                 + 1.18484 w_k1_MSFT_s010 + 1.0632 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.45467 w_k1_ADM_s010
                                 + 1.2107 w_k1_AXP_s010 + 1.34808 w_k1_DIS_s010
                                 + 1.33735 w_k1_FDX_s010
                                 + 0.784397 w_k1_IBM_s010
                                 + 0.969101 w_k1_INTC_s010
                                 + 1.10057 w_k1_KO_s010
                                 + 0.979776 w_k1_MAT_s010
                                 + 1.28197 w_k1_MSFT_s010
                                 + 0.976542 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.48426 w_k1_ADM_s010
                                 + 0.950216 w_k1_AXP_s010
                                 + 1.18765 w_k1_DIS_s010
                                 + 1.11026 w_k1_FDX_s010
                                 + 0.783989 w_k1_IBM_s010
                                 + 0.775115 w_k1_INTC_s010
                                 + 0.939984 w_k1_KO_s010
                                 + 0.967418 w_k1_MAT_s010
                                 + 0.99168 w_k1_MSFT_s010
                                 + 0.934304 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.36828 w_k1_ADM_s010
                                 + 1.69114 w_k1_AXP_s010
                                 + 1.88284 w_k1_DIS_s010
                                 + 1.56653 w_k1_FDX_s010
                                 + 1.11112 w_k1_IBM_s010
                                 + 1.43746 w_k1_INTC_s010
                                 + 1.44013 w_k1_KO_s010
                                 + 0.774576 w_k1_MAT_s010
                                 + 1.32109 w_k1_MSFT_s010
                                 + 1.10644 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.2067 w_k1_ADM_s010
                                 + 1.23748 w_k1_AXP_s010
                                 + 1.52547 w_k1_DIS_s010
                                 + 1.29523 w_k1_FDX_s010
                                 + 1.24922 w_k1_IBM_s010
                                 + 1.08208 w_k1_INTC_s010
                                 + 1.38355 w_k1_KO_s010 + 1.38882 w_k1_MAT_s010
                                 + 1.32263 w_k1_MSFT_s010
                                 + 1.06387 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.26376 w_k1_ADM_s010
                                 + 0.937473 w_k1_AXP_s010
                                 + 0.938343 w_k1_DIS_s010
                                 + 1.02506 w_k1_FDX_s010
                                 + 1.02627 w_k1_IBM_s010
                                 + 1.02233 w_k1_INTC_s010 + 1.0279 w_k1_KO_s010
                                 + 0.978706 w_k1_MAT_s010
                                 + 1.18186 w_k1_MSFT_s010
                                 + 0.907915 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.906675 w_k1_ADM_s011
                                 + 1.31581 w_k1_AXP_s011
                                 + 1.65312 w_k1_DIS_s011
                                 + 1.24326 w_k1_FDX_s011
                                 + 1.49161 w_k1_IBM_s011
                                 + 1.2006 w_k1_INTC_s011 + 1.43776 w_k1_KO_s011
                                 + 1.68484 w_k1_MAT_s011
                                 + 1.32022 w_k1_MSFT_s011
                                 + 1.37775 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.996796 w_k1_ADM_s011
                                 + 1.43417 w_k1_AXP_s011
                                 + 1.51406 w_k1_DIS_s011
                                 + 1.42368 w_k1_FDX_s011
                                 + 0.94476 w_k1_IBM_s011
                                 + 1.29429 w_k1_INTC_s011 + 1.2319 w_k1_KO_s011
                                 + 1.2559 w_k1_MAT_s011
                                 + 1.25299 w_k1_MSFT_s011
                                 + 1.07109 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.720953 w_k1_ADM_s011
                                 + 0.871845 w_k1_AXP_s011
                                 + 0.922186 w_k1_DIS_s011
                                 + 0.929992 w_k1_FDX_s011
                                 + 1.01295 w_k1_IBM_s011
                                 + 1.02426 w_k1_INTC_s011
                                 + 0.880601 w_k1_KO_s011
                                 + 0.725818 w_k1_MAT_s011
                                 + 0.936187 w_k1_MSFT_s011
                                 + 0.810333 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.771587 w_k1_ADM_s011
                                 + 0.858281 w_k1_AXP_s011
                                 + 0.960923 w_k1_DIS_s011
                                 + 0.882369 w_k1_FDX_s011
                                 + 1.26694 w_k1_IBM_s011
                                 + 0.903042 w_k1_INTC_s011
                                 + 1.06333 w_k1_KO_s011 + 1.06539 w_k1_MAT_s011
                                 + 1.01654 w_k1_MSFT_s011 + 1.0332 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.558759 w_k1_ADM_s011
                                 + 0.641671 w_k1_AXP_s011
                                 + 0.516802 w_k1_DIS_s011
                                 + 0.764053 w_k1_FDX_s011
                                 + 0.854491 w_k1_IBM_s011
                                 + 0.807734 w_k1_INTC_s011
                                 + 0.647923 w_k1_KO_s011
                                 + 0.804791 w_k1_MAT_s011
                                 + 0.860565 w_k1_MSFT_s011
                                 + 0.812826 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.878018 w_k1_ADM_s011
                                 + 0.765783 w_k1_AXP_s011
                                 + 0.815788 w_k1_DIS_s011
                                 + 0.96013 w_k1_FDX_s011
                                 + 0.831025 w_k1_IBM_s011
                                 + 0.840585 w_k1_INTC_s011
                                 + 0.947478 w_k1_KO_s011
                                 + 0.868134 w_k1_MAT_s011
                                 + 1.07566 w_k1_MSFT_s011
                                 + 0.922671 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.05976 w_k1_ADM_s011
                                 + 1.24942 w_k1_AXP_s011
                                 + 1.26509 w_k1_DIS_s011
                                 + 1.21895 w_k1_FDX_s011
                                 + 0.913728 w_k1_IBM_s011
                                 + 1.13538 w_k1_INTC_s011
                                 + 1.17958 w_k1_KO_s011
                                 + 0.944408 w_k1_MAT_s011
                                 + 1.27382 w_k1_MSFT_s011
                                 + 1.14963 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.00107 w_k1_ADM_s011
                                 + 1.27016 w_k1_AXP_s011
                                 + 1.29696 w_k1_DIS_s011
                                 + 1.32317 w_k1_FDX_s011
                                 + 1.12046 w_k1_IBM_s011
                                 + 1.2433 w_k1_INTC_s011 + 1.20449 w_k1_KO_s011
                                 + 1.00537 w_k1_MAT_s011
                                 + 1.15552 w_k1_MSFT_s011
                                 + 1.15565 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.051 w_k1_ADM_s012
                                 + 0.890619 w_k1_AXP_s012
                                 + 1.17766 w_k1_DIS_s012
                                 + 0.991722 w_k1_FDX_s012
                                 + 1.03349 w_k1_IBM_s012
                                 + 0.830684 w_k1_INTC_s012
                                 + 0.982528 w_k1_KO_s012
                                 + 1.01523 w_k1_MAT_s012
                                 + 1.0776 w_k1_MSFT_s012 + 1.04662 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.936136 w_k1_ADM_s012
                                 + 0.949525 w_k1_AXP_s012
                                 + 0.940862 w_k1_DIS_s012
                                 + 0.894002 w_k1_FDX_s012
                                 + 1.13926 w_k1_IBM_s012
                                 + 0.818199 w_k1_INTC_s012
                                 + 1.00169 w_k1_KO_s012 + 1.04996 w_k1_MAT_s012
                                 + 0.974507 w_k1_MSFT_s012
                                 + 0.85448 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.19383 w_k1_ADM_s012
                                 + 1.24697 w_k1_AXP_s012
                                 + 1.35665 w_k1_DIS_s012
                                 + 1.23517 w_k1_FDX_s012
                                 + 1.01923 w_k1_IBM_s012
                                 + 1.0817 w_k1_INTC_s012 + 1.115 w_k1_KO_s012
                                 + 0.998928 w_k1_MAT_s012
                                 + 1.29887 w_k1_MSFT_s012
                                 + 1.08537 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.933555 w_k1_ADM_s012
                                 + 0.78972 w_k1_AXP_s012
                                 + 0.775411 w_k1_DIS_s012
                                 + 0.784929 w_k1_FDX_s012
                                 + 0.725122 w_k1_IBM_s012
                                 + 0.783307 w_k1_INTC_s012
                                 + 0.794588 w_k1_KO_s012
                                 + 0.72602 w_k1_MAT_s012
                                 + 0.910325 w_k1_MSFT_s012
                                 + 0.869641 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.24161 w_k1_ADM_s012
                                 + 1.0038 w_k1_AXP_s012 + 1.40612 w_k1_DIS_s012
                                 + 1.13609 w_k1_FDX_s012
                                 + 0.878939 w_k1_IBM_s012
                                 + 0.994785 w_k1_INTC_s012
                                 + 1.15696 w_k1_KO_s012
                                 + 0.946668 w_k1_MAT_s012
                                 + 1.32324 w_k1_MSFT_s012
                                 + 1.05903 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.37783 w_k1_ADM_s012
                                 + 1.38496 w_k1_AXP_s012 + 1.688 w_k1_DIS_s012
                                 + 1.28509 w_k1_FDX_s012
                                 + 1.37945 w_k1_IBM_s012
                                 + 0.872768 w_k1_INTC_s012
                                 + 1.22743 w_k1_KO_s012 + 1.56706 w_k1_MAT_s012
                                 + 1.16856 w_k1_MSFT_s012 + 1.196 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.94213 w_k1_ADM_s012
                                 + 0.990585 w_k1_AXP_s012
                                 + 1.04697 w_k1_DIS_s012
                                 + 0.999242 w_k1_FDX_s012
                                 + 0.825426 w_k1_IBM_s012
                                 + 0.884675 w_k1_INTC_s012
                                 + 0.887058 w_k1_KO_s012
                                 + 0.610616 w_k1_MAT_s012
                                 + 1.16146 w_k1_MSFT_s012
                                 + 0.910387 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.971033 w_k1_ADM_s012
                                 + 1.08124 w_k1_AXP_s012
                                 + 1.23414 w_k1_DIS_s012
                                 + 1.06774 w_k1_FDX_s012
                                 + 1.43729 w_k1_IBM_s012
                                 + 1.13699 w_k1_INTC_s012
                                 + 1.28902 w_k1_KO_s012 + 1.26859 w_k1_MAT_s012
                                 + 1.16524 w_k1_MSFT_s012
                                 + 1.04329 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.996246 w_k1_ADM_s013
                                 + 0.818556 w_k1_AXP_s013
                                 + 0.942957 w_k1_DIS_s013
                                 + 0.842458 w_k1_FDX_s013
                                 + 1.22406 w_k1_IBM_s013
                                 + 0.888721 w_k1_INTC_s013
                                 + 1.05767 w_k1_KO_s013 + 1.45995 w_k1_MAT_s013
                                 + 0.885918 w_k1_MSFT_s013
                                 + 0.975046 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.917952 w_k1_ADM_s013
                                 + 0.954115 w_k1_AXP_s013
                                 + 0.931684 w_k1_DIS_s013
                                 + 0.941442 w_k1_FDX_s013
                                 + 0.975427 w_k1_IBM_s013
                                 + 0.845305 w_k1_INTC_s013
                                 + 1.00994 w_k1_KO_s013 + 1.08028 w_k1_MAT_s013
                                 + 0.917507 w_k1_MSFT_s013
                                 + 0.932746 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.29946 w_k1_ADM_s013
                                 + 1.31556 w_k1_AXP_s013
                                 + 1.23863 w_k1_DIS_s013 + 1.2924 w_k1_FDX_s013
                                 + 0.956325 w_k1_IBM_s013
                                 + 1.32686 w_k1_INTC_s013
                                 + 1.15056 w_k1_KO_s013 + 1.13876 w_k1_MAT_s013
                                 + 1.2588 w_k1_MSFT_s013 + 1.25209 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.973021 w_k1_ADM_s013
                                 + 1.02049 w_k1_AXP_s013
                                 + 1.05606 w_k1_DIS_s013
                                 + 1.02258 w_k1_FDX_s013
                                 + 0.881235 w_k1_IBM_s013
                                 + 0.952571 w_k1_INTC_s013
                                 + 0.989271 w_k1_KO_s013
                                 + 1.01481 w_k1_MAT_s013
                                 + 1.10212 w_k1_MSFT_s013
                                 + 1.09297 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.08737 w_k1_ADM_s013
                                 + 1.06355 w_k1_AXP_s013
                                 + 1.36151 w_k1_DIS_s013
                                 + 1.25343 w_k1_FDX_s013
                                 + 1.10977 w_k1_IBM_s013
                                 + 1.38611 w_k1_INTC_s013
                                 + 1.17598 w_k1_KO_s013
                                 + 0.811406 w_k1_MAT_s013
                                 + 1.39139 w_k1_MSFT_s013
                                 + 1.04209 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.789802 w_k1_ADM_s013
                                 + 0.672157 w_k1_AXP_s013
                                 + 0.791206 w_k1_DIS_s013
                                 + 0.921777 w_k1_FDX_s013
                                 + 0.706072 w_k1_IBM_s013
                                 + 0.924459 w_k1_INTC_s013
                                 + 0.799335 w_k1_KO_s013
                                 + 0.758607 w_k1_MAT_s013
                                 + 0.872084 w_k1_MSFT_s013
                                 + 0.942906 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.08184 w_k1_ADM_s013
                                 + 1.36778 w_k1_AXP_s013
                                 + 1.74386 w_k1_DIS_s013
                                 + 1.28634 w_k1_FDX_s013
                                 + 1.54076 w_k1_IBM_s013
                                 + 1.10831 w_k1_INTC_s013
                                 + 1.28385 w_k1_KO_s013 + 1.83619 w_k1_MAT_s013
                                 + 1.34447 w_k1_MSFT_s013
                                 + 1.06837 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.27082 w_k1_ADM_s013
                                 + 1.04003 w_k1_AXP_s013
                                 + 1.22266 w_k1_DIS_s013
                                 + 1.07734 w_k1_FDX_s013
                                 + 1.40254 w_k1_IBM_s013
                                 + 1.1421 w_k1_INTC_s013 + 1.42871 w_k1_KO_s013
                                 + 1.54495 w_k1_MAT_s013
                                 + 1.13695 w_k1_MSFT_s013
                                 + 1.10305 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.22812 w_k1_ADM_s014
                                 + 1.09304 w_k1_AXP_s014
                                 + 1.40009 w_k1_DIS_s014
                                 + 1.13997 w_k1_FDX_s014
                                 + 1.50859 w_k1_IBM_s014
                                 + 1.23027 w_k1_INTC_s014
                                 + 1.41258 w_k1_KO_s014 + 1.54226 w_k1_MAT_s014
                                 + 1.22625 w_k1_MSFT_s014
                                 + 1.13443 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.928467 w_k1_ADM_s014
                                 + 0.714413 w_k1_AXP_s014
                                 + 1.17152 w_k1_DIS_s014
                                 + 0.996082 w_k1_FDX_s014
                                 + 1.00176 w_k1_IBM_s014
                                 + 1.12336 w_k1_INTC_s014
                                 + 1.03883 w_k1_KO_s014
                                 + 0.796941 w_k1_MAT_s014
                                 + 1.04523 w_k1_MSFT_s014
                                 + 0.979464 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.58798 w_k1_ADM_s014
                                 + 1.19022 w_k1_AXP_s014
                                 + 1.53624 w_k1_DIS_s014
                                 + 1.42164 w_k1_FDX_s014
                                 + 1.15146 w_k1_IBM_s014
                                 + 1.36521 w_k1_INTC_s014
                                 + 1.18627 w_k1_KO_s014
                                 + 0.922755 w_k1_MAT_s014
                                 + 1.20392 w_k1_MSFT_s014
                                 + 1.16831 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.794292 w_k1_ADM_s014
                                 + 0.837373 w_k1_AXP_s014
                                 + 0.889655 w_k1_DIS_s014
                                 + 0.972109 w_k1_FDX_s014
                                 + 0.706776 w_k1_IBM_s014
                                 + 1.25927 w_k1_INTC_s014
                                 + 0.688067 w_k1_KO_s014
                                 + 0.608055 w_k1_MAT_s014
                                 + 0.949111 w_k1_MSFT_s014
                                 + 0.880881 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.857381 w_k1_ADM_s014
                                 + 0.730131 w_k1_AXP_s014
                                 + 1.00881 w_k1_DIS_s014
                                 + 0.822179 w_k1_FDX_s014
                                 + 1.34214 w_k1_IBM_s014
                                 + 0.983097 w_k1_INTC_s014
                                 + 1.18436 w_k1_KO_s014 + 1.48343 w_k1_MAT_s014
                                 + 0.928331 w_k1_MSFT_s014
                                 + 1.06969 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.08298 w_k1_ADM_s014
                                 + 0.926283 w_k1_AXP_s014
                                 + 0.736647 w_k1_DIS_s014
                                 + 0.953906 w_k1_FDX_s014
                                 + 1.03728 w_k1_IBM_s014
                                 + 1.15327 w_k1_INTC_s014
                                 + 0.893832 w_k1_KO_s014
                                 + 1.12651 w_k1_MAT_s014
                                 + 0.921011 w_k1_MSFT_s014
                                 + 0.936112 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.03831 w_k1_ADM_s014
                                 + 1.39607 w_k1_AXP_s014
                                 + 1.36658 w_k1_DIS_s014
                                 + 1.19156 w_k1_FDX_s014 + 1.1707 w_k1_IBM_s014
                                 + 1.00634 w_k1_INTC_s014
                                 + 1.04788 w_k1_KO_s014 + 1.6736 w_k1_MAT_s014
                                 + 0.899647 w_k1_MSFT_s014
                                 + 1.06893 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.712373 w_k1_ADM_s014
                                 + 0.731347 w_k1_AXP_s014
                                 + 0.559275 w_k1_DIS_s014
                                 + 0.678465 w_k1_FDX_s014
                                 + 0.919996 w_k1_IBM_s014
                                 + 0.888773 w_k1_INTC_s014
                                 + 0.73914 w_k1_KO_s014
                                 + 0.781696 w_k1_MAT_s014
                                 + 0.678144 w_k1_MSFT_s014
                                 + 0.796091 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.954971 w_k1_ADM_s015
                                 + 0.687723 w_k1_AXP_s015
                                 + 0.659188 w_k1_DIS_s015
                                 + 0.753642 w_k1_FDX_s015
                                 + 0.980644 w_k1_IBM_s015
                                 + 0.64904 w_k1_INTC_s015
                                 + 0.982592 w_k1_KO_s015
                                 + 0.763447 w_k1_MAT_s015
                                 + 0.685875 w_k1_MSFT_s015
                                 + 0.840782 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.51726 w_k1_ADM_s015
                                 + 0.961676 w_k1_AXP_s015
                                 + 1.08771 w_k1_DIS_s015
                                 + 1.15004 w_k1_FDX_s015
                                 + 0.777837 w_k1_IBM_s015
                                 + 0.788188 w_k1_INTC_s015
                                 + 1.02298 w_k1_KO_s015
                                 + 0.831561 w_k1_MAT_s015
                                 + 1.04466 w_k1_MSFT_s015
                                 + 0.91077 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.00179 w_k1_ADM_s015
                                 + 0.784789 w_k1_AXP_s015
                                 + 0.610506 w_k1_DIS_s015
                                 + 0.806459 w_k1_FDX_s015
                                 + 0.684629 w_k1_IBM_s015
                                 + 0.679152 w_k1_INTC_s015
                                 + 0.665793 w_k1_KO_s015
                                 + 0.662414 w_k1_MAT_s015
                                 + 0.758532 w_k1_MSFT_s015
                                 + 0.831338 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.858582 w_k1_ADM_s015
                                 + 0.80356 w_k1_AXP_s015
                                 + 0.621607 w_k1_DIS_s015
                                 + 0.737034 w_k1_FDX_s015
                                 + 0.891787 w_k1_IBM_s015
                                 + 0.627596 w_k1_INTC_s015
                                 + 0.797336 w_k1_KO_s015
                                 + 1.08352 w_k1_MAT_s015
                                 + 0.63914 w_k1_MSFT_s015
                                 + 0.845557 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.42024 w_k1_ADM_s015
                                 + 1.27475 w_k1_AXP_s015 + 1.3238 w_k1_DIS_s015
                                 + 1.32439 w_k1_FDX_s015
                                 + 1.06834 w_k1_IBM_s015
                                 + 1.08295 w_k1_INTC_s015
                                 + 1.29301 w_k1_KO_s015 + 1.03175 w_k1_MAT_s015
                                 + 1.14789 w_k1_MSFT_s015
                                 + 1.11483 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.36433 w_k1_ADM_s015
                                 + 1.27536 w_k1_AXP_s015
                                 + 1.18168 w_k1_DIS_s015
                                 + 1.07874 w_k1_FDX_s015
                                 + 1.66585 w_k1_IBM_s015
                                 + 1.18788 w_k1_INTC_s015
                                 + 1.22611 w_k1_KO_s015 + 1.70079 w_k1_MAT_s015
                                 + 1.1699 w_k1_MSFT_s015 + 1.35623 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.02985 w_k1_ADM_s015
                                 + 0.921144 w_k1_AXP_s015
                                 + 0.983895 w_k1_DIS_s015
                                 + 1.04683 w_k1_FDX_s015
                                 + 1.03272 w_k1_IBM_s015
                                 + 0.920891 w_k1_INTC_s015
                                 + 0.973303 w_k1_KO_s015
                                 + 1.19859 w_k1_MAT_s015
                                 + 0.925218 w_k1_MSFT_s015
                                 + 0.863529 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.61541 w_k1_ADM_s015
                                 + 1.13928 w_k1_AXP_s015
                                 + 1.33373 w_k1_DIS_s015
                                 + 1.20632 w_k1_FDX_s015 + 1.2683 w_k1_IBM_s015
                                 + 0.736926 w_k1_INTC_s015
                                 + 1.34906 w_k1_KO_s015 + 1.2961 w_k1_MAT_s015
                                 + 1.03932 w_k1_MSFT_s015 + 1.2279 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.91824 w_k1_ADM_s016
                                 + 0.898628 w_k1_AXP_s016
                                 + 0.861618 w_k1_DIS_s016
                                 + 0.797104 w_k1_FDX_s016
                                 + 0.965499 w_k1_IBM_s016
                                 + 0.78341 w_k1_INTC_s016
                                 + 0.863018 w_k1_KO_s016
                                 + 0.952232 w_k1_MAT_s016
                                 + 0.826149 w_k1_MSFT_s016
                                 + 0.945071 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.0776 w_k1_ADM_s016
                                 + 1.03099 w_k1_AXP_s016
                                 + 0.851591 w_k1_DIS_s016
                                 + 0.985837 w_k1_FDX_s016
                                 + 0.818044 w_k1_IBM_s016
                                 + 0.721359 w_k1_INTC_s016
                                 + 0.828029 w_k1_KO_s016
                                 + 0.96639 w_k1_MAT_s016
                                 + 0.784355 w_k1_MSFT_s016
                                 + 0.934645 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.790755 w_k1_ADM_s016
                                 + 0.952211 w_k1_AXP_s016
                                 + 0.861722 w_k1_DIS_s016
                                 + 0.829971 w_k1_FDX_s016
                                 + 1.25092 w_k1_IBM_s016
                                 + 0.944611 w_k1_INTC_s016
                                 + 0.948414 w_k1_KO_s016
                                 + 1.39063 w_k1_MAT_s016
                                 + 0.809144 w_k1_MSFT_s016
                                 + 1.06749 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.27501 w_k1_ADM_s016
                                 + 1.1412 w_k1_AXP_s016 + 1.65302 w_k1_DIS_s016
                                 + 1.2241 w_k1_FDX_s016 + 1.08002 w_k1_IBM_s016
                                 + 1.22315 w_k1_INTC_s016
                                 + 1.40162 w_k1_KO_s016
                                 + 0.878119 w_k1_MAT_s016
                                 + 1.16616 w_k1_MSFT_s016
                                 + 1.40003 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.08564 w_k1_ADM_s016
                                 + 1.11359 w_k1_AXP_s016
                                 + 1.21646 w_k1_DIS_s016
                                 + 1.00711 w_k1_FDX_s016
                                 + 1.58945 w_k1_IBM_s016
                                 + 0.990413 w_k1_INTC_s016
                                 + 1.28006 w_k1_KO_s016 + 1.40977 w_k1_MAT_s016
                                 + 0.940329 w_k1_MSFT_s016
                                 + 1.07752 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.858884 w_k1_ADM_s016
                                 + 0.678636 w_k1_AXP_s016
                                 + 0.629948 w_k1_DIS_s016
                                 + 0.74633 w_k1_FDX_s016
                                 + 0.70362 w_k1_IBM_s016
                                 + 0.694502 w_k1_INTC_s016
                                 + 0.658807 w_k1_KO_s016
                                 + 0.608783 w_k1_MAT_s016
                                 + 0.656496 w_k1_MSFT_s016
                                 + 0.817978 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.06415 w_k1_ADM_s016
                                 + 1.06819 w_k1_AXP_s016
                                 + 1.05067 w_k1_DIS_s016
                                 + 1.05467 w_k1_FDX_s016
                                 + 0.674566 w_k1_IBM_s016
                                 + 0.790178 w_k1_INTC_s016
                                 + 0.741483 w_k1_KO_s016
                                 + 0.822824 w_k1_MAT_s016
                                 + 0.889376 w_k1_MSFT_s016
                                 + 0.949779 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.09428 w_k1_ADM_s016
                                 + 1.38326 w_k1_AXP_s016
                                 + 1.21337 w_k1_DIS_s016
                                 + 1.14778 w_k1_FDX_s016
                                 + 1.34558 w_k1_IBM_s016
                                 + 1.01718 w_k1_INTC_s016 + 1.2491 w_k1_KO_s016
                                 + 1.41899 w_k1_MAT_s016
                                 + 1.00301 w_k1_MSFT_s016
                                 + 1.23065 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.28062 w_k1_ADM_s017
                                 + 1.02296 w_k1_AXP_s017
                                 + 1.33844 w_k1_DIS_s017
                                 + 1.30072 w_k1_FDX_s017
                                 + 0.779574 w_k1_IBM_s017
                                 + 1.42545 w_k1_INTC_s017
                                 + 1.09518 w_k1_KO_s017
                                 + 0.983461 w_k1_MAT_s017
                                 + 1.45311 w_k1_MSFT_s017
                                 + 1.03652 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.20693 w_k1_ADM_s017
                                 + 1.25407 w_k1_AXP_s017
                                 + 2.02297 w_k1_DIS_s017
                                 + 1.41212 w_k1_FDX_s017 + 1.1399 w_k1_IBM_s017
                                 + 1.23921 w_k1_INTC_s017 + 1.4682 w_k1_KO_s017
                                 + 1.12185 w_k1_MAT_s017
                                 + 1.71289 w_k1_MSFT_s017
                                 + 1.32084 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.88044 w_k1_ADM_s017
                                 + 0.781564 w_k1_AXP_s017
                                 + 0.557482 w_k1_DIS_s017
                                 + 0.722325 w_k1_FDX_s017
                                 + 1.07805 w_k1_IBM_s017
                                 + 0.73229 w_k1_INTC_s017
                                 + 0.836472 w_k1_KO_s017
                                 + 1.34348 w_k1_MAT_s017
                                 + 0.687168 w_k1_MSFT_s017
                                 + 0.929437 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.918831 w_k1_ADM_s017
                                 + 1.24932 w_k1_AXP_s017
                                 + 1.21266 w_k1_DIS_s017 + 1.2434 w_k1_FDX_s017
                                 + 1.0077 w_k1_IBM_s017
                                 + 1.04112 w_k1_INTC_s017
                                 + 1.08395 w_k1_KO_s017 + 1.05278 w_k1_MAT_s017
                                 + 1.08053 w_k1_MSFT_s017
                                 + 1.09984 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.976319 w_k1_ADM_s017
                                 + 1.13225 w_k1_AXP_s017
                                 + 1.25033 w_k1_DIS_s017
                                 + 1.14473 w_k1_FDX_s017
                                 + 1.65045 w_k1_IBM_s017
                                 + 1.02439 w_k1_INTC_s017
                                 + 1.46657 w_k1_KO_s017 + 2.1638 w_k1_MAT_s017
                                 + 1.07585 w_k1_MSFT_s017
                                 + 1.28672 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.665891 w_k1_ADM_s017
                                 + 0.706046 w_k1_AXP_s017
                                 + 0.697345 w_k1_DIS_s017
                                 + 0.776645 w_k1_FDX_s017
                                 + 0.960014 w_k1_IBM_s017
                                 + 0.802889 w_k1_INTC_s017
                                 + 0.793015 w_k1_KO_s017 + 0.7874 w_k1_MAT_s017
                                 + 0.882784 w_k1_MSFT_s017
                                 + 0.863172 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.03694 w_k1_ADM_s017
                                 + 1.08747 w_k1_AXP_s017
                                 + 0.939107 w_k1_DIS_s017
                                 + 0.966957 w_k1_FDX_s017
                                 + 1.0666 w_k1_IBM_s017
                                 + 0.827679 w_k1_INTC_s017
                                 + 0.882028 w_k1_KO_s017
                                 + 1.04857 w_k1_MAT_s017
                                 + 0.848112 w_k1_MSFT_s017
                                 + 1.07566 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.882716 w_k1_ADM_s017
                                 + 0.756357 w_k1_AXP_s017
                                 + 0.81641 w_k1_DIS_s017
                                 + 0.939038 w_k1_FDX_s017
                                 + 0.646203 w_k1_IBM_s017
                                 + 0.90461 w_k1_INTC_s017
                                 + 0.692431 w_k1_KO_s017
                                 + 0.686536 w_k1_MAT_s017
                                 + 0.815239 w_k1_MSFT_s017
                                 + 0.900356 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04423 w_k1_ADM_s100
                                 + 1.12631 w_k1_AXP_s100
                                 + 0.952864 w_k1_DIS_s100
                                 + 1.09457 w_k1_FDX_s100
                                 + 0.736579 w_k1_IBM_s100
                                 + 1.30682 w_k1_INTC_s100
                                 + 0.92045 w_k1_KO_s100 + 0.84601 w_k1_MAT_s100
                                 + 1.10861 w_k1_MSFT_s100
                                 + 1.03151 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.0597 w_k1_ADM_s100
                                 + 1.38628 w_k1_AXP_s100
                                 + 1.06247 w_k1_DIS_s100 + 1.0485 w_k1_FDX_s100
                                 + 1.34318 w_k1_IBM_s100
                                 + 1.22316 w_k1_INTC_s100 + 1.0616 w_k1_KO_s100
                                 + 1.32153 w_k1_MAT_s100
                                 + 1.04805 w_k1_MSFT_s100
                                 + 1.18311 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.0356 w_k1_ADM_s100
                                 + 1.37958 w_k1_AXP_s100 + 1.3899 w_k1_DIS_s100
                                 + 1.33157 w_k1_FDX_s100
                                 + 1.05769 w_k1_IBM_s100
                                 + 1.57886 w_k1_INTC_s100
                                 + 1.06506 w_k1_KO_s100
                                 + 0.895924 w_k1_MAT_s100
                                 + 1.31933 w_k1_MSFT_s100
                                 + 1.11871 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.724996 w_k1_ADM_s100
                                 + 0.97576 w_k1_AXP_s100
                                 + 0.770615 w_k1_DIS_s100
                                 + 0.850638 w_k1_FDX_s100
                                 + 1.27876 w_k1_IBM_s100
                                 + 1.20448 w_k1_INTC_s100
                                 + 0.958307 w_k1_KO_s100
                                 + 1.06534 w_k1_MAT_s100
                                 + 0.877175 w_k1_MSFT_s100
                                 + 0.932572 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.632658 w_k1_ADM_s100
                                 + 0.638963 w_k1_AXP_s100
                                 + 0.611388 w_k1_DIS_s100
                                 + 0.717933 w_k1_FDX_s100
                                 + 0.773882 w_k1_IBM_s100
                                 + 0.79957 w_k1_INTC_s100
                                 + 0.645148 w_k1_KO_s100
                                 + 0.529429 w_k1_MAT_s100
                                 + 0.632604 w_k1_MSFT_s100
                                 + 0.810313 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.995638 w_k1_ADM_s100
                                 + 1.05403 w_k1_AXP_s100
                                 + 0.861965 w_k1_DIS_s100
                                 + 0.953395 w_k1_FDX_s100
                                 + 1.11995 w_k1_IBM_s100
                                 + 1.21592 w_k1_INTC_s100
                                 + 0.873157 w_k1_KO_s100
                                 + 0.889501 w_k1_MAT_s100
                                 + 0.942981 w_k1_MSFT_s100
                                 + 0.96163 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.799604 w_k1_ADM_s100
                                 + 0.909487 w_k1_AXP_s100
                                 + 0.75176 w_k1_DIS_s100
                                 + 0.873543 w_k1_FDX_s100
                                 + 0.822034 w_k1_IBM_s100
                                 + 1.08176 w_k1_INTC_s100
                                 + 0.824069 w_k1_KO_s100
                                 + 0.911813 w_k1_MAT_s100
                                 + 0.948488 w_k1_MSFT_s100
                                 + 0.96628 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.01421 w_k1_ADM_s100
                                 + 1.38477 w_k1_AXP_s100
                                 + 1.48646 w_k1_DIS_s100
                                 + 1.18571 w_k1_FDX_s100
                                 + 1.46183 w_k1_IBM_s100
                                 + 1.50191 w_k1_INTC_s100
                                 + 1.33427 w_k1_KO_s100 + 1.18468 w_k1_MAT_s100
                                 + 1.15938 w_k1_MSFT_s100
                                 + 1.05869 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.778925 w_k1_ADM_s101
                                 + 0.722029 w_k1_AXP_s101
                                 + 0.688781 w_k1_DIS_s101
                                 + 0.939298 w_k1_FDX_s101
                                 + 0.658993 w_k1_IBM_s101
                                 + 0.91008 w_k1_INTC_s101
                                 + 0.74667 w_k1_KO_s101
                                 + 0.719799 w_k1_MAT_s101
                                 + 0.803138 w_k1_MSFT_s101
                                 + 0.933838 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.33425 w_k1_ADM_s101
                                 + 1.30863 w_k1_AXP_s101
                                 + 1.18999 w_k1_DIS_s101
                                 + 1.12537 w_k1_FDX_s101
                                 + 1.18482 w_k1_IBM_s101
                                 + 1.06886 w_k1_INTC_s101
                                 + 1.09836 w_k1_KO_s101 + 1.45699 w_k1_MAT_s101
                                 + 1.03526 w_k1_MSFT_s101
                                 + 1.11346 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.918776 w_k1_ADM_s101
                                 + 1.02351 w_k1_AXP_s101
                                 + 0.990884 w_k1_DIS_s101
                                 + 1.00886 w_k1_FDX_s101
                                 + 1.55735 w_k1_IBM_s101
                                 + 1.11126 w_k1_INTC_s101
                                 + 1.16786 w_k1_KO_s101 + 1.45287 w_k1_MAT_s101
                                 + 0.949991 w_k1_MSFT_s101
                                 + 1.01574 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.872928 w_k1_ADM_s101
                                 + 0.611358 w_k1_AXP_s101
                                 + 0.618891 w_k1_DIS_s101
                                 + 0.636805 w_k1_FDX_s101
                                 + 0.862039 w_k1_IBM_s101
                                 + 0.670293 w_k1_INTC_s101
                                 + 0.792227 w_k1_KO_s101
                                 + 0.723633 w_k1_MAT_s101
                                 + 0.627161 w_k1_MSFT_s101
                                 + 0.71775 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.20224 w_k1_ADM_s101
                                 + 1.35085 w_k1_AXP_s101
                                 + 2.03752 w_k1_DIS_s101
                                 + 1.48846 w_k1_FDX_s101
                                 + 1.29441 w_k1_IBM_s101
                                 + 0.943286 w_k1_INTC_s101
                                 + 1.29658 w_k1_KO_s101 + 1.60791 w_k1_MAT_s101
                                 + 1.23025 w_k1_MSFT_s101
                                 + 1.22715 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.02577 w_k1_ADM_s101
                                 + 0.689377 w_k1_AXP_s101
                                 + 0.855431 w_k1_DIS_s101
                                 + 0.902001 w_k1_FDX_s101
                                 + 1.05879 w_k1_IBM_s101
                                 + 0.949824 w_k1_INTC_s101
                                 + 1.0224 w_k1_KO_s101 + 0.932944 w_k1_MAT_s101
                                 + 0.863814 w_k1_MSFT_s101
                                 + 0.935401 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.71794 w_k1_ADM_s101
                                 + 1.10518 w_k1_AXP_s101
                                 + 1.17954 w_k1_DIS_s101
                                 + 1.19857 w_k1_FDX_s101
                                 + 0.99152 w_k1_IBM_s101
                                 + 1.04892 w_k1_INTC_s101
                                 + 1.12261 w_k1_KO_s101 + 1.00694 w_k1_MAT_s101
                                 + 1.08024 w_k1_MSFT_s101
                                 + 1.07236 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.17294 w_k1_ADM_s101
                                 + 0.836957 w_k1_AXP_s101
                                 + 0.954747 w_k1_DIS_s101
                                 + 0.985547 w_k1_FDX_s101
                                 + 0.830018 w_k1_IBM_s101
                                 + 0.892308 w_k1_INTC_s101
                                 + 0.920135 w_k1_KO_s101
                                 + 1.07935 w_k1_MAT_s101
                                 + 0.879482 w_k1_MSFT_s101
                                 + 1.03696 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.07392 w_k1_ADM_s102
                                 + 1.18188 w_k1_AXP_s102
                                 + 1.23249 w_k1_DIS_s102
                                 + 1.22394 w_k1_FDX_s102
                                 + 0.85548 w_k1_IBM_s102
                                 + 1.07392 w_k1_INTC_s102
                                 + 0.992073 w_k1_KO_s102
                                 + 1.63181 w_k1_MAT_s102
                                 + 1.15274 w_k1_MSFT_s102
                                 + 1.13442 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.65151 w_k1_ADM_s102
                                 + 1.6392 w_k1_AXP_s102 + 1.90106 w_k1_DIS_s102
                                 + 1.55785 w_k1_FDX_s102 + 1.4509 w_k1_IBM_s102
                                 + 1.4948 w_k1_INTC_s102 + 1.52595 w_k1_KO_s102
                                 + 1.64309 w_k1_MAT_s102
                                 + 1.56269 w_k1_MSFT_s102
                                 + 1.38493 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.0266 w_k1_ADM_s102
                                 + 1.18742 w_k1_AXP_s102
                                 + 1.12847 w_k1_DIS_s102
                                 + 1.16007 w_k1_FDX_s102
                                 + 1.27656 w_k1_IBM_s102
                                 + 1.30611 w_k1_INTC_s102
                                 + 1.04153 w_k1_KO_s102 + 1.72922 w_k1_MAT_s102
                                 + 1.10982 w_k1_MSFT_s102
                                 + 1.09621 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.978582 w_k1_ADM_s102
                                 + 1.17983 w_k1_AXP_s102 + 1.195 w_k1_DIS_s102
                                 + 1.07303 w_k1_FDX_s102
                                 + 1.73109 w_k1_IBM_s102
                                 + 1.37042 w_k1_INTC_s102
                                 + 1.31526 w_k1_KO_s102 + 2.46873 w_k1_MAT_s102
                                 + 1.13821 w_k1_MSFT_s102
                                 + 1.11237 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.926468 w_k1_ADM_s102
                                 + 1.22593 w_k1_AXP_s102
                                 + 0.793613 w_k1_DIS_s102
                                 + 0.982803 w_k1_FDX_s102
                                 + 0.994759 w_k1_IBM_s102
                                 + 1.06639 w_k1_INTC_s102
                                 + 0.846794 w_k1_KO_s102 + 1.365 w_k1_MAT_s102
                                 + 0.930761 w_k1_MSFT_s102
                                 + 1.07228 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.00234 w_k1_ADM_s102
                                 + 0.856347 w_k1_AXP_s102
                                 + 0.813547 w_k1_DIS_s102
                                 + 1.04369 w_k1_FDX_s102
                                 + 0.660856 w_k1_IBM_s102
                                 + 1.14137 w_k1_INTC_s102
                                 + 0.822496 w_k1_KO_s102
                                 + 0.805653 w_k1_MAT_s102
                                 + 1.03489 w_k1_MSFT_s102
                                 + 0.956213 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.947951 w_k1_ADM_s102
                                 + 0.636822 w_k1_AXP_s102
                                 + 0.568589 w_k1_DIS_s102
                                 + 0.751838 w_k1_FDX_s102
                                 + 0.621862 w_k1_IBM_s102
                                 + 0.813992 w_k1_INTC_s102
                                 + 0.663846 w_k1_KO_s102
                                 + 0.810353 w_k1_MAT_s102
                                 + 0.874392 w_k1_MSFT_s102
                                 + 0.888521 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.863765 w_k1_ADM_s102
                                 + 0.923521 w_k1_AXP_s102
                                 + 0.715186 w_k1_DIS_s102
                                 + 0.778561 w_k1_FDX_s102
                                 + 0.851312 w_k1_IBM_s102
                                 + 0.910032 w_k1_INTC_s102
                                 + 0.874778 w_k1_KO_s102
                                 + 1.28985 w_k1_MAT_s102
                                 + 0.814085 w_k1_MSFT_s102
                                 + 0.962313 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.637429 w_k1_ADM_s103
                                 + 0.656379 w_k1_AXP_s103
                                 + 0.534928 w_k1_DIS_s103
                                 + 0.651966 w_k1_FDX_s103
                                 + 0.674714 w_k1_IBM_s103
                                 + 0.734826 w_k1_INTC_s103
                                 + 0.667269 w_k1_KO_s103
                                 + 0.598445 w_k1_MAT_s103
                                 + 0.682416 w_k1_MSFT_s103
                                 + 0.780525 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.20763 w_k1_ADM_s103
                                 + 1.50803 w_k1_AXP_s103
                                 + 1.44626 w_k1_DIS_s103
                                 + 1.41342 w_k1_FDX_s103
                                 + 0.822136 w_k1_IBM_s103
                                 + 1.20401 w_k1_INTC_s103
                                 + 1.09291 w_k1_KO_s103
                                 + 0.962673 w_k1_MAT_s103
                                 + 1.31514 w_k1_MSFT_s103 + 1.2482 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.937516 w_k1_ADM_s103
                                 + 0.799951 w_k1_AXP_s103
                                 + 0.930931 w_k1_DIS_s103
                                 + 0.966783 w_k1_FDX_s103
                                 + 0.649404 w_k1_IBM_s103
                                 + 0.94257 w_k1_INTC_s103
                                 + 0.857258 w_k1_KO_s103
                                 + 0.564764 w_k1_MAT_s103
                                 + 1.04377 w_k1_MSFT_s103
                                 + 0.972463 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.754522 w_k1_ADM_s103
                                 + 0.862879 w_k1_AXP_s103
                                 + 0.817271 w_k1_DIS_s103
                                 + 0.699925 w_k1_FDX_s103
                                 + 1.2209 w_k1_IBM_s103
                                 + 0.976255 w_k1_INTC_s103
                                 + 1.06259 w_k1_KO_s103
                                 + 0.811475 w_k1_MAT_s103
                                 + 0.819257 w_k1_MSFT_s103
                                 + 0.922331 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.981621 w_k1_ADM_s103
                                 + 1.16469 w_k1_AXP_s103
                                 + 1.23355 w_k1_DIS_s103
                                 + 1.06915 w_k1_FDX_s103
                                 + 0.972058 w_k1_IBM_s103
                                 + 1.13674 w_k1_INTC_s103
                                 + 1.15515 w_k1_KO_s103
                                 + 0.786012 w_k1_MAT_s103
                                 + 1.16088 w_k1_MSFT_s103
                                 + 1.14193 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.874578 w_k1_ADM_s103
                                 + 1.12253 w_k1_AXP_s103
                                 + 1.13967 w_k1_DIS_s103
                                 + 1.02474 w_k1_FDX_s103
                                 + 1.13994 w_k1_IBM_s103
                                 + 0.880135 w_k1_INTC_s103
                                 + 1.17981 w_k1_KO_s103 + 1.24011 w_k1_MAT_s103
                                 + 0.960007 w_k1_MSFT_s103
                                 + 1.06684 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.995758 w_k1_ADM_s103
                                 + 1.12441 w_k1_AXP_s103
                                 + 0.79176 w_k1_DIS_s103
                                 + 0.926223 w_k1_FDX_s103
                                 + 0.916355 w_k1_IBM_s103
                                 + 0.75933 w_k1_INTC_s103
                                 + 0.856284 w_k1_KO_s103
                                 + 1.16587 w_k1_MAT_s103
                                 + 0.783348 w_k1_MSFT_s103
                                 + 0.98246 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.16519 w_k1_ADM_s103
                                 + 1.65353 w_k1_AXP_s103
                                 + 1.56686 w_k1_DIS_s103
                                 + 1.15109 w_k1_FDX_s103
                                 + 1.51949 w_k1_IBM_s103
                                 + 1.16383 w_k1_INTC_s103
                                 + 1.43873 w_k1_KO_s103 + 1.52493 w_k1_MAT_s103
                                 + 1.20547 w_k1_MSFT_s103
                                 + 1.20247 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.21401 w_k1_ADM_s104
                                 + 1.14605 w_k1_AXP_s104
                                 + 1.26856 w_k1_DIS_s104
                                 + 1.09975 w_k1_FDX_s104
                                 + 1.11872 w_k1_IBM_s104
                                 + 1.13022 w_k1_INTC_s104
                                 + 1.40475 w_k1_KO_s104 + 1.07985 w_k1_MAT_s104
                                 + 1.17589 w_k1_MSFT_s104
                                 + 1.11319 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.06244 w_k1_ADM_s104
                                 + 0.776758 w_k1_AXP_s104
                                 + 0.745156 w_k1_DIS_s104
                                 + 0.751385 w_k1_FDX_s104
                                 + 0.729378 w_k1_IBM_s104
                                 + 0.779926 w_k1_INTC_s104
                                 + 0.769235 w_k1_KO_s104
                                 + 0.713637 w_k1_MAT_s104
                                 + 0.849889 w_k1_MSFT_s104
                                 + 0.865338 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.18655 w_k1_ADM_s104
                                 + 1.37201 w_k1_AXP_s104
                                 + 1.28129 w_k1_DIS_s104
                                 + 1.14421 w_k1_FDX_s104
                                 + 1.47864 w_k1_IBM_s104
                                 + 1.38508 w_k1_INTC_s104 + 1.2587 w_k1_KO_s104
                                 + 1.96742 w_k1_MAT_s104
                                 + 1.14067 w_k1_MSFT_s104
                                 + 1.11404 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.10268 w_k1_ADM_s104
                                 + 0.88736 w_k1_AXP_s104
                                 + 0.953299 w_k1_DIS_s104
                                 + 0.901992 w_k1_FDX_s104
                                 + 0.947094 w_k1_IBM_s104
                                 + 0.859287 w_k1_INTC_s104
                                 + 0.877575 w_k1_KO_s104
                                 + 0.708791 w_k1_MAT_s104
                                 + 0.839003 w_k1_MSFT_s104
                                 + 0.963985 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.12018 w_k1_ADM_s104
                                 + 1.01949 w_k1_AXP_s104
                                 + 1.13508 w_k1_DIS_s104
                                 + 1.13927 w_k1_FDX_s104
                                 + 0.674775 w_k1_IBM_s104
                                 + 1.18734 w_k1_INTC_s104
                                 + 0.859771 w_k1_KO_s104
                                 + 0.487569 w_k1_MAT_s104
                                 + 1.2484 w_k1_MSFT_s104 + 1.04633 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.37467 w_k1_ADM_s104
                                 + 1.28536 w_k1_AXP_s104
                                 + 0.840904 w_k1_DIS_s104
                                 + 1.08735 w_k1_FDX_s104
                                 + 0.735224 w_k1_IBM_s104
                                 + 0.959392 w_k1_INTC_s104
                                 + 0.99219 w_k1_KO_s104
                                 + 0.781592 w_k1_MAT_s104
                                 + 0.912686 w_k1_MSFT_s104
                                 + 1.07482 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.57301 w_k1_ADM_s104
                                 + 1.34631 w_k1_AXP_s104
                                 + 1.88213 w_k1_DIS_s104
                                 + 1.36564 w_k1_FDX_s104
                                 + 1.04668 w_k1_IBM_s104
                                 + 1.50736 w_k1_INTC_s104
                                 + 1.42906 w_k1_KO_s104
                                 + 0.786317 w_k1_MAT_s104
                                 + 1.56247 w_k1_MSFT_s104
                                 + 1.17081 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.18844 w_k1_ADM_s104
                                 + 1.26579 w_k1_AXP_s104
                                 + 0.947931 w_k1_DIS_s104
                                 + 0.912328 w_k1_FDX_s104
                                 + 1.33745 w_k1_IBM_s104
                                 + 0.914182 w_k1_INTC_s104
                                 + 1.15309 w_k1_KO_s104 + 1.53941 w_k1_MAT_s104
                                 + 0.885628 w_k1_MSFT_s104
                                 + 0.959649 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.728536 w_k1_ADM_s105
                                 + 0.990931 w_k1_AXP_s105
                                 + 0.664569 w_k1_DIS_s105
                                 + 0.795567 w_k1_FDX_s105
                                 + 0.770026 w_k1_IBM_s105
                                 + 0.97205 w_k1_INTC_s105
                                 + 0.958155 w_k1_KO_s105 + 1.3142 w_k1_MAT_s105
                                 + 0.760699 w_k1_MSFT_s105
                                 + 1.00531 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.962748 w_k1_ADM_s105
                                 + 1.50727 w_k1_AXP_s105
                                 + 1.39959 w_k1_DIS_s105
                                 + 1.37997 w_k1_FDX_s105
                                 + 1.06607 w_k1_IBM_s105
                                 + 1.30646 w_k1_INTC_s105
                                 + 1.14984 w_k1_KO_s105 + 1.88509 w_k1_MAT_s105
                                 + 1.09816 w_k1_MSFT_s105 + 1.111 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.796635 w_k1_ADM_s105
                                 + 0.884481 w_k1_AXP_s105
                                 + 1.10233 w_k1_DIS_s105
                                 + 1.10163 w_k1_FDX_s105
                                 + 0.935228 w_k1_IBM_s105
                                 + 1.24594 w_k1_INTC_s105
                                 + 1.11922 w_k1_KO_s105 + 1.13755 w_k1_MAT_s105
                                 + 1.10271 w_k1_MSFT_s105
                                 + 1.07485 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.876034 w_k1_ADM_s105
                                 + 0.863523 w_k1_AXP_s105
                                 + 0.812333 w_k1_DIS_s105
                                 + 0.9077 w_k1_FDX_s105 + 1.04533 w_k1_IBM_s105
                                 + 1.1891 w_k1_INTC_s105 + 1.07049 w_k1_KO_s105
                                 + 1.10057 w_k1_MAT_s105
                                 + 0.947264 w_k1_MSFT_s105
                                 + 1.06126 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.844156 w_k1_ADM_s105
                                 + 1.03641 w_k1_AXP_s105
                                 + 0.812158 w_k1_DIS_s105
                                 + 0.89309 w_k1_FDX_s105
                                 + 1.30564 w_k1_IBM_s105
                                 + 1.28544 w_k1_INTC_s105
                                 + 1.13632 w_k1_KO_s105 + 1.53845 w_k1_MAT_s105
                                 + 0.966772 w_k1_MSFT_s105
                                 + 0.99739 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.720445 w_k1_ADM_s105
                                 + 0.810565 w_k1_AXP_s105
                                 + 0.671741 w_k1_DIS_s105
                                 + 0.94543 w_k1_FDX_s105
                                 + 0.657315 w_k1_IBM_s105
                                 + 1.13476 w_k1_INTC_s105
                                 + 0.749751 w_k1_KO_s105
                                 + 0.845324 w_k1_MAT_s105
                                 + 0.942298 w_k1_MSFT_s105
                                 + 0.822224 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.790118 w_k1_ADM_s105
                                 + 0.79073 w_k1_AXP_s105
                                 + 0.542546 w_k1_DIS_s105
                                 + 0.692001 w_k1_FDX_s105
                                 + 0.827461 w_k1_IBM_s105
                                 + 0.706979 w_k1_INTC_s105
                                 + 0.725564 w_k1_KO_s105
                                 + 1.18844 w_k1_MAT_s105
                                 + 0.712906 w_k1_MSFT_s105
                                 + 0.767367 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.15688 w_k1_ADM_s105
                                 + 1.22475 w_k1_AXP_s105
                                 + 1.35593 w_k1_DIS_s105
                                 + 1.08778 w_k1_FDX_s105
                                 + 1.61406 w_k1_IBM_s105
                                 + 1.30802 w_k1_INTC_s105
                                 + 1.37019 w_k1_KO_s105 + 1.73822 w_k1_MAT_s105
                                 + 1.03919 w_k1_MSFT_s105
                                 + 1.21515 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.05726 w_k1_ADM_s106
                                 + 1.00803 w_k1_AXP_s106
                                 + 0.99742 w_k1_DIS_s106
                                 + 1.02459 w_k1_FDX_s106
                                 + 1.25306 w_k1_IBM_s106
                                 + 1.13193 w_k1_INTC_s106
                                 + 1.07943 w_k1_KO_s106 + 1.36986 w_k1_MAT_s106
                                 + 1.1626 w_k1_MSFT_s106 + 1.08688 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12788 w_k1_ADM_s106
                                 + 1.18964 w_k1_AXP_s106
                                 + 1.68966 w_k1_DIS_s106
                                 + 1.24425 w_k1_FDX_s106
                                 + 1.36235 w_k1_IBM_s106
                                 + 1.48286 w_k1_INTC_s106
                                 + 1.47857 w_k1_KO_s106
                                 + 0.993691 w_k1_MAT_s106
                                 + 1.30092 w_k1_MSFT_s106
                                 + 1.19646 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.15227 w_k1_ADM_s106
                                 + 1.1093 w_k1_AXP_s106 + 1.30587 w_k1_DIS_s106
                                 + 1.35312 w_k1_FDX_s106
                                 + 0.902203 w_k1_IBM_s106
                                 + 1.27029 w_k1_INTC_s106
                                 + 1.09462 w_k1_KO_s106 + 1.1606 w_k1_MAT_s106
                                 + 1.3617 w_k1_MSFT_s106 + 1.18882 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.930864 w_k1_ADM_s106
                                 + 0.74434 w_k1_AXP_s106
                                 + 0.936102 w_k1_DIS_s106
                                 + 0.886366 w_k1_FDX_s106
                                 + 0.978398 w_k1_IBM_s106
                                 + 1.04984 w_k1_INTC_s106
                                 + 0.934512 w_k1_KO_s106
                                 + 0.969993 w_k1_MAT_s106
                                 + 1.02784 w_k1_MSFT_s106
                                 + 0.890732 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.00709 w_k1_ADM_s106
                                 + 1.02345 w_k1_AXP_s106
                                 + 0.778219 w_k1_DIS_s106
                                 + 0.987771 w_k1_FDX_s106
                                 + 0.795597 w_k1_IBM_s106
                                 + 1.03596 w_k1_INTC_s106
                                 + 0.748583 w_k1_KO_s106
                                 + 0.986912 w_k1_MAT_s106
                                 + 0.997157 w_k1_MSFT_s106
                                 + 0.944003 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.579103 w_k1_ADM_s106
                                 + 0.73091 w_k1_AXP_s106
                                 + 0.59302 w_k1_DIS_s106
                                 + 0.691789 w_k1_FDX_s106
                                 + 1.01167 w_k1_IBM_s106
                                 + 0.840237 w_k1_INTC_s106
                                 + 0.76057 w_k1_KO_s106 + 1.24984 w_k1_MAT_s106
                                 + 0.965938 w_k1_MSFT_s106
                                 + 0.803517 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.732156 w_k1_ADM_s106
                                 + 0.583297 w_k1_AXP_s106
                                 + 0.696666 w_k1_DIS_s106
                                 + 0.791768 w_k1_FDX_s106
                                 + 0.654132 w_k1_IBM_s106
                                 + 0.772291 w_k1_INTC_s106
                                 + 0.680821 w_k1_KO_s106
                                 + 0.601456 w_k1_MAT_s106
                                 + 0.914412 w_k1_MSFT_s106
                                 + 0.745977 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.914054 w_k1_ADM_s106
                                 + 0.889029 w_k1_AXP_s106
                                 + 1.05888 w_k1_DIS_s106
                                 + 0.99393 w_k1_FDX_s106
                                 + 1.67208 w_k1_IBM_s106
                                 + 1.13372 w_k1_INTC_s106
                                 + 1.25065 w_k1_KO_s106 + 1.62289 w_k1_MAT_s106
                                 + 1.04164 w_k1_MSFT_s106
                                 + 1.15111 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.702614 w_k1_ADM_s107
                                 + 0.85135 w_k1_AXP_s107
                                 + 0.844606 w_k1_DIS_s107
                                 + 0.873859 w_k1_FDX_s107
                                 + 0.837807 w_k1_IBM_s107
                                 + 0.84362 w_k1_INTC_s107
                                 + 0.782373 w_k1_KO_s107
                                 + 0.914491 w_k1_MAT_s107
                                 + 0.969817 w_k1_MSFT_s107
                                 + 0.906346 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.42749 w_k1_ADM_s107
                                 + 1.53223 w_k1_AXP_s107
                                 + 1.19266 w_k1_DIS_s107
                                 + 1.40568 w_k1_FDX_s107
                                 + 1.06639 w_k1_IBM_s107
                                 + 1.04214 w_k1_INTC_s107
                                 + 0.923747 w_k1_KO_s107
                                 + 0.895336 w_k1_MAT_s107
                                 + 1.18917 w_k1_MSFT_s107
                                 + 1.19684 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.00113 w_k1_ADM_s107
                                 + 0.857592 w_k1_AXP_s107
                                 + 0.750619 w_k1_DIS_s107
                                 + 0.807407 w_k1_FDX_s107
                                 + 1.13114 w_k1_IBM_s107
                                 + 0.804448 w_k1_INTC_s107
                                 + 0.940678 w_k1_KO_s107
                                 + 0.90145 w_k1_MAT_s107
                                 + 0.785362 w_k1_MSFT_s107
                                 + 0.903263 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.07333 w_k1_ADM_s107
                                 + 1.08598 w_k1_AXP_s107
                                 + 1.28263 w_k1_DIS_s107
                                 + 1.16676 w_k1_FDX_s107
                                 + 1.28288 w_k1_IBM_s107
                                 + 0.97251 w_k1_INTC_s107 + 1.129 w_k1_KO_s107
                                 + 1.22282 w_k1_MAT_s107
                                 + 0.99355 w_k1_MSFT_s107
                                 + 1.10385 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.886392 w_k1_ADM_s107
                                 + 0.973056 w_k1_AXP_s107
                                 + 0.997454 w_k1_DIS_s107 + 1.067 w_k1_FDX_s107
                                 + 1.05438 w_k1_IBM_s107
                                 + 0.995917 w_k1_INTC_s107
                                 + 0.888932 w_k1_KO_s107
                                 + 0.752897 w_k1_MAT_s107
                                 + 0.965695 w_k1_MSFT_s107
                                 + 0.996078 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.906927 w_k1_ADM_s107
                                 + 0.963828 w_k1_AXP_s107
                                 + 1.02619 w_k1_DIS_s107
                                 + 1.13157 w_k1_FDX_s107
                                 + 0.757978 w_k1_IBM_s107
                                 + 0.877583 w_k1_INTC_s107
                                 + 0.901437 w_k1_KO_s107
                                 + 0.758578 w_k1_MAT_s107
                                 + 1.08122 w_k1_MSFT_s107
                                 + 1.02924 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.919787 w_k1_ADM_s107
                                 + 1.56184 w_k1_AXP_s107
                                 + 1.26749 w_k1_DIS_s107
                                 + 1.07236 w_k1_FDX_s107
                                 + 1.81575 w_k1_IBM_s107
                                 + 0.980282 w_k1_INTC_s107
                                 + 1.30818 w_k1_KO_s107 + 2.08724 w_k1_MAT_s107
                                 + 1.13021 w_k1_MSFT_s107
                                 + 1.21619 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.706187 w_k1_ADM_s107
                                 + 0.771826 w_k1_AXP_s107
                                 + 0.652173 w_k1_DIS_s107
                                 + 0.83969 w_k1_FDX_s107
                                 + 0.677883 w_k1_IBM_s107
                                 + 0.751891 w_k1_INTC_s107
                                 + 0.622628 w_k1_KO_s107
                                 + 0.490109 w_k1_MAT_s107
                                 + 0.722871 w_k1_MSFT_s107
                                 + 0.755625 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.866419 w_k1_ADM_s110
                                 + 0.63459 w_k1_AXP_s110
                                 + 0.635739 w_k1_DIS_s110
                                 + 0.764326 w_k1_FDX_s110
                                 + 0.702759 w_k1_IBM_s110
                                 + 0.661618 w_k1_INTC_s110
                                 + 0.740875 w_k1_KO_s110
                                 + 0.566907 w_k1_MAT_s110
                                 + 0.854803 w_k1_MSFT_s110
                                 + 0.864001 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.32992 w_k1_ADM_s110
                                 + 1.08876 w_k1_AXP_s110
                                 + 1.30971 w_k1_DIS_s110
                                 + 1.22748 w_k1_FDX_s110
                                 + 0.917596 w_k1_IBM_s110
                                 + 1.10016 w_k1_INTC_s110
                                 + 1.02651 w_k1_KO_s110
                                 + 0.998548 w_k1_MAT_s110
                                 + 1.16138 w_k1_MSFT_s110
                                 + 1.00784 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.790203 w_k1_ADM_s110
                                 + 0.905481 w_k1_AXP_s110
                                 + 0.7135 w_k1_DIS_s110
                                 + 0.741572 w_k1_FDX_s110
                                 + 1.1393 w_k1_IBM_s110
                                 + 0.817445 w_k1_INTC_s110
                                 + 0.942917 w_k1_KO_s110
                                 + 1.20057 w_k1_MAT_s110
                                 + 0.860122 w_k1_MSFT_s110
                                 + 0.928604 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.19516 w_k1_ADM_s110
                                 + 0.989011 w_k1_AXP_s110
                                 + 1.11893 w_k1_DIS_s110
                                 + 0.980611 w_k1_FDX_s110
                                 + 1.1876 w_k1_IBM_s110
                                 + 0.959434 w_k1_INTC_s110
                                 + 1.10767 w_k1_KO_s110
                                 + 0.950343 w_k1_MAT_s110
                                 + 0.981849 w_k1_MSFT_s110
                                 + 1.00364 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.18849 w_k1_ADM_s110
                                 + 1.48227 w_k1_AXP_s110
                                 + 1.97257 w_k1_DIS_s110
                                 + 1.38954 w_k1_FDX_s110 + 1.4427 w_k1_IBM_s110
                                 + 1.50548 w_k1_INTC_s110
                                 + 1.73654 w_k1_KO_s110 + 1.6331 w_k1_MAT_s110
                                 + 1.69871 w_k1_MSFT_s110
                                 + 1.44187 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.815897 w_k1_ADM_s110
                                 + 0.562228 w_k1_AXP_s110
                                 + 0.500373 w_k1_DIS_s110
                                 + 0.612995 w_k1_FDX_s110
                                 + 0.787765 w_k1_IBM_s110
                                 + 0.704965 w_k1_INTC_s110
                                 + 0.741835 w_k1_KO_s110
                                 + 0.698747 w_k1_MAT_s110
                                 + 0.677643 w_k1_MSFT_s110
                                 + 0.800003 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.09807 w_k1_ADM_s110
                                 + 0.859182 w_k1_AXP_s110
                                 + 0.976599 w_k1_DIS_s110
                                 + 1.00074 w_k1_FDX_s110
                                 + 0.829994 w_k1_IBM_s110
                                 + 1.06482 w_k1_INTC_s110
                                 + 1.03861 w_k1_KO_s110
                                 + 0.925905 w_k1_MAT_s110
                                 + 1.08925 w_k1_MSFT_s110
                                 + 0.951321 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.34969 w_k1_ADM_s110
                                 + 1.24054 w_k1_AXP_s110
                                 + 1.39779 w_k1_DIS_s110
                                 + 1.09182 w_k1_FDX_s110
                                 + 1.49559 w_k1_IBM_s110
                                 + 1.09693 w_k1_INTC_s110
                                 + 1.57797 w_k1_KO_s110 + 1.43948 w_k1_MAT_s110
                                 + 1.14586 w_k1_MSFT_s110
                                 + 1.11665 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.935978 w_k1_ADM_s111
                                 + 1.06933 w_k1_AXP_s111
                                 + 1.26528 w_k1_DIS_s111
                                 + 1.12464 w_k1_FDX_s111
                                 + 0.861363 w_k1_IBM_s111
                                 + 1.10038 w_k1_INTC_s111
                                 + 1.26759 w_k1_KO_s111
                                 + 0.996759 w_k1_MAT_s111
                                 + 1.17664 w_k1_MSFT_s111 + 1.031 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.883125 w_k1_ADM_s111
                                 + 0.94941 w_k1_AXP_s111
                                 + 0.94538 w_k1_DIS_s111
                                 + 0.901398 w_k1_FDX_s111
                                 + 1.04487 w_k1_IBM_s111
                                 + 0.884932 w_k1_INTC_s111
                                 + 1.14122 w_k1_KO_s111 + 1.20674 w_k1_MAT_s111
                                 + 0.929125 w_k1_MSFT_s111
                                 + 1.11449 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.94363 w_k1_ADM_s111
                                 + 1.10658 w_k1_AXP_s111
                                 + 0.978957 w_k1_DIS_s111
                                 + 0.988735 w_k1_FDX_s111
                                 + 1.23297 w_k1_IBM_s111
                                 + 0.921029 w_k1_INTC_s111
                                 + 1.08525 w_k1_KO_s111 + 1.38502 w_k1_MAT_s111
                                 + 0.96812 w_k1_MSFT_s111
                                 + 1.01826 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.01199 w_k1_ADM_s111
                                 + 1.5942 w_k1_AXP_s111 + 1.73836 w_k1_DIS_s111
                                 + 1.4215 w_k1_FDX_s111 + 1.09982 w_k1_IBM_s111
                                 + 1.40015 w_k1_INTC_s111
                                 + 1.51991 w_k1_KO_s111 + 1.45969 w_k1_MAT_s111
                                 + 1.36355 w_k1_MSFT_s111
                                 + 1.27639 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.746127 w_k1_ADM_s111
                                 + 0.760776 w_k1_AXP_s111
                                 + 0.751331 w_k1_DIS_s111
                                 + 0.720311 w_k1_FDX_s111
                                 + 0.792506 w_k1_IBM_s111
                                 + 0.630154 w_k1_INTC_s111
                                 + 0.878429 w_k1_KO_s111
                                 + 0.986482 w_k1_MAT_s111
                                 + 0.747942 w_k1_MSFT_s111
                                 + 0.87406 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.952546 w_k1_ADM_s111
                                 + 1.16777 w_k1_AXP_s111
                                 + 1.35932 w_k1_DIS_s111
                                 + 1.11966 w_k1_FDX_s111
                                 + 1.14398 w_k1_IBM_s111
                                 + 1.05717 w_k1_INTC_s111
                                 + 1.33064 w_k1_KO_s111 + 1.15855 w_k1_MAT_s111
                                 + 1.03444 w_k1_MSFT_s111
                                 + 1.06582 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.04005 w_k1_ADM_s111
                                 + 1.2415 w_k1_AXP_s111 + 1.03404 w_k1_DIS_s111
                                 + 1.12127 w_k1_FDX_s111
                                 + 0.682747 w_k1_IBM_s111
                                 + 0.761036 w_k1_INTC_s111
                                 + 0.936135 w_k1_KO_s111 + 1.1078 w_k1_MAT_s111
                                 + 0.925935 w_k1_MSFT_s111
                                 + 1.05455 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.16771 w_k1_ADM_s111
                                 + 1.31152 w_k1_AXP_s111
                                 + 1.53107 w_k1_DIS_s111
                                 + 1.04895 w_k1_FDX_s111
                                 + 1.74076 w_k1_IBM_s111
                                 + 1.00319 w_k1_INTC_s111
                                 + 1.74344 w_k1_KO_s111 + 2.23281 w_k1_MAT_s111
                                 + 1.13171 w_k1_MSFT_s111
                                 + 1.29507 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.0469 w_k1_ADM_s112
                                 + 0.950609 w_k1_AXP_s112
                                 + 0.945071 w_k1_DIS_s112
                                 + 0.922718 w_k1_FDX_s112
                                 + 0.776118 w_k1_IBM_s112
                                 + 1.08706 w_k1_INTC_s112
                                 + 0.900006 w_k1_KO_s112
                                 + 0.607456 w_k1_MAT_s112
                                 + 0.983785 w_k1_MSFT_s112
                                 + 0.910321 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.917293 w_k1_ADM_s112
                                 + 1.25892 w_k1_AXP_s112
                                 + 0.884445 w_k1_DIS_s112
                                 + 0.87811 w_k1_FDX_s112
                                 + 1.22075 w_k1_IBM_s112 + 1.034 w_k1_INTC_s112
                                 + 0.872134 w_k1_KO_s112
                                 + 1.33873 w_k1_MAT_s112
                                 + 0.892492 w_k1_MSFT_s112
                                 + 1.00973 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.835342 w_k1_ADM_s112
                                 + 0.819448 w_k1_AXP_s112
                                 + 0.645435 w_k1_DIS_s112
                                 + 0.823556 w_k1_FDX_s112
                                 + 0.596289 w_k1_IBM_s112
                                 + 0.908178 w_k1_INTC_s112
                                 + 0.606319 w_k1_KO_s112
                                 + 0.533399 w_k1_MAT_s112
                                 + 0.816875 w_k1_MSFT_s112
                                 + 0.800102 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.862553 w_k1_ADM_s112
                                 + 0.965264 w_k1_AXP_s112
                                 + 0.937989 w_k1_DIS_s112
                                 + 0.900218 w_k1_FDX_s112
                                 + 1.00058 w_k1_IBM_s112
                                 + 1.02813 w_k1_INTC_s112
                                 + 0.951563 w_k1_KO_s112
                                 + 0.863082 w_k1_MAT_s112
                                 + 0.895228 w_k1_MSFT_s112
                                 + 0.986944 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.827593 w_k1_ADM_s112
                                 + 0.945426 w_k1_AXP_s112 + 0.598 w_k1_DIS_s112
                                 + 0.797058 w_k1_FDX_s112
                                 + 0.837315 w_k1_IBM_s112
                                 + 0.978893 w_k1_INTC_s112
                                 + 0.724163 w_k1_KO_s112
                                 + 0.864442 w_k1_MAT_s112
                                 + 0.795675 w_k1_MSFT_s112
                                 + 0.805327 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.03382 w_k1_ADM_s112
                                 + 1.0135 w_k1_AXP_s112 + 0.97235 w_k1_DIS_s112
                                 + 0.950436 w_k1_FDX_s112 + 1.212 w_k1_IBM_s112
                                 + 0.987128 w_k1_INTC_s112
                                 + 1.00115 w_k1_KO_s112 + 1.22043 w_k1_MAT_s112
                                 + 0.993507 w_k1_MSFT_s112
                                 + 0.981192 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.26248 w_k1_ADM_s112
                                 + 1.1594 w_k1_AXP_s112 + 1.33737 w_k1_DIS_s112
                                 + 1.17945 w_k1_FDX_s112
                                 + 1.27499 w_k1_IBM_s112
                                 + 0.982451 w_k1_INTC_s112
                                 + 1.09192 w_k1_KO_s112 + 0.96088 w_k1_MAT_s112
                                 + 1.04158 w_k1_MSFT_s112
                                 + 1.10444 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.41548 w_k1_ADM_s112
                                 + 1.76457 w_k1_AXP_s112
                                 + 1.70324 w_k1_DIS_s112
                                 + 1.33671 w_k1_FDX_s112
                                 + 1.29464 w_k1_IBM_s112
                                 + 1.26073 w_k1_INTC_s112
                                 + 1.46049 w_k1_KO_s112 + 1.29455 w_k1_MAT_s112
                                 + 1.27427 w_k1_MSFT_s112
                                 + 1.15411 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.874083 w_k1_ADM_s113
                                 + 0.89159 w_k1_AXP_s113
                                 + 0.839142 w_k1_DIS_s113
                                 + 0.874744 w_k1_FDX_s113
                                 + 0.895492 w_k1_IBM_s113
                                 + 0.976496 w_k1_INTC_s113
                                 + 0.772245 w_k1_KO_s113
                                 + 0.978738 w_k1_MAT_s113
                                 + 1.09749 w_k1_MSFT_s113
                                 + 0.96783 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.835844 w_k1_ADM_s113
                                 + 0.711019 w_k1_AXP_s113
                                 + 0.506256 w_k1_DIS_s113
                                 + 0.768924 w_k1_FDX_s113
                                 + 0.666294 w_k1_IBM_s113
                                 + 0.853812 w_k1_INTC_s113
                                 + 0.578251 w_k1_KO_s113
                                 + 0.654957 w_k1_MAT_s113
                                 + 0.811071 w_k1_MSFT_s113
                                 + 0.789107 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.863906 w_k1_ADM_s113
                                 + 1.01114 w_k1_AXP_s113
                                 + 1.10314 w_k1_DIS_s113
                                 + 0.984733 w_k1_FDX_s113
                                 + 1.20129 w_k1_IBM_s113
                                 + 1.00059 w_k1_INTC_s113 + 1.0946 w_k1_KO_s113
                                 + 1.3376 w_k1_MAT_s113
                                 + 1.05764 w_k1_MSFT_s113
                                 + 1.00945 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.963228 w_k1_ADM_s113
                                 + 0.777289 w_k1_AXP_s113
                                 + 0.718608 w_k1_DIS_s113
                                 + 0.809541 w_k1_FDX_s113
                                 + 1.1289 w_k1_IBM_s113
                                 + 1.06069 w_k1_INTC_s113
                                 + 0.91339 w_k1_KO_s113 + 0.87592 w_k1_MAT_s113
                                 + 0.950757 w_k1_MSFT_s113
                                 + 0.875622 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.30204 w_k1_ADM_s113
                                 + 1.14978 w_k1_AXP_s113
                                 + 1.30062 w_k1_DIS_s113
                                 + 1.25143 w_k1_FDX_s113
                                 + 1.28995 w_k1_IBM_s113
                                 + 1.15688 w_k1_INTC_s113 + 1.2943 w_k1_KO_s113
                                 + 1.3446 w_k1_MAT_s113 + 1.2121 w_k1_MSFT_s113
                                 + 1.22495 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.90553 w_k1_ADM_s113
                                 + 0.895984 w_k1_AXP_s113
                                 + 1.02073 w_k1_DIS_s113
                                 + 0.949471 w_k1_FDX_s113
                                 + 1.29183 w_k1_IBM_s113
                                 + 1.1586 w_k1_INTC_s113 + 1.10498 w_k1_KO_s113
                                 + 1.31671 w_k1_MAT_s113
                                 + 1.08759 w_k1_MSFT_s113
                                 + 1.20009 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.15476 w_k1_ADM_s113
                                 + 1.27496 w_k1_AXP_s113
                                 + 1.56402 w_k1_DIS_s113
                                 + 1.26365 w_k1_FDX_s113
                                 + 1.18641 w_k1_IBM_s113
                                 + 1.48325 w_k1_INTC_s113
                                 + 1.41284 w_k1_KO_s113 + 1.29996 w_k1_MAT_s113
                                 + 1.52847 w_k1_MSFT_s113
                                 + 1.29603 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.14904 w_k1_ADM_s113
                                 + 0.999115 w_k1_AXP_s113
                                 + 1.08095 w_k1_DIS_s113
                                 + 1.18861 w_k1_FDX_s113
                                 + 0.728553 w_k1_IBM_s113
                                 + 1.23816 w_k1_INTC_s113
                                 + 0.989241 w_k1_KO_s113
                                 + 0.844437 w_k1_MAT_s113
                                 + 1.20663 w_k1_MSFT_s113
                                 + 1.03031 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.11392 w_k1_ADM_s114
                                 + 0.900123 w_k1_AXP_s114
                                 + 1.02357 w_k1_DIS_s114
                                 + 1.09324 w_k1_FDX_s114
                                 + 0.792246 w_k1_IBM_s114
                                 + 0.895311 w_k1_INTC_s114
                                 + 0.88837 w_k1_KO_s114 + 1.04699 w_k1_MAT_s114
                                 + 1.1554 w_k1_MSFT_s114
                                 + 0.979015 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.52533 w_k1_ADM_s114
                                 + 1.22735 w_k1_AXP_s114
                                 + 1.39372 w_k1_DIS_s114
                                 + 1.26641 w_k1_FDX_s114
                                 + 1.25256 w_k1_IBM_s114
                                 + 1.02886 w_k1_INTC_s114
                                 + 1.36563 w_k1_KO_s114 + 1.32056 w_k1_MAT_s114
                                 + 1.26897 w_k1_MSFT_s114
                                 + 1.14724 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.0019 w_k1_ADM_s114
                                 + 0.623525 w_k1_AXP_s114
                                 + 0.619067 w_k1_DIS_s114
                                 + 0.745771 w_k1_FDX_s114
                                 + 0.685835 w_k1_IBM_s114
                                 + 0.691049 w_k1_INTC_s114
                                 + 0.811242 w_k1_KO_s114
                                 + 0.569997 w_k1_MAT_s114
                                 + 0.872627 w_k1_MSFT_s114
                                 + 0.831175 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.837134 w_k1_ADM_s114
                                 + 0.859098 w_k1_AXP_s114
                                 + 0.80678 w_k1_DIS_s114
                                 + 0.785133 w_k1_FDX_s114
                                 + 1.34515 w_k1_IBM_s114
                                 + 0.762789 w_k1_INTC_s114
                                 + 1.17008 w_k1_KO_s114 + 1.43016 w_k1_MAT_s114
                                 + 0.882986 w_k1_MSFT_s114
                                 + 1.04469 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.928372 w_k1_ADM_s114
                                 + 0.883721 w_k1_AXP_s114
                                 + 0.986785 w_k1_DIS_s114
                                 + 0.96568 w_k1_FDX_s114
                                 + 0.946547 w_k1_IBM_s114
                                 + 0.95721 w_k1_INTC_s114
                                 + 1.01925 w_k1_KO_s114 + 1.12266 w_k1_MAT_s114
                                 + 1.08322 w_k1_MSFT_s114
                                 + 0.965334 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.16239 w_k1_ADM_s114
                                 + 1.09174 w_k1_AXP_s114 + 1.0768 w_k1_DIS_s114
                                 + 1.08615 w_k1_FDX_s114
                                 + 1.28061 w_k1_IBM_s114
                                 + 1.0067 w_k1_INTC_s114 + 1.11056 w_k1_KO_s114
                                 + 0.958546 w_k1_MAT_s114
                                 + 1.00996 w_k1_MSFT_s114 + 0.9768 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.664462 w_k1_ADM_s114
                                 + 0.819756 w_k1_AXP_s114
                                 + 0.704127 w_k1_DIS_s114
                                 + 0.793676 w_k1_FDX_s114
                                 + 0.977323 w_k1_IBM_s114
                                 + 0.842158 w_k1_INTC_s114
                                 + 0.804721 w_k1_KO_s114
                                 + 0.986514 w_k1_MAT_s114
                                 + 0.915022 w_k1_MSFT_s114
                                 + 0.882676 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.56118 w_k1_ADM_s114
                                 + 1.72814 w_k1_AXP_s114 + 1.8417 w_k1_DIS_s114
                                 + 1.50329 w_k1_FDX_s114
                                 + 1.42773 w_k1_IBM_s114
                                 + 1.13887 w_k1_INTC_s114
                                 + 1.39297 w_k1_KO_s114 + 1.78716 w_k1_MAT_s114
                                 + 1.65051 w_k1_MSFT_s114 + 1.408 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.02514 w_k1_ADM_s115
                                 + 1.28097 w_k1_AXP_s115
                                 + 1.26215 w_k1_DIS_s115
                                 + 1.14444 w_k1_FDX_s115 + 1.5302 w_k1_IBM_s115
                                 + 1.21419 w_k1_INTC_s115
                                 + 1.43501 w_k1_KO_s115 + 1.21527 w_k1_MAT_s115
                                 + 1.27478 w_k1_MSFT_s115
                                 + 1.23054 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.695973 w_k1_ADM_s115
                                 + 0.526762 w_k1_AXP_s115
                                 + 0.471214 w_k1_DIS_s115
                                 + 0.7181 w_k1_FDX_s115
                                 + 0.523646 w_k1_IBM_s115
                                 + 0.788944 w_k1_INTC_s115
                                 + 0.60596 w_k1_KO_s115
                                 + 0.296049 w_k1_MAT_s115
                                 + 0.727755 w_k1_MSFT_s115
                                 + 0.781724 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.08409 w_k1_ADM_s115
                                 + 1.07685 w_k1_AXP_s115
                                 + 1.16242 w_k1_DIS_s115
                                 + 1.27092 w_k1_FDX_s115
                                 + 0.861698 w_k1_IBM_s115
                                 + 1.23737 w_k1_INTC_s115
                                 + 1.04928 w_k1_KO_s115
                                 + 0.671021 w_k1_MAT_s115
                                 + 1.19642 w_k1_MSFT_s115 + 1.0251 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.01578 w_k1_ADM_s115
                                 + 1.07057 w_k1_AXP_s115
                                 + 1.09105 w_k1_DIS_s115
                                 + 1.14727 w_k1_FDX_s115
                                 + 1.12181 w_k1_IBM_s115
                                 + 1.05192 w_k1_INTC_s115
                                 + 1.12585 w_k1_KO_s115 + 1.12534 w_k1_MAT_s115
                                 + 1.23512 w_k1_MSFT_s115
                                 + 1.15209 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.844096 w_k1_ADM_s115
                                 + 0.877423 w_k1_AXP_s115
                                 + 0.679009 w_k1_DIS_s115
                                 + 0.846418 w_k1_FDX_s115
                                 + 0.894601 w_k1_IBM_s115
                                 + 0.863715 w_k1_INTC_s115
                                 + 0.95252 w_k1_KO_s115 + 0.87451 w_k1_MAT_s115
                                 + 0.917651 w_k1_MSFT_s115
                                 + 0.901954 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.32791 w_k1_ADM_s115
                                 + 1.76056 w_k1_AXP_s115
                                 + 1.83649 w_k1_DIS_s115
                                 + 1.62444 w_k1_FDX_s115
                                 + 1.39468 w_k1_IBM_s115
                                 + 1.38752 w_k1_INTC_s115
                                 + 1.75901 w_k1_KO_s115 + 1.52402 w_k1_MAT_s115
                                 + 1.59863 w_k1_MSFT_s115
                                 + 1.44226 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.762571 w_k1_ADM_s115
                                 + 0.756307 w_k1_AXP_s115
                                 + 0.708461 w_k1_DIS_s115
                                 + 0.90833 w_k1_FDX_s115
                                 + 0.653519 w_k1_IBM_s115
                                 + 0.914727 w_k1_INTC_s115
                                 + 0.844176 w_k1_KO_s115
                                 + 0.564231 w_k1_MAT_s115
                                 + 0.969366 w_k1_MSFT_s115
                                 + 0.814083 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.96582 w_k1_ADM_s115
                                 + 0.944719 w_k1_AXP_s115
                                 + 1.09981 w_k1_DIS_s115
                                 + 0.922665 w_k1_FDX_s115
                                 + 1.67459 w_k1_IBM_s115
                                 + 0.871163 w_k1_INTC_s115
                                 + 1.38848 w_k1_KO_s115 + 1.60902 w_k1_MAT_s115
                                 + 1.12901 w_k1_MSFT_s115
                                 + 1.04555 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.876911 w_k1_ADM_s116
                                 + 0.701499 w_k1_AXP_s116
                                 + 0.796665 w_k1_DIS_s116
                                 + 0.789004 w_k1_FDX_s116
                                 + 0.94908 w_k1_IBM_s116
                                 + 0.894621 w_k1_INTC_s116
                                 + 0.844084 w_k1_KO_s116
                                 + 0.603471 w_k1_MAT_s116
                                 + 0.892157 w_k1_MSFT_s116
                                 + 0.917912 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.4183 w_k1_ADM_s116
                                 + 1.66188 w_k1_AXP_s116
                                 + 2.21701 w_k1_DIS_s116
                                 + 1.57149 w_k1_FDX_s116
                                 + 1.60515 w_k1_IBM_s116
                                 + 1.21261 w_k1_INTC_s116
                                 + 1.80403 w_k1_KO_s116 + 1.46987 w_k1_MAT_s116
                                 + 1.35743 w_k1_MSFT_s116
                                 + 1.54126 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.22011 w_k1_ADM_s116
                                 + 1.02054 w_k1_AXP_s116
                                 + 0.956938 w_k1_DIS_s116
                                 + 1.0233 w_k1_FDX_s116 + 1.13826 w_k1_IBM_s116
                                 + 1.08932 w_k1_INTC_s116
                                 + 1.02503 w_k1_KO_s116
                                 + 0.849818 w_k1_MAT_s116
                                 + 1.03304 w_k1_MSFT_s116
                                 + 1.18293 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.911648 w_k1_ADM_s116
                                 + 1.16487 w_k1_AXP_s116
                                 + 1.15632 w_k1_DIS_s116
                                 + 0.982029 w_k1_FDX_s116 + 1.139 w_k1_IBM_s116
                                 + 1.16351 w_k1_INTC_s116
                                 + 1.03068 w_k1_KO_s116
                                 + 0.919694 w_k1_MAT_s116
                                 + 1.0908 w_k1_MSFT_s116 + 1.08269 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.746846 w_k1_ADM_s116
                                 + 0.538844 w_k1_AXP_s116
                                 + 0.506748 w_k1_DIS_s116
                                 + 0.673314 w_k1_FDX_s116
                                 + 0.66038 w_k1_IBM_s116
                                 + 0.732714 w_k1_INTC_s116
                                 + 0.593425 w_k1_KO_s116
                                 + 0.492061 w_k1_MAT_s116
                                 + 0.677989 w_k1_MSFT_s116
                                 + 0.836055 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.19977 w_k1_ADM_s116
                                 + 1.29636 w_k1_AXP_s116
                                 + 1.39791 w_k1_DIS_s116
                                 + 1.20725 w_k1_FDX_s116
                                 + 1.39231 w_k1_IBM_s116
                                 + 1.23089 w_k1_INTC_s116 + 1.0957 w_k1_KO_s116
                                 + 1.27362 w_k1_MAT_s116
                                 + 1.3106 w_k1_MSFT_s116 + 1.13572 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.781038 w_k1_ADM_s116
                                 + 0.717667 w_k1_AXP_s116
                                 + 0.795672 w_k1_DIS_s116
                                 + 0.894684 w_k1_FDX_s116
                                 + 0.692438 w_k1_IBM_s116
                                 + 0.992043 w_k1_INTC_s116
                                 + 0.739056 w_k1_KO_s116
                                 + 0.57669 w_k1_MAT_s116
                                 + 0.879511 w_k1_MSFT_s116
                                 + 0.938765 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.04068 w_k1_ADM_s116
                                 + 1.06498 w_k1_AXP_s116
                                 + 0.844655 w_k1_DIS_s116
                                 + 1.12326 w_k1_FDX_s116
                                 + 0.648992 w_k1_IBM_s116
                                 + 1.34207 w_k1_INTC_s116
                                 + 0.860046 w_k1_KO_s116
                                 + 0.753285 w_k1_MAT_s116
                                 + 0.995806 w_k1_MSFT_s116
                                 + 1.03838 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.926061 w_k1_ADM_s117
                                 + 0.958151 w_k1_AXP_s117
                                 + 0.62766 w_k1_DIS_s117
                                 + 0.947247 w_k1_FDX_s117
                                 + 0.785626 w_k1_IBM_s117
                                 + 0.967075 w_k1_INTC_s117
                                 + 0.694697 w_k1_KO_s117
                                 + 0.90168 w_k1_MAT_s117
                                 + 0.849483 w_k1_MSFT_s117
                                 + 0.954938 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.913395 w_k1_ADM_s117
                                 + 1.17867 w_k1_AXP_s117
                                 + 1.38968 w_k1_DIS_s117
                                 + 1.19097 w_k1_FDX_s117 + 1.5699 w_k1_IBM_s117
                                 + 1.17528 w_k1_INTC_s117
                                 + 1.31473 w_k1_KO_s117 + 1.49534 w_k1_MAT_s117
                                 + 1.07873 w_k1_MSFT_s117 + 1.1453 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.864368 w_k1_ADM_s117
                                 + 0.629173 w_k1_AXP_s117
                                 + 0.783777 w_k1_DIS_s117
                                 + 0.913822 w_k1_FDX_s117
                                 + 0.717745 w_k1_IBM_s117
                                 + 0.882484 w_k1_INTC_s117
                                 + 0.766151 w_k1_KO_s117
                                 + 0.544569 w_k1_MAT_s117
                                 + 0.974218 w_k1_MSFT_s117
                                 + 0.876548 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.903027 w_k1_ADM_s117
                                 + 0.925519 w_k1_AXP_s117
                                 + 0.972262 w_k1_DIS_s117
                                 + 0.921637 w_k1_FDX_s117
                                 + 1.28228 w_k1_IBM_s117
                                 + 0.805717 w_k1_INTC_s117
                                 + 1.04731 w_k1_KO_s117 + 1.23519 w_k1_MAT_s117
                                 + 0.803082 w_k1_MSFT_s117
                                 + 0.945137 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.00672 w_k1_ADM_s117
                                 + 0.98206 w_k1_AXP_s117
                                 + 0.998017 w_k1_DIS_s117
                                 + 1.13735 w_k1_FDX_s117
                                 + 0.75465 w_k1_IBM_s117
                                 + 0.934586 w_k1_INTC_s117
                                 + 0.839689 w_k1_KO_s117
                                 + 0.632314 w_k1_MAT_s117
                                 + 1.06761 w_k1_MSFT_s117
                                 + 0.899784 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.17663 w_k1_ADM_s117
                                 + 0.996776 w_k1_AXP_s117
                                 + 1.19208 w_k1_DIS_s117
                                 + 1.16962 w_k1_FDX_s117
                                 + 1.05987 w_k1_IBM_s117
                                 + 1.31943 w_k1_INTC_s117 + 1.2092 w_k1_KO_s117
                                 + 0.990976 w_k1_MAT_s117
                                 + 1.22231 w_k1_MSFT_s117 + 1.0923 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.29798 w_k1_ADM_s117
                                 + 1.24073 w_k1_AXP_s117
                                 + 1.56131 w_k1_DIS_s117
                                 + 1.44021 w_k1_FDX_s117
                                 + 1.01483 w_k1_IBM_s117
                                 + 1.5654 w_k1_INTC_s117 + 1.2516 w_k1_KO_s117
                                 + 0.685668 w_k1_MAT_s117
                                 + 1.2703 w_k1_MSFT_s117 + 1.12557 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.849714 w_k1_ADM_s117
                                 + 0.884042 w_k1_AXP_s117
                                 + 0.722525 w_k1_DIS_s117
                                 + 0.859744 w_k1_FDX_s117
                                 + 0.948197 w_k1_IBM_s117
                                 + 0.849032 w_k1_INTC_s117
                                 + 0.855959 w_k1_KO_s117
                                 + 0.785476 w_k1_MAT_s117
                                 + 0.81852 w_k1_MSFT_s117
                                 + 0.885351 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0466971018798343 w_k1_ADM_s000
                                 + 0.0166033094338251 w_k1_ADM_s001
                                 + 0.0277855710453453 w_k1_ADM_s002
                                 + 0.0229640033388007 w_k1_ADM_s003
                                 + 0.0164096979440612 w_k1_ADM_s004
                                 + 0.0556896325646571 w_k1_ADM_s005
                                 + 0.0379486169110528 w_k1_ADM_s006
                                 + 0.0428694659973901 w_k1_ADM_s007
                                 + 0.0206511373207736 w_k1_ADM_s010
                                 + 0.00613513097531705 w_k1_ADM_s011
                                 + 0.0185619145670788 w_k1_ADM_s012
                                 + 0.0075631187147957 w_k1_ADM_s013
                                 + 0.00788969813488973 w_k1_ADM_s014
                                 + 0.0181547284788556 w_k1_ADM_s015
                                 + 0.0147144681116701 w_k1_ADM_s016
                                 + 0.0184230442464476 w_k1_ADM_s017
                                 + 0.0203271180959296 w_k1_ADM_s100
                                 + 0.0281351987836588 w_k1_ADM_s101
                                 + 0.0446608336544094 w_k1_ADM_s102
                                 + 0.0281718467810158 w_k1_ADM_s103
                                 + 0.0388578623963911 w_k1_ADM_s104
                                 + 0.0186161896626814 w_k1_ADM_s105
                                 + 0.0128931964876368 w_k1_ADM_s106
                                 + 0.0174732763697654 w_k1_ADM_s107
                                 + 0.07153956782502 w_k1_ADM_s110
                                 + 0.0469371793913348 w_k1_ADM_s111
                                 + 0.0654041331430263 w_k1_ADM_s112
                                 + 0.0232943254701703 w_k1_ADM_s113
                                 + 0.0384003756145281 w_k1_ADM_s114
                                 + 0.0893870946792831 w_k1_ADM_s115
                                 + 0.0884556407191259 w_k1_ADM_s116
                                 + 0.0222621057101822 w_k1_ADM_s117
                                 + 0.0379970734708247 w_k1_AXP_s000
                                 + 0.0152357025771056 w_k1_AXP_s001
                                 + 0.0279520631965429 w_k1_AXP_s002
                                 + 0.0247760752206239 w_k1_AXP_s003
                                 + 0.0170940859587566 w_k1_AXP_s004
                                 + 0.0543146159652666 w_k1_AXP_s005
                                 + 0.0393973110322375 w_k1_AXP_s006
                                 + 0.0343737529285714 w_k1_AXP_s007
                                 + 0.0181525119327692 w_k1_AXP_s010
                                 + 0.00702648709556648 w_k1_AXP_s011
                                 + 0.0178602877396577 w_k1_AXP_s012
                                 + 0.00726540140377674 w_k1_AXP_s013
                                 + 0.00707466741559187 w_k1_AXP_s014
                                 + 0.0145893367056911 w_k1_AXP_s015
                                 + 0.015131412821538 w_k1_AXP_s016
                                 + 0.0182952463133114 w_k1_AXP_s017
                                 + 0.0240681718121645 w_k1_AXP_s100
                                 + 0.0235923683850923 w_k1_AXP_s101
                                 + 0.0483318238848996 w_k1_AXP_s102
                                 + 0.0333705347802192 w_k1_AXP_s103
                                 + 0.034528713716626 w_k1_AXP_s104
                                 + 0.0212349962182187 w_k1_AXP_s105
                                 + 0.0123483699045813 w_k1_AXP_s106
                                 + 0.0189621527483768 w_k1_AXP_s107
                                 + 0.0625144743431262 w_k1_AXP_s110
                                 + 0.0549530228025223 w_k1_AXP_s111
                                 + 0.0685986371902945 w_k1_AXP_s112
                                 + 0.0224273966731962 w_k1_AXP_s113
                                 + 0.0350900517306162 w_k1_AXP_s114
                                 + 0.0955756542590871 w_k1_AXP_s115
                                 + 0.0885862372742471 w_k1_AXP_s116
                                 + 0.0220940572764346 w_k1_AXP_s117
                                 + 0.0399012920740278 w_k1_DIS_s000
                                 + 0.0164569252913205 w_k1_DIS_s001
                                 + 0.0275687600725768 w_k1_DIS_s002
                                 + 0.0241198840707664 w_k1_DIS_s003
                                 + 0.0164434697119854 w_k1_DIS_s004
                                 + 0.0556194313906464 w_k1_DIS_s005
                                 + 0.041688425982426 w_k1_DIS_s006
                                 + 0.0376627631507037 w_k1_DIS_s007
                                 + 0.01789904396164 w_k1_DIS_s010
                                 + 0.00749666338321078 w_k1_DIS_s011
                                 + 0.0197305361828855 w_k1_DIS_s012
                                 + 0.00804606001905081 w_k1_DIS_s013
                                 + 0.00797578398000255 w_k1_DIS_s014
                                 + 0.0146084758051922 w_k1_DIS_s015
                                 + 0.0155349451747292 w_k1_DIS_s016
                                 + 0.0196747180134229 w_k1_DIS_s017
                                 + 0.0214445335582057 w_k1_DIS_s100
                                 + 0.0262148070775883 w_k1_DIS_s101
                                 + 0.0429623947309187 w_k1_DIS_s102
                                 + 0.031553869027411 w_k1_DIS_s103
                                 + 0.03549027474271 w_k1_DIS_s104
                                 + 0.0208931422644421 w_k1_DIS_s105
                                 + 0.0134152296606715 w_k1_DIS_s106
                                 + 0.018712136295206 w_k1_DIS_s107
                                 + 0.0688074276654048 w_k1_DIS_s110
                                 + 0.0558002776767977 w_k1_DIS_s111
                                 + 0.0653648963122476 w_k1_DIS_s112
                                 + 0.0234729922746593 w_k1_DIS_s113
                                 + 0.0363848821456339 w_k1_DIS_s114
                                 + 0.094530957663765 w_k1_DIS_s115
                                 + 0.0936098656899366 w_k1_DIS_s116
                                 + 0.0230116244502484 w_k1_DIS_s117
                                 + 0.0390571644168933 w_k1_FDX_s000
                                 + 0.0151196578544512 w_k1_FDX_s001
                                 + 0.0267086933299091 w_k1_FDX_s002
                                 + 0.0221072688923857 w_k1_FDX_s003
                                 + 0.0161822335660111 w_k1_FDX_s004
                                 + 0.0530149876066713 w_k1_FDX_s005
                                 + 0.0392801536643441 w_k1_FDX_s006
                                 + 0.0364173168987365 w_k1_FDX_s007
                                 + 0.0175945410937042 w_k1_FDX_s010
                                 + 0.00748597149454509 w_k1_FDX_s011
                                 + 0.0177645220462144 w_k1_FDX_s012
                                 + 0.00776267715221533 w_k1_FDX_s013
                                 + 0.00786763488609151 w_k1_FDX_s014
                                 + 0.015234885735978 w_k1_FDX_s015
                                 + 0.0141421583832467 w_k1_FDX_s016
                                 + 0.0197384978277416 w_k1_FDX_s017
                                 + 0.0220445220092652 w_k1_FDX_s100
                                 + 0.0258524517939707 w_k1_FDX_s101
                                 + 0.0461037532171629 w_k1_FDX_s102
                                 + 0.029799269321742 w_k1_FDX_s103
                                 + 0.0326022376635437 w_k1_FDX_s104
                                 + 0.0211216473556021 w_k1_FDX_s105
                                 + 0.0134473089507435 w_k1_FDX_s106
                                 + 0.0194101313494199 w_k1_FDX_s107
                                 + 0.0643478220126574 w_k1_FDX_s110
                                 + 0.0506156948482063 w_k1_FDX_s111
                                 + 0.061860079267654 w_k1_FDX_s112
                                 + 0.023416200295199 w_k1_FDX_s113
                                 + 0.0364843546499859 w_k1_FDX_s114
                                 + 0.0999710990077588 w_k1_FDX_s115
                                 + 0.088973144631973 w_k1_FDX_s116
                                 + 0.023639745502219 w_k1_FDX_s117
                                 + 0.039429445894203 w_k1_IBM_s000
                                 + 0.015348130034541 w_k1_IBM_s001
                                 + 0.0276226810082101 w_k1_IBM_s002
                                 + 0.0234591691384414 w_k1_IBM_s003
                                 + 0.0162219822861039 w_k1_IBM_s004
                                 + 0.055893805639997 w_k1_IBM_s005
                                 + 0.0406201774094833 w_k1_IBM_s006
                                 + 0.0357289971550853 w_k1_IBM_s007
                                 + 0.0169531685570227 w_k1_IBM_s010
                                 + 0.00742260502630631 w_k1_IBM_s011
                                 + 0.0183734584742836 w_k1_IBM_s012
                                 + 0.00780969583320874 w_k1_IBM_s013
                                 + 0.00789126807607431 w_k1_IBM_s014
                                 + 0.0156559830608925 w_k1_IBM_s015
                                 + 0.015233061002382 w_k1_IBM_s016
                                 + 0.0188408307067994 w_k1_IBM_s017
                                 + 0.0224334770048874 w_k1_IBM_s100
                                 + 0.0263177651593857 w_k1_IBM_s101
                                 + 0.0456474421487523 w_k1_IBM_s102
                                 + 0.0300976112729945 w_k1_IBM_s103
                                 + 0.0320469862587627 w_k1_IBM_s104
                                 + 0.022763263239975 w_k1_IBM_s105
                                 + 0.0143518944151906 w_k1_IBM_s106
                                 + 0.0199158270905727 w_k1_IBM_s107
                                 + 0.0674596981843644 w_k1_IBM_s110
                                 + 0.0522140823326432 w_k1_IBM_s111
                                 + 0.0667991746058528 w_k1_IBM_s112
                                 + 0.0240580702009502 w_k1_IBM_s113
                                 + 0.0372472778248646 w_k1_IBM_s114
                                 + 0.095291397617966 w_k1_IBM_s115
                                 + 0.0914625940676548 w_k1_IBM_s116
                                 + 0.024151171222179 w_k1_IBM_s117
                                 + 0.0413428281446626 w_k1_INTC_s000
                                 + 0.0160637743320843 w_k1_INTC_s001
                                 + 0.0264285993310624 w_k1_INTC_s002
                                 + 0.0215538410229881 w_k1_INTC_s003
                                 + 0.0142814762995114 w_k1_INTC_s004
                                 + 0.0542443858903641 w_k1_INTC_s005
                                 + 0.0410067209065621 w_k1_INTC_s006
                                 + 0.0368777744431236 w_k1_INTC_s007
                                 + 0.0154931033287159 w_k1_INTC_s010
                                 + 0.00719618285176803 w_k1_INTC_s011
                                 + 0.0161478746679478 w_k1_INTC_s012
                                 + 0.00764829394445756 w_k1_INTC_s013
                                 + 0.00891457676905896 w_k1_INTC_s014
                                 + 0.0126229864645143 w_k1_INTC_s015
                                 + 0.0132487434453384 w_k1_INTC_s016
                                 + 0.0187124934887242 w_k1_INTC_s017
                                 + 0.0271293075663535 w_k1_INTC_s100
                                 + 0.0238761980780499 w_k1_INTC_s101
                                 + 0.0500185126799619 w_k1_INTC_s102
                                 + 0.0286652817292988 w_k1_INTC_s103
                                 + 0.0334556667154616 w_k1_INTC_s104
                                 + 0.0248333679092807 w_k1_INTC_s105
                                 + 0.0147491056591094 w_k1_INTC_s106
                                 + 0.0173091249944367 w_k1_INTC_s107
                                 + 0.0641021860129626 w_k1_INTC_s110
                                 + 0.046243403480701 w_k1_INTC_s111
                                 + 0.0643770085106522 w_k1_INTC_s112
                                 + 0.0257656188746061 w_k1_INTC_s113
                                 + 0.0330585543926007 w_k1_INTC_s114
                                 + 0.0965289805572142 w_k1_INTC_s115
                                 + 0.0949306545627219 w_k1_INTC_s116
                                 + 0.0237370169135061 w_k1_INTC_s117
                                 + 0.0391561747252435 w_k1_KO_s000
                                 + 0.0151402745842401 w_k1_KO_s001
                                 + 0.0256528387844441 w_k1_KO_s002
                                 + 0.0228720562669053 w_k1_KO_s003
                                 + 0.0156120258685885 w_k1_KO_s004
                                 + 0.0496087785400486 w_k1_KO_s005
                                 + 0.0376439370645462 w_k1_KO_s006
                                 + 0.0380370894489068 w_k1_KO_s007
                                 + 0.0178329326216497 w_k1_KO_s010
                                 + 0.00750916932824713 w_k1_KO_s011
                                 + 0.0181593826169629 w_k1_KO_s012
                                 + 0.0079642330528772 w_k1_KO_s013
                                 + 0.00741697513196728 w_k1_KO_s014
                                 + 0.0154603476113518 w_k1_KO_s015
                                 + 0.0146906289376265 w_k1_KO_s016
                                 + 0.0185683071374054 w_k1_KO_s017
                                 + 0.0206935914179333 w_k1_KO_s100
                                 + 0.0254274418237231 w_k1_KO_s101
                                 + 0.0424379037927861 w_k1_KO_s102
                                 + 0.0313504675301652 w_k1_KO_s103
                                 + 0.0341372228503369 w_k1_KO_s104
                                 + 0.0227711236538377 w_k1_KO_s105
                                 + 0.0133921426160174 w_k1_KO_s106
                                 + 0.0175811836860802 w_k1_KO_s107
                                 + 0.071011608585248 w_k1_KO_s110
                                 + 0.0590258396743242 w_k1_KO_s111
                                 + 0.0611574274321284 w_k1_KO_s112
                                 + 0.0233198490616545 w_k1_KO_s113
                                 + 0.0370160787745001 w_k1_KO_s114
                                 + 0.103200947749692 w_k1_KO_s115
                                 + 0.0858022786406851 w_k1_KO_s116
                                 + 0.0232001825461121 w_k1_KO_s117
                                 + 0.0388494282076183 w_k1_MAT_s000
                                 + 0.0142191825875203 w_k1_MAT_s001
                                 + 0.0230890724711014 w_k1_MAT_s002
                                 + 0.0236529338878752 w_k1_MAT_s003
                                 + 0.0224603671405258 w_k1_MAT_s004
                                 + 0.0560730267858578 w_k1_MAT_s005
                                 + 0.0448458758066668 w_k1_MAT_s006
                                 + 0.0371752067200656 w_k1_MAT_s007
                                 + 0.0175564137770924 w_k1_MAT_s010
                                 + 0.00714390642368571 w_k1_MAT_s011
                                 + 0.0176494179337228 w_k1_MAT_s012
                                 + 0.00866936628410582 w_k1_MAT_s013
                                 + 0.00767417090516263 w_k1_MAT_s014
                                 + 0.0159610682635508 w_k1_MAT_s015
                                 + 0.0152601659074924 w_k1_MAT_s016
                                 + 0.0205470684890767 w_k1_MAT_s017
                                 + 0.0204165642809634 w_k1_MAT_s100
                                 + 0.0278154778397795 w_k1_MAT_s101
                                 + 0.0640911368099518 w_k1_MAT_s102
                                 + 0.0299670056420189 w_k1_MAT_s103
                                 + 0.0301566427095284 w_k1_MAT_s104
                                 + 0.0280556059817879 w_k1_MAT_s105
                                 + 0.0152564519197259 w_k1_MAT_s106
                                 + 0.0179176788671002 w_k1_MAT_s107
                                 + 0.0664127059147498 w_k1_MAT_s110
                                 + 0.0637619095663894 w_k1_MAT_s111
                                 + 0.0602920246758164 w_k1_MAT_s112
                                 + 0.0249742783798078 w_k1_MAT_s113
                                 + 0.0393894961904164 w_k1_MAT_s114
                                 + 0.0881490480336468 w_k1_MAT_s115
                                 + 0.0757003998354765 w_k1_MAT_s116
                                 + 0.0217681696867138 w_k1_MAT_s117
                                 + 0.0376724104995044 w_k1_MSFT_s000
                                 + 0.0147953495089297 w_k1_MSFT_s001
                                 + 0.0256099921331911 w_k1_MSFT_s002
                                 + 0.0226814831765778 w_k1_MSFT_s003
                                 + 0.0158203661917142 w_k1_MSFT_s004
                                 + 0.0547514617242893 w_k1_MSFT_s005
                                 + 0.0437899816252401 w_k1_MSFT_s006
                                 + 0.0370051861459044 w_k1_MSFT_s007
                                 + 0.0172347522764745 w_k1_MSFT_s010
                                 + 0.0078207713370713 w_k1_MSFT_s011
                                 + 0.0193719522719899 w_k1_MSFT_s012
                                 + 0.00788530924286387 w_k1_MSFT_s013
                                 + 0.00757492623414952 w_k1_MSFT_s014
                                 + 0.0139465636882263 w_k1_MSFT_s015
                                 + 0.013091579917054 w_k1_MSFT_s016
                                 + 0.019331772255675 w_k1_MSFT_s017
                                 + 0.0221646744397164 w_k1_MSFT_s100
                                 + 0.0232945030213584 w_k1_MSFT_s101
                                 + 0.0457442215516729 w_k1_MSFT_s102
                                 + 0.0294598823449807 w_k1_MSFT_s103
                                 + 0.0335341900744793 w_k1_MSFT_s104
                                 + 0.0205230655519482 w_k1_MSFT_s105
                                 + 0.0148934761474229 w_k1_MSFT_s106
                                 + 0.0182548481617502 w_k1_MSFT_s107
                                 + 0.0690403583687303 w_k1_MSFT_s110
                                 + 0.0497434177516808 w_k1_MSFT_s111
                                 + 0.0607831010384633 w_k1_MSFT_s112
                                 + 0.0261466992605862 w_k1_MSFT_s113
                                 + 0.0394118907559132 w_k1_MSFT_s114
                                 + 0.104411411769595 w_k1_MSFT_s115
                                 + 0.0910686841286033 w_k1_MSFT_s116
                                 + 0.0223660970407643 w_k1_MSFT_s117
                                 + 0.0381402765874394 w_k1_PG_s000
                                 + 0.0138357204492022 w_k1_PG_s001
                                 + 0.0270999688792895 w_k1_PG_s002
                                 + 0.0244477538914884 w_k1_PG_s003
                                 + 0.0157543942634586 w_k1_PG_s004
                                 + 0.0529144019409397 w_k1_PG_s005
                                 + 0.036711965903582 w_k1_PG_s006
                                 + 0.0396970134529447 w_k1_PG_s007
                                 + 0.0151718556143918 w_k1_PG_s010
                                 + 0.00726792139701245 w_k1_PG_s011
                                 + 0.0174233984239361 w_k1_PG_s012
                                 + 0.00775734644648621 w_k1_PG_s013
                                 + 0.00752414237068527 w_k1_PG_s014
                                 + 0.0148516910894506 w_k1_PG_s015
                                 + 0.0155874856453964 w_k1_PG_s016
                                 + 0.0197591744111127 w_k1_PG_s017
                                 + 0.0217728739545295 w_k1_PG_s100
                                 + 0.0250735564916686 w_k1_PG_s101
                                 + 0.0469067689952231 w_k1_PG_s102
                                 + 0.0312445334077798 w_k1_PG_s103
                                 + 0.0329601206138721 w_k1_PG_s104
                                 + 0.0218363113057315 w_k1_PG_s105
                                 + 0.0134895588996323 w_k1_PG_s106
                                 + 0.0188952455547709 w_k1_PG_s107
                                 + 0.0657123684794004 w_k1_PG_s110
                                 + 0.0541048178726717 w_k1_PG_s111
                                 + 0.0619590646406268 w_k1_PG_s112
                                 + 0.0242656408638751 w_k1_PG_s113
                                 + 0.0363011523591872 w_k1_PG_s114
                                 + 0.0956033228262095 w_k1_PG_s115
                                 + 0.09392839270695 w_k1_PG_s116
                                 + 0.0224517281403972 w_k1_PG_s117 >= 1.229
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
