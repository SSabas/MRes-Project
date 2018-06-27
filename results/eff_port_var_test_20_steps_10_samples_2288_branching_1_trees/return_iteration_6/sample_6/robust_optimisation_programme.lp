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
 w_k1_ADM_s0_return_constraint:  1.13928 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.919417 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.959139 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.13299 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.13563 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.01715 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.977213 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.28719 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.05371 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.807952 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.2503 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.32138 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.916215 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.02679 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.762131 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.42777 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.15915 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.721389 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.931759 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.04634 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.22672 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.945675 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.684831 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.72245 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.84899 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.737991 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.995565 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.83717 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.24633 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.35262 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.850529 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.03409 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.08769 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.641745 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.761212 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.3518 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.00861 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.45581 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23761 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.917527 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.903872 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.33819 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.16729 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.960179 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.18617 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.37373 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.926048 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.63277 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.59834 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.28914 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.917886 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.04205 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.864933 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.18625 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.02167 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.658002 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.735292 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.918618 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.51403 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.10657 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.610448 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.618614 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.932449 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.846949 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.00501 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.885611 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.34395 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.56296 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.703491 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.97238 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.16392 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.558872 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.752312 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.48119 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.994123 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.36811 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38137 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.890864 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.864447 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.32429 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.2235 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.942182 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.10344 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.64131 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.15245 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.637368 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.53751 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.29603 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.819183 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.09773 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.778152 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.90992 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.978033 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.594438 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.03123 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.812053 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.32836 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.29941 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.549252 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.597122 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.904707 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.674266 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.921338 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.742127 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.43169 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.50501 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.692317 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.872771 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.02778 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.497262 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.67243 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.46931 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.357 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.89633 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.25498 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.93459 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.902549 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.16668 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.03593 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.985985 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.06912 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.35269 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.07986 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.690265 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.25889 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.36829 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.898054 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.963062 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.748504 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.5241 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.983506 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.692553 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.933116 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.928092 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.29165 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.01872 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.632336 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.7399 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.866512 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.790153 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.00054 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.890008 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.31852 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.2947 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.781263 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.986299 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.996234 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.588689 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.707627 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.45951 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.16305 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.47593 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15229 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.993013 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.858701 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.33583 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.28866 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.746293 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.12975 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.0869 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.789611 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.960354 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.49633 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.728899 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.830608 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.56997 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.26807 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.934044 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.0272 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.746571 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.769851 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.701358 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.950407 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.63334 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.950701 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.620961 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.38651 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.963714 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.883401 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.705578 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.929864 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.62466 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.755376 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.904133 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.3871 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.719637 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.21701 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.919346 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.980928 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.42125 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.21482 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.911965 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.853774 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.19147 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.06777 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.961516 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.03299 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.2157 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.17914 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.784886 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.30793 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.14817 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.872222 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.950456 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.752791 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.22748 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.05987 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.733854 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.960673 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.783685 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.23004 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.982442 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.707096 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.840611 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.976993 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.93525 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.02403 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.915094 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.25983 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.36564 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.889588 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.0642 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.11186 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.792446 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.880464 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.24201 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.17312 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.34911 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20418 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.952612 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.925538 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.25937 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.1656 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.86541 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.04506 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.10914 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.954936 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.753189 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.28961 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.889965 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.771412 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.20762 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.10272 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.43642 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.23087 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.762195 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.956817 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.760054 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.05571 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.35295 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.703993 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.710877 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.08897 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.889175 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.02722 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.782874 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.2367 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.52253 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.804509 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.880517 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.08438 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.651743 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.896911 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.03351 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.07213 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.60042 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.16764 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.986787 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.803068 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.29788 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.21167 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.787831 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.23715 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.49958 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.719853 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.890823 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.64171 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.06857 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.04455 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.77282 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.41532 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.03789 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.12397 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.736241 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.821804 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.913516 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.15473 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.86545 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.937568 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.584539 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.59879 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.953901 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.03905 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.959505 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.14168 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.79001 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.695828 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.783278 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.36232 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.648192 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.11605 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.19772 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.760822 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.5055 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.25009 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.888045 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.933851 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.09204 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.13607 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.00608 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.00515 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.16782 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.07044 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.683683 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.11437 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.20753 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.882759 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.915054 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.789342 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.48743 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.03978 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.765449 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.07985 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.894335 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.23469 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.05211 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.657479 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.718011 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.961295 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.801106 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.966205 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.873153 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.2945 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.27428 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.798669 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.02761 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.05866 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.732617 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.799023 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.30745 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.13265 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.58809 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12797 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.976232 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.975331 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10859 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.09231 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.881442 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.09096 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.10043 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.957525 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.796619 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.17592 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.04267 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.848302 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08535 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.943649 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.28409 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.09557 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.817916 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.931751 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.00164 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.20911 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.22099 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.767794 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.795117 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.00893 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.868797 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.913537 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.863329 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.13754 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.27469 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.958314 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.03224 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.06419 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.803295 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.985322 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.19064 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.19745 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.43997 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.256999350433453 z_k1_s0000
                                 - 0.208999503286253 z_k1_s0001
                                 - 0.150999552917129 z_k1_s0002
                                 - 0.160999453539666 z_k1_s0003
                                 - 0.354998700693336 z_k1_s0004
                                 - 0.280999274007052 z_k1_s0005
                                 - 0.253998853488281 z_k1_s0006
                                 - 0.357999197638508 z_k1_s0007
                                 - 0.1179999851229 z_k1_s0010
                                 - 0.0870000637770565 z_k1_s0011
                                 - 0.157000206816071 z_k1_s0012
                                 - 0.196999608552622 z_k1_s0013
                                 - 0.189000008205324 z_k1_s0014
                                 - 0.106000024601929 z_k1_s0015
                                 - 0.191000608292179 z_k1_s0016
                                 - 0.35499941541209 z_k1_s0017
                                 - 0.195000007518217 z_k1_s0020
                                 - 0.16900036224892 z_k1_s0021
                                 - 0.204999665852678 z_k1_s0022
                                 - 0.120999867844715 z_k1_s0023
                                 - 0.166999363382632 z_k1_s0024
                                 - 0.0569999201053714 z_k1_s0025
                                 - 0.177000355716337 z_k1_s0026
                                 - 0.242999968322724 z_k1_s0027
                                 - 0.19499943368428 z_k1_s0030
                                 - 0.310999186558784 z_k1_s0031
                                 - 0.482999478557105 z_k1_s0032
                                 - 0.115999752874505 z_k1_s0033
                                 - 0.333999919440376 z_k1_s0034
                                 - 0.223999504502542 z_k1_s0035
                                 - 0.564999093534487 z_k1_s0036
                                 - 0.423997944468593 z_k1_s0037
                                 - 0.221000285890558 z_k1_s0040
                                 - 0.11499998222187 z_k1_s0041
                                 - 0.323999832708633 z_k1_s0042
                                 - 0.068999911573162 z_k1_s0043
                                 - 0.102999935617874 z_k1_s0044
                                 - 0.132000123844159 z_k1_s0045
                                 - 0.068999911573162 z_k1_s0046
                                 - 0.262999736204913 z_k1_s0047
                                 - 0.123000378602625 z_k1_s0050
                                 - 0.0889999364656578 z_k1_s0051
                                 - 0.140000167271549 z_k1_s0052
                                 - 0.0619999350918276 z_k1_s0053
                                 - 0.184999989394783 z_k1_s0054
                                 - 0.0919999246072067 z_k1_s0055
                                 - 0.200999854083117 z_k1_s0056
                                 - 0.188999685317142 z_k1_s0057
                                 - 0.373000461235671 z_k1_s0060
                                 - 0.487000602203142 z_k1_s0061
                                 - 0.588000727095374 z_k1_s0062
                                 - 0.359000443923877 z_k1_s0063
                                 - 0.896001107954856 z_k1_s0064
                                 - 1.12000138494357 z_k1_s0065
                                 - 0.276000341289665 z_k1_s0066
                                 - 0.90100111413764 z_k1_s0067
                                 - 0.237999514658346 z_k1_s0070
                                 - 0.105999807978035 z_k1_s0071
                                 - 0.29899979257411 z_k1_s0072
                                 - 0.207999189618214 z_k1_s0073
                                 - 0.160999637886601 z_k1_s0074
                                 - 0.331999001120742 z_k1_s0075
                                 - 0.272999893871833 z_k1_s0076
                                 - 0.298000164738609 z_k1_s0077
                                 - 0.144999974941552 z_k1_s0100
                                 - 0.0600000337861884 z_k1_s0101
                                 - 0.118999897917496 z_k1_s0102
                                 - 0.140000111667794 z_k1_s0103
                                 - 0.0530000282027987 z_k1_s0104
                                 - 0.243000644109395 z_k1_s0105
                                 - 0.148000286906065 z_k1_s0106
                                 - 0.077000061417287 z_k1_s0107
                                 - 0.146000416579747 z_k1_s0110
                                 - 0.0870001093658789 z_k1_s0111
                                 - 0.0730000460897162 z_k1_s0112
                                 - 0.0530000367952766 z_k1_s0113
                                 - 0.0560000909044937 z_k1_s0114
                                 - 0.105000271821024 z_k1_s0115
                                 - 0.118999767396636 z_k1_s0116
                                 - 0.172000290792385 z_k1_s0117
                                 - 0.206999884878766 z_k1_s0120
                                 - 0.416000393644902 z_k1_s0121
                                 - 0.302000457045002 z_k1_s0122
                                 - 0.254000483739781 z_k1_s0123
                                 - 0.143999919915663 z_k1_s0124
                                 - 0.1139999365999 z_k1_s0125
                                 - 0.254999858183987 z_k1_s0126
                                 - 0.182999898226155 z_k1_s0127
                                 - 0.0640000349963634 z_k1_s0130
                                 - 0.125000125164818 z_k1_s0131
                                 - 0.0940003720761683 z_k1_s0132
                                 - 0.149000168588479 z_k1_s0133
                                 - 0.0460000428286505 z_k1_s0134
                                 - 0.0940003720761683 z_k1_s0135
                                 - 0.126000429974383 z_k1_s0136
                                 - 0.110000401025275 z_k1_s0137
                                 - 0.14999988813179 z_k1_s0140
                                 - 0.118000029863712 z_k1_s0141
                                 - 0.089000060558236 z_k1_s0142
                                 - 0.279000009525199 z_k1_s0143
                                 - 0.268000325995633 z_k1_s0144
                                 - 0.0870000439164729 z_k1_s0145
                                 - 0.200000304175865 z_k1_s0146
                                 - 0.169000454228669 z_k1_s0147
                                 - 0.0989999002139899 z_k1_s0150
                                 - 0.202999295388786 z_k1_s0151
                                 - 0.305998791571421 z_k1_s0152
                                 - 0.434999621546259 z_k1_s0153
                                 - 0.178999739578911 z_k1_s0154
                                 - 0.339999317300914 z_k1_s0155
                                 - 0.104999888166359 z_k1_s0156
                                 - 0.313000064514555 z_k1_s0157
                                 - 0.161999246901554 z_k1_s0160
                                 - 0.208999283575978 z_k1_s0161
                                 - 0.137999664670832 z_k1_s0162
                                 - 0.124999674071468 z_k1_s0163
                                 - 0.232999172005837 z_k1_s0164
                                 - 0.349999393599249 z_k1_s0165
                                 - 0.0779997904966134 z_k1_s0166
                                 - 0.235999923557415 z_k1_s0167
                                 - 0.288000210794413 z_k1_s0170
                                 - 0.15000028535582 z_k1_s0171
                                 - 0.513001463230007 z_k1_s0172
                                 - 0.225001252435593 z_k1_s0173
                                 - 0.123000208343715 z_k1_s0174
                                 - 0.165000368851527 z_k1_s0175
                                 - 0.166000459911434 z_k1_s0176
                                 - 0.202000440460632 z_k1_s0177
                                 - 0.233999196014692 z_k1_s1000
                                 - 0.369000736178146 z_k1_s1001
                                 - 0.161000148231017 z_k1_s1002
                                 - 0.146000049468452 z_k1_s1003
                                 - 0.529001355445262 z_k1_s1004
                                 - 0.143000159976013 z_k1_s1005
                                 - 0.206000010318468 z_k1_s1006
                                 - 0.382999352075707 z_k1_s1007
                                 - 0.317999991583056 z_k1_s1010
                                 - 0.800999876478032 z_k1_s1011
                                 - 0.413000893490323 z_k1_s1012
                                 - 0.324000076103526 z_k1_s1013
                                 - 0.629000944226008 z_k1_s1014
                                 - 0.492999527095551 z_k1_s1015
                                 - 0.235000006717044 z_k1_s1016
                                 - 0.526999881378165 z_k1_s1017
                                 - 0.640999813150968 z_k1_s1020
                                 - 0.852998284502653 z_k1_s1021
                                 - 0.371998975382644 z_k1_s1022
                                 - 0.582000311284039 z_k1_s1023
                                 - 0.623998214990412 z_k1_s1024
                                 - 0.452998822720618 z_k1_s1025
                                 - 1.05599500306579 z_k1_s1026
                                 - 0.55699886775147 z_k1_s1027
                                 - 0.221000140419798 z_k1_s1030
                                 - 0.57900206485123 z_k1_s1031
                                 - 0.511000871890498 z_k1_s1032
                                 - 0.205999997160896 z_k1_s1033
                                 - 0.538000246535912 z_k1_s1034
                                 - 0.498999262602345 z_k1_s1035
                                 - 0.476001556720431 z_k1_s1036
                                 - 0.367001942446727 z_k1_s1037
                                 - 0.888003387070315 z_k1_s1040
                                 - 0.875001204699997 z_k1_s1041
                                 - 1.56799872022871 z_k1_s1042
                                 - 0.485000207222059 z_k1_s1043
                                 - 0.569999959667023 z_k1_s1044
                                 - 1.07400309819942 z_k1_s1045
                                 - 0.399000441887146 z_k1_s1046
                                 - 0.857002315881582 z_k1_s1047
                                 - 0.443001787201544 z_k1_s1050
                                 - 0.295000207896492 z_k1_s1051
                                 - 0.334001103646137 z_k1_s1052
                                 - 0.408000350297565 z_k1_s1053
                                 - 0.415999267988899 z_k1_s1054
                                 - 0.403001026740481 z_k1_s1055
                                 - 0.344999615471722 z_k1_s1056
                                 - 0.441998836487931 z_k1_s1057
                                 - 1.00099708511655 z_k1_s1060
                                 - 1.35700303173443 z_k1_s1061
                                 - 1.66500345438719 z_k1_s1062
                                 - 0.59700105315123 z_k1_s1063
                                 - 0.50400053566797 z_k1_s1064
                                 - 1.00700099199087 z_k1_s1065
                                 - 1.76599774502312 z_k1_s1066
                                 - 0.680000777183832 z_k1_s1067
                                 - 0.647002361527086 z_k1_s1070
                                 - 0.254000389932853 z_k1_s1071
                                 - 0.605002179349103 z_k1_s1072
                                 - 0.405001489787445 z_k1_s1073
                                 - 0.45900119015864 z_k1_s1074
                                 - 0.435999933702951 z_k1_s1075
                                 - 0.474001522151025 z_k1_s1076
                                 - 0.457001893293685 z_k1_s1077
                                 - 0.976997271586067 z_k1_s1100
                                 - 0.386999860008675 z_k1_s1101
                                 - 0.530999820418875 z_k1_s1102
                                 - 0.352000022669335 z_k1_s1103
                                 - 0.251000046704543 z_k1_s1104
                                 - 0.858000189631455 z_k1_s1105
                                 - 0.965000275925798 z_k1_s1106
                                 - 1.05399949373168 z_k1_s1107
                                 - 1.43800429395131 z_k1_s1110
                                 - 0.386000078131779 z_k1_s1111
                                 - 0.633000246634338 z_k1_s1112
                                 - 1.66600145205883 z_k1_s1113
                                 - 1.69900449601153 z_k1_s1114
                                 - 0.728000897897358 z_k1_s1115
                                 - 0.558000490411544 z_k1_s1116
                                 - 1.36300199642636 z_k1_s1117
                                 - 0.504000596204586 z_k1_s1120
                                 - 1.48500329963231 z_k1_s1121
                                 - 0.838998886737845 z_k1_s1122
                                 - 0.549001255979282 z_k1_s1123
                                 - 0.385000943741866 z_k1_s1124
                                 - 0.317000291047379 z_k1_s1125
                                 - 0.446000950872663 z_k1_s1126
                                 - 1.43400012855671 z_k1_s1127
                                 - 0.415000354295314 z_k1_s1130
                                 - 0.196999784978934 z_k1_s1131
                                 - 0.395000869449487 z_k1_s1132
                                 - 0.508000462077141 z_k1_s1133
                                 - 0.144999946380381 z_k1_s1134
                                 - 0.273999715884398 z_k1_s1135
                                 - 0.361000861712028 z_k1_s1136
                                 - 0.649999454981171 z_k1_s1137
                                 - 0.197999901865035 z_k1_s1140
                                 - 0.553000278843152 z_k1_s1141
                                 - 0.478000008833726 z_k1_s1142
                                 - 0.451998608684036 z_k1_s1143
                                 - 0.602999332516313 z_k1_s1144
                                 - 0.45099842487061 z_k1_s1145
                                 - 0.412001393145129 z_k1_s1146
                                 - 0.232000069322633 z_k1_s1147
                                 - 0.307999629076328 z_k1_s1150
                                 - 0.479998363756013 z_k1_s1151
                                 - 0.925999219911076 z_k1_s1152
                                 - 0.230999721807246 z_k1_s1153
                                 - 0.190000031319641 z_k1_s1154
                                 - 0.568000100776607 z_k1_s1155
                                 - 0.399998636463344 z_k1_s1156
                                 - 0.291999683617794 z_k1_s1157
                                 - 0.225999897936813 z_k1_s1160
                                 - 0.211999742655914 z_k1_s1161
                                 - 0.393999189309868 z_k1_s1162
                                 - 0.453998752514692 z_k1_s1163
                                 - 0.371000385547114 z_k1_s1164
                                 - 0.5169998370784 z_k1_s1165
                                 - 0.136999774250336 z_k1_s1166
                                 - 0.260998614340532 z_k1_s1167
                                 - 0.497000140282299 z_k1_s1170
                                 - 0.474999257904148 z_k1_s1171
                                 - 0.284999900342275 z_k1_s1172
                                 - 0.591998609463985 z_k1_s1173
                                 - 0.262999709163696 z_k1_s1174
                                 - 0.358998689725915 z_k1_s1175
                                 - 0.259999903021027 z_k1_s1176
                                 - 0.724999231116633 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.884926 w_k1_ADM_s000
                                 + 0.89102 w_k1_AXP_s000
                                 + 0.832041 w_k1_DIS_s000
                                 + 0.842498 w_k1_FDX_s000
                                 + 1.16632 w_k1_IBM_s000
                                 + 0.883691 w_k1_INTC_s000
                                 + 0.922699 w_k1_KO_s000
                                 + 1.24853 w_k1_MAT_s000
                                 + 0.887418 w_k1_MSFT_s000
                                 + 0.848583 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.48871 w_k1_ADM_s000
                                 + 1.41529 w_k1_AXP_s000
                                 + 1.26018 w_k1_DIS_s000
                                 + 1.37879 w_k1_FDX_s000
                                 + 0.924865 w_k1_IBM_s000
                                 + 1.09486 w_k1_INTC_s000
                                 + 1.00873 w_k1_KO_s000 + 1.2051 w_k1_MAT_s000
                                 + 1.07579 w_k1_MSFT_s000
                                 + 1.09626 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.12605 w_k1_ADM_s000
                                 + 0.981199 w_k1_AXP_s000
                                 + 0.6899 w_k1_DIS_s000
                                 + 0.926034 w_k1_FDX_s000
                                 + 0.587013 w_k1_IBM_s000
                                 + 1.01148 w_k1_INTC_s000
                                 + 0.711881 w_k1_KO_s000
                                 + 0.715968 w_k1_MAT_s000
                                 + 0.901959 w_k1_MSFT_s000
                                 + 0.776409 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.901136 w_k1_ADM_s000
                                 + 1.01251 w_k1_AXP_s000
                                 + 0.793561 w_k1_DIS_s000
                                 + 0.87628 w_k1_FDX_s000
                                 + 0.888381 w_k1_IBM_s000
                                 + 1.03924 w_k1_INTC_s000
                                 + 0.886253 w_k1_KO_s000
                                 + 1.17987 w_k1_MAT_s000
                                 + 0.839436 w_k1_MSFT_s000
                                 + 0.888974 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.864805 w_k1_ADM_s000
                                 + 0.756361 w_k1_AXP_s000
                                 + 0.772396 w_k1_DIS_s000
                                 + 0.797064 w_k1_FDX_s000
                                 + 0.795966 w_k1_IBM_s000
                                 + 0.947906 w_k1_INTC_s000
                                 + 0.909294 w_k1_KO_s000
                                 + 0.95509 w_k1_MAT_s000
                                 + 0.872575 w_k1_MSFT_s000
                                 + 0.897662 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.37046 w_k1_ADM_s000
                                 + 1.42033 w_k1_AXP_s000
                                 + 1.45544 w_k1_DIS_s000
                                 + 1.19767 w_k1_FDX_s000
                                 + 1.43462 w_k1_IBM_s000
                                 + 1.32353 w_k1_INTC_s000
                                 + 1.56824 w_k1_KO_s000 + 1.83252 w_k1_MAT_s000
                                 + 1.36768 w_k1_MSFT_s000
                                 + 1.21749 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.2767 w_k1_ADM_s000
                                 + 0.935881 w_k1_AXP_s000
                                 + 1.13953 w_k1_DIS_s000
                                 + 1.05647 w_k1_FDX_s000
                                 + 0.796023 w_k1_IBM_s000
                                 + 0.934277 w_k1_INTC_s000
                                 + 1.03042 w_k1_KO_s000
                                 + 0.865643 w_k1_MAT_s000
                                 + 1.11213 w_k1_MSFT_s000
                                 + 0.962048 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.18543 w_k1_ADM_s000
                                 + 1.21879 w_k1_AXP_s000
                                 + 1.15229 w_k1_DIS_s000
                                 + 1.06417 w_k1_FDX_s000
                                 + 1.03834 w_k1_IBM_s000
                                 + 1.14884 w_k1_INTC_s000 + 1.0483 w_k1_KO_s000
                                 + 1.19878 w_k1_MAT_s000
                                 + 1.10668 w_k1_MSFT_s000
                                 + 0.989676 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.948503 w_k1_ADM_s001
                                 + 0.811335 w_k1_AXP_s001
                                 + 0.966207 w_k1_DIS_s001
                                 + 0.966414 w_k1_FDX_s001
                                 + 0.777755 w_k1_IBM_s001
                                 + 1.19321 w_k1_INTC_s001
                                 + 1.03951 w_k1_KO_s001
                                 + 0.692069 w_k1_MAT_s001
                                 + 1.18829 w_k1_MSFT_s001
                                 + 1.00375 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.974902 w_k1_ADM_s001
                                 + 0.905622 w_k1_AXP_s001
                                 + 1.16085 w_k1_DIS_s001
                                 + 0.996139 w_k1_FDX_s001
                                 + 1.33722 w_k1_IBM_s001
                                 + 0.968211 w_k1_INTC_s001
                                 + 1.4008 w_k1_KO_s001 + 1.2494 w_k1_MAT_s001
                                 + 1.17459 w_k1_MSFT_s001
                                 + 1.06945 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.728587 w_k1_ADM_s001
                                 + 0.679164 w_k1_AXP_s001
                                 + 0.5734 w_k1_DIS_s001
                                 + 0.747229 w_k1_FDX_s001
                                 + 0.692356 w_k1_IBM_s001
                                 + 0.835 w_k1_INTC_s001 + 0.708436 w_k1_KO_s001
                                 + 0.509021 w_k1_MAT_s001
                                 + 0.825483 w_k1_MSFT_s001
                                 + 0.896925 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.818048 w_k1_ADM_s001
                                 + 0.736503 w_k1_AXP_s001
                                 + 0.698328 w_k1_DIS_s001
                                 + 0.84753 w_k1_FDX_s001 + 1.1034 w_k1_IBM_s001
                                 + 0.934479 w_k1_INTC_s001
                                 + 0.994513 w_k1_KO_s001
                                 + 0.99541 w_k1_MAT_s001
                                 + 0.933295 w_k1_MSFT_s001
                                 + 0.943392 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.15942 w_k1_ADM_s001
                                 + 0.860481 w_k1_AXP_s001
                                 + 1.08369 w_k1_DIS_s001
                                 + 1.14951 w_k1_FDX_s001
                                 + 0.756139 w_k1_IBM_s001
                                 + 0.965984 w_k1_INTC_s001
                                 + 0.932066 w_k1_KO_s001
                                 + 0.706037 w_k1_MAT_s001
                                 + 1.27945 w_k1_MSFT_s001
                                 + 1.06189 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.756583 w_k1_ADM_s001
                                 + 0.567403 w_k1_AXP_s001
                                 + 0.851206 w_k1_DIS_s001
                                 + 0.771738 w_k1_FDX_s001
                                 + 0.898077 w_k1_IBM_s001
                                 + 0.962891 w_k1_INTC_s001
                                 + 1.0113 w_k1_KO_s001 + 0.624554 w_k1_MAT_s001
                                 + 1.02441 w_k1_MSFT_s001
                                 + 0.902956 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05408 w_k1_ADM_s001
                                 + 1.56689 w_k1_AXP_s001
                                 + 1.75431 w_k1_DIS_s001
                                 + 1.50087 w_k1_FDX_s001
                                 + 1.19523 w_k1_IBM_s001
                                 + 1.54054 w_k1_INTC_s001
                                 + 1.59133 w_k1_KO_s001 + 1.67149 w_k1_MAT_s001
                                 + 1.47102 w_k1_MSFT_s001
                                 + 1.31969 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.01022 w_k1_ADM_s001
                                 + 1.22822 w_k1_AXP_s001
                                 + 1.06375 w_k1_DIS_s001
                                 + 1.08751 w_k1_FDX_s001
                                 + 1.00355 w_k1_IBM_s001
                                 + 1.18808 w_k1_INTC_s001
                                 + 1.16528 w_k1_KO_s001
                                 + 0.908944 w_k1_MAT_s001
                                 + 1.14008 w_k1_MSFT_s001
                                 + 1.08103 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.39125 w_k1_ADM_s002
                                 + 1.38952 w_k1_AXP_s002
                                 + 1.60112 w_k1_DIS_s002
                                 + 1.40885 w_k1_FDX_s002
                                 + 1.31173 w_k1_IBM_s002
                                 + 1.34398 w_k1_INTC_s002
                                 + 1.35247 w_k1_KO_s002 + 1.11625 w_k1_MAT_s002
                                 + 1.27445 w_k1_MSFT_s002 + 1.3216 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.777626 w_k1_ADM_s002
                                 + 0.609595 w_k1_AXP_s002
                                 + 0.629415 w_k1_DIS_s002
                                 + 0.756026 w_k1_FDX_s002
                                 + 0.706658 w_k1_IBM_s002
                                 + 0.674473 w_k1_INTC_s002
                                 + 0.696366 w_k1_KO_s002
                                 + 0.583857 w_k1_MAT_s002
                                 + 0.686208 w_k1_MSFT_s002
                                 + 0.784311 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.18226 w_k1_ADM_s002
                                 + 1.06873 w_k1_AXP_s002
                                 + 1.20476 w_k1_DIS_s002
                                 + 1.07558 w_k1_FDX_s002
                                 + 1.27924 w_k1_IBM_s002
                                 + 1.15592 w_k1_INTC_s002
                                 + 1.24863 w_k1_KO_s002 + 1.07284 w_k1_MAT_s002
                                 + 1.05654 w_k1_MSFT_s002
                                 + 1.21239 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.28704 w_k1_ADM_s002
                                 + 1.28098 w_k1_AXP_s002
                                 + 1.05863 w_k1_DIS_s002
                                 + 1.07393 w_k1_FDX_s002
                                 + 1.20807 w_k1_IBM_s002
                                 + 0.892928 w_k1_INTC_s002
                                 + 1.07731 w_k1_KO_s002 + 1.79682 w_k1_MAT_s002
                                 + 0.933481 w_k1_MSFT_s002
                                 + 1.04498 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.90789 w_k1_ADM_s002
                                 + 0.934039 w_k1_AXP_s002
                                 + 0.777043 w_k1_DIS_s002
                                 + 0.866094 w_k1_FDX_s002
                                 + 1.17909 w_k1_IBM_s002
                                 + 0.688973 w_k1_INTC_s002
                                 + 0.903724 w_k1_KO_s002
                                 + 1.40602 w_k1_MAT_s002
                                 + 0.791035 w_k1_MSFT_s002
                                 + 0.935086 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.742415 w_k1_ADM_s002
                                 + 0.652054 w_k1_AXP_s002
                                 + 0.826648 w_k1_DIS_s002
                                 + 0.772006 w_k1_FDX_s002
                                 + 0.821144 w_k1_IBM_s002
                                 + 0.877262 w_k1_INTC_s002
                                 + 0.841878 w_k1_KO_s002
                                 + 0.787993 w_k1_MAT_s002
                                 + 0.910254 w_k1_MSFT_s002
                                 + 0.903583 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.963102 w_k1_ADM_s002
                                 + 0.853876 w_k1_AXP_s002
                                 + 0.860127 w_k1_DIS_s002
                                 + 0.977385 w_k1_FDX_s002
                                 + 0.722479 w_k1_IBM_s002
                                 + 0.887146 w_k1_INTC_s002
                                 + 0.828299 w_k1_KO_s002
                                 + 0.767806 w_k1_MAT_s002
                                 + 1.02103 w_k1_MSFT_s002
                                 + 0.997172 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.00364 w_k1_ADM_s002
                                 + 0.961894 w_k1_AXP_s002
                                 + 1.02561 w_k1_DIS_s002
                                 + 1.11711 w_k1_FDX_s002
                                 + 0.779286 w_k1_IBM_s002
                                 + 0.921145 w_k1_INTC_s002
                                 + 0.842714 w_k1_KO_s002
                                 + 1.15208 w_k1_MAT_s002
                                 + 1.14781 w_k1_MSFT_s002
                                 + 0.956512 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.805681 w_k1_ADM_s003
                                 + 1.0803 w_k1_AXP_s003
                                 + 0.839013 w_k1_DIS_s003
                                 + 0.862097 w_k1_FDX_s003
                                 + 1.38373 w_k1_IBM_s003
                                 + 0.9107 w_k1_INTC_s003 + 1.08579 w_k1_KO_s003
                                 + 1.77415 w_k1_MAT_s003
                                 + 0.9226 w_k1_MSFT_s003
                                 + 0.888226 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.935816 w_k1_ADM_s003
                                 + 0.995064 w_k1_AXP_s003
                                 + 0.757668 w_k1_DIS_s003
                                 + 0.917939 w_k1_FDX_s003
                                 + 0.984904 w_k1_IBM_s003
                                 + 0.949297 w_k1_INTC_s003
                                 + 1.04706 w_k1_KO_s003
                                 + 0.865819 w_k1_MAT_s003
                                 + 0.994095 w_k1_MSFT_s003
                                 + 1.03331 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.13719 w_k1_ADM_s003
                                 + 1.45182 w_k1_AXP_s003
                                 + 1.17916 w_k1_DIS_s003
                                 + 1.18031 w_k1_FDX_s003
                                 + 1.41346 w_k1_IBM_s003
                                 + 1.20821 w_k1_INTC_s003
                                 + 1.31406 w_k1_KO_s003 + 1.74563 w_k1_MAT_s003
                                 + 1.17435 w_k1_MSFT_s003
                                 + 1.07212 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.01573 w_k1_ADM_s003
                                 + 1.73381 w_k1_AXP_s003
                                 + 1.42582 w_k1_DIS_s003
                                 + 1.17884 w_k1_FDX_s003
                                 + 2.00113 w_k1_IBM_s003
                                 + 1.20899 w_k1_INTC_s003
                                 + 1.69671 w_k1_KO_s003 + 2.02963 w_k1_MAT_s003
                                 + 1.30731 w_k1_MSFT_s003
                                 + 1.29778 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.954995 w_k1_ADM_s003
                                 + 0.919495 w_k1_AXP_s003
                                 + 0.710072 w_k1_DIS_s003
                                 + 0.983962 w_k1_FDX_s003
                                 + 0.470136 w_k1_IBM_s003
                                 + 1.01798 w_k1_INTC_s003
                                 + 0.680666 w_k1_KO_s003
                                 + 0.498162 w_k1_MAT_s003
                                 + 1.02693 w_k1_MSFT_s003
                                 + 0.829512 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.761564 w_k1_ADM_s003
                                 + 0.809022 w_k1_AXP_s003
                                 + 0.511982 w_k1_DIS_s003
                                 + 0.756148 w_k1_FDX_s003
                                 + 0.628446 w_k1_IBM_s003
                                 + 0.8267 w_k1_INTC_s003
                                 + 0.602014 w_k1_KO_s003
                                 + 0.534331 w_k1_MAT_s003
                                 + 0.72357 w_k1_MSFT_s003
                                 + 0.741539 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.08982 w_k1_ADM_s003
                                 + 1.06565 w_k1_AXP_s003
                                 + 1.24544 w_k1_DIS_s003
                                 + 1.14201 w_k1_FDX_s003
                                 + 1.00207 w_k1_IBM_s003
                                 + 1.12705 w_k1_INTC_s003 + 1.1066 w_k1_KO_s003
                                 + 0.953284 w_k1_MAT_s003
                                 + 1.28869 w_k1_MSFT_s003
                                 + 1.02976 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.918211 w_k1_ADM_s003
                                 + 1.11622 w_k1_AXP_s003
                                 + 0.965418 w_k1_DIS_s003
                                 + 0.988212 w_k1_FDX_s003
                                 + 0.694187 w_k1_IBM_s003
                                 + 0.988844 w_k1_INTC_s003
                                 + 0.915414 w_k1_KO_s003
                                 + 1.01977 w_k1_MAT_s003
                                 + 1.05454 w_k1_MSFT_s003
                                 + 0.907418 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.895555 w_k1_ADM_s004
                                 + 0.747055 w_k1_AXP_s004
                                 + 0.700406 w_k1_DIS_s004
                                 + 0.81114 w_k1_FDX_s004
                                 + 0.884925 w_k1_IBM_s004
                                 + 0.701225 w_k1_INTC_s004
                                 + 0.814817 w_k1_KO_s004
                                 + 0.975477 w_k1_MAT_s004
                                 + 0.851241 w_k1_MSFT_s004
                                 + 0.914102 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.39384 w_k1_ADM_s004
                                 + 1.22388 w_k1_AXP_s004
                                 + 1.61268 w_k1_DIS_s004
                                 + 1.28065 w_k1_FDX_s004
                                 + 1.82874 w_k1_IBM_s004
                                 + 1.22733 w_k1_INTC_s004
                                 + 1.71801 w_k1_KO_s004 + 2.02329 w_k1_MAT_s004
                                 + 1.41661 w_k1_MSFT_s004
                                 + 1.53872 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.948691 w_k1_ADM_s004
                                 + 0.993506 w_k1_AXP_s004
                                 + 1.41608 w_k1_DIS_s004
                                 + 1.20623 w_k1_FDX_s004
                                 + 1.09113 w_k1_IBM_s004
                                 + 1.02795 w_k1_INTC_s004
                                 + 1.27774 w_k1_KO_s004
                                 + 0.970304 w_k1_MAT_s004
                                 + 1.28881 w_k1_MSFT_s004
                                 + 1.25615 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.4589 w_k1_ADM_s004
                                 + 1.51996 w_k1_AXP_s004 + 1.1869 w_k1_DIS_s004
                                 + 1.40789 w_k1_FDX_s004
                                 + 0.982412 w_k1_IBM_s004
                                 + 1.16615 w_k1_INTC_s004
                                 + 1.17417 w_k1_KO_s004 + 1.32617 w_k1_MAT_s004
                                 + 1.41621 w_k1_MSFT_s004
                                 + 1.12828 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.02993 w_k1_ADM_s004
                                 + 0.853571 w_k1_AXP_s004
                                 + 0.97411 w_k1_DIS_s004 + 1.0584 w_k1_FDX_s004
                                 + 0.812699 w_k1_IBM_s004
                                 + 0.837688 w_k1_INTC_s004
                                 + 0.976084 w_k1_KO_s004
                                 + 1.14678 w_k1_MAT_s004
                                 + 1.06991 w_k1_MSFT_s004
                                 + 0.964137 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.77483 w_k1_ADM_s004
                                 + 0.549906 w_k1_AXP_s004
                                 + 0.728809 w_k1_DIS_s004
                                 + 0.90738 w_k1_FDX_s004
                                 + 0.660182 w_k1_IBM_s004
                                 + 0.938587 w_k1_INTC_s004
                                 + 0.794245 w_k1_KO_s004
                                 + 0.490613 w_k1_MAT_s004
                                 + 0.969416 w_k1_MSFT_s004
                                 + 0.821662 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.778936 w_k1_ADM_s004
                                 + 0.527086 w_k1_AXP_s004
                                 + 0.967325 w_k1_DIS_s004
                                 + 0.884088 w_k1_FDX_s004
                                 + 0.813134 w_k1_IBM_s004
                                 + 0.908081 w_k1_INTC_s004
                                 + 0.990863 w_k1_KO_s004
                                 + 0.75185 w_k1_MAT_s004
                                 + 1.03838 w_k1_MSFT_s004
                                 + 0.84852 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.84928 w_k1_ADM_s004
                                 + 0.839101 w_k1_AXP_s004
                                 + 0.98957 w_k1_DIS_s004
                                 + 0.923826 w_k1_FDX_s004
                                 + 1.13022 w_k1_IBM_s004
                                 + 0.901485 w_k1_INTC_s004
                                 + 1.10282 w_k1_KO_s004 + 1.15602 w_k1_MAT_s004
                                 + 0.982659 w_k1_MSFT_s004
                                 + 0.98255 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.34625 w_k1_ADM_s005
                                 + 1.4313 w_k1_AXP_s005 + 1.48001 w_k1_DIS_s005
                                 + 1.31197 w_k1_FDX_s005
                                 + 1.56719 w_k1_IBM_s005
                                 + 1.63188 w_k1_INTC_s005 + 1.6957 w_k1_KO_s005
                                 + 1.59004 w_k1_MAT_s005
                                 + 1.4962 w_k1_MSFT_s005 + 1.27611 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.08191 w_k1_ADM_s005
                                 + 1.36414 w_k1_AXP_s005
                                 + 0.981351 w_k1_DIS_s005
                                 + 0.982986 w_k1_FDX_s005
                                 + 1.27907 w_k1_IBM_s005
                                 + 1.17915 w_k1_INTC_s005
                                 + 1.05736 w_k1_KO_s005 + 1.20974 w_k1_MAT_s005
                                 + 0.86976 w_k1_MSFT_s005
                                 + 1.15562 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.931474 w_k1_ADM_s005
                                 + 0.613579 w_k1_AXP_s005
                                 + 0.66258 w_k1_DIS_s005
                                 + 0.784732 w_k1_FDX_s005
                                 + 0.679079 w_k1_IBM_s005
                                 + 0.930863 w_k1_INTC_s005
                                 + 0.914409 w_k1_KO_s005
                                 + 0.384105 w_k1_MAT_s005
                                 + 0.724575 w_k1_MSFT_s005
                                 + 0.856748 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.16067 w_k1_ADM_s005
                                 + 1.36636 w_k1_AXP_s005
                                 + 1.14613 w_k1_DIS_s005
                                 + 1.21402 w_k1_FDX_s005 + 1.104 w_k1_IBM_s005
                                 + 1.10521 w_k1_INTC_s005 + 1.1873 w_k1_KO_s005
                                 + 1.01365 w_k1_MAT_s005
                                 + 1.20766 w_k1_MSFT_s005
                                 + 1.08428 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.96225 w_k1_ADM_s005
                                 + 1.00869 w_k1_AXP_s005
                                 + 1.06726 w_k1_DIS_s005 + 1.0827 w_k1_FDX_s005
                                 + 0.800795 w_k1_IBM_s005
                                 + 1.09298 w_k1_INTC_s005
                                 + 0.838617 w_k1_KO_s005
                                 + 0.914172 w_k1_MAT_s005
                                 + 1.01621 w_k1_MSFT_s005
                                 + 1.00479 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.206 w_k1_ADM_s005
                                 + 1.57502 w_k1_AXP_s005
                                 + 1.28925 w_k1_DIS_s005
                                 + 1.37058 w_k1_FDX_s005
                                 + 1.03612 w_k1_IBM_s005
                                 + 1.35933 w_k1_INTC_s005
                                 + 0.981677 w_k1_KO_s005
                                 + 1.01022 w_k1_MAT_s005
                                 + 1.18319 w_k1_MSFT_s005
                                 + 1.09299 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.87052 w_k1_ADM_s005
                                 + 0.940548 w_k1_AXP_s005
                                 + 0.728256 w_k1_DIS_s005
                                 + 0.877977 w_k1_FDX_s005
                                 + 0.949436 w_k1_IBM_s005
                                 + 0.981157 w_k1_INTC_s005
                                 + 0.862399 w_k1_KO_s005
                                 + 0.770133 w_k1_MAT_s005
                                 + 0.750238 w_k1_MSFT_s005
                                 + 0.997156 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.987373 w_k1_ADM_s005
                                 + 0.76146 w_k1_AXP_s005
                                 + 0.961948 w_k1_DIS_s005
                                 + 0.870687 w_k1_FDX_s005
                                 + 1.12151 w_k1_IBM_s005
                                 + 0.869251 w_k1_INTC_s005
                                 + 1.21135 w_k1_KO_s005
                                 + 0.855795 w_k1_MAT_s005
                                 + 0.903786 w_k1_MSFT_s005
                                 + 1.0474 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.878818 w_k1_ADM_s006
                                 + 0.669829 w_k1_AXP_s006
                                 + 0.58786 w_k1_DIS_s006
                                 + 0.851543 w_k1_FDX_s006
                                 + 0.55155 w_k1_IBM_s006
                                 + 0.828286 w_k1_INTC_s006
                                 + 0.713924 w_k1_KO_s006
                                 + 0.586314 w_k1_MAT_s006
                                 + 0.771616 w_k1_MSFT_s006
                                 + 0.852937 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.05004 w_k1_ADM_s006
                                 + 1.09803 w_k1_AXP_s006
                                 + 1.07373 w_k1_DIS_s006
                                 + 1.03424 w_k1_FDX_s006
                                 + 1.69398 w_k1_IBM_s006
                                 + 0.950653 w_k1_INTC_s006
                                 + 1.2316 w_k1_KO_s006 + 1.29533 w_k1_MAT_s006
                                 + 0.939171 w_k1_MSFT_s006
                                 + 1.19546 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.26326 w_k1_ADM_s006
                                 + 1.01683 w_k1_AXP_s006
                                 + 1.20021 w_k1_DIS_s006
                                 + 1.26652 w_k1_FDX_s006
                                 + 0.641255 w_k1_IBM_s006
                                 + 1.20376 w_k1_INTC_s006
                                 + 0.967602 w_k1_KO_s006
                                 + 0.665798 w_k1_MAT_s006
                                 + 1.02276 w_k1_MSFT_s006
                                 + 1.11275 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.887419 w_k1_ADM_s006
                                 + 1.02207 w_k1_AXP_s006
                                 + 0.867626 w_k1_DIS_s006
                                 + 0.933308 w_k1_FDX_s006
                                 + 1.20238 w_k1_IBM_s006
                                 + 1.21292 w_k1_INTC_s006
                                 + 1.12945 w_k1_KO_s006 + 1.21662 w_k1_MAT_s006
                                 + 0.926077 w_k1_MSFT_s006
                                 + 0.938512 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.27031 w_k1_ADM_s006
                                 + 1.41084 w_k1_AXP_s006
                                 + 1.45721 w_k1_DIS_s006
                                 + 1.30663 w_k1_FDX_s006
                                 + 1.28217 w_k1_IBM_s006
                                 + 1.41438 w_k1_INTC_s006
                                 + 1.45308 w_k1_KO_s006 + 1.17174 w_k1_MAT_s006
                                 + 1.12769 w_k1_MSFT_s006
                                 + 1.15353 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.1117 w_k1_ADM_s006
                                 + 0.864167 w_k1_AXP_s006
                                 + 0.855439 w_k1_DIS_s006
                                 + 0.947247 w_k1_FDX_s006
                                 + 0.970253 w_k1_IBM_s006
                                 + 0.921164 w_k1_INTC_s006
                                 + 0.984433 w_k1_KO_s006
                                 + 0.839735 w_k1_MAT_s006
                                 + 0.902719 w_k1_MSFT_s006
                                 + 1.00778 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.942203 w_k1_ADM_s006
                                 + 0.831495 w_k1_AXP_s006
                                 + 0.984961 w_k1_DIS_s006
                                 + 0.940501 w_k1_FDX_s006
                                 + 1.2924 w_k1_IBM_s006
                                 + 1.09383 w_k1_INTC_s006
                                 + 1.09951 w_k1_KO_s006
                                 + 0.991238 w_k1_MAT_s006
                                 + 1.00987 w_k1_MSFT_s006
                                 + 1.05587 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.982767 w_k1_ADM_s006
                                 + 0.852938 w_k1_AXP_s006
                                 + 0.804201 w_k1_DIS_s006
                                 + 0.952952 w_k1_FDX_s006
                                 + 0.716892 w_k1_IBM_s006
                                 + 0.965277 w_k1_INTC_s006
                                 + 0.984472 w_k1_KO_s006
                                 + 0.703846 w_k1_MAT_s006
                                 + 0.882132 w_k1_MSFT_s006
                                 + 0.976154 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.24461 w_k1_ADM_s007
                                 + 1.26255 w_k1_AXP_s007
                                 + 1.53801 w_k1_DIS_s007
                                 + 1.28609 w_k1_FDX_s007
                                 + 1.28901 w_k1_IBM_s007
                                 + 1.17128 w_k1_INTC_s007
                                 + 1.44324 w_k1_KO_s007 + 1.29445 w_k1_MAT_s007
                                 + 1.2799 w_k1_MSFT_s007 + 1.25641 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.948258 w_k1_ADM_s007
                                 + 1.1335 w_k1_AXP_s007
                                 + 0.762548 w_k1_DIS_s007
                                 + 0.900887 w_k1_FDX_s007
                                 + 0.978479 w_k1_IBM_s007
                                 + 0.973735 w_k1_INTC_s007
                                 + 0.905724 w_k1_KO_s007
                                 + 1.17036 w_k1_MAT_s007
                                 + 0.823288 w_k1_MSFT_s007
                                 + 0.9751 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.14078 w_k1_ADM_s007
                                 + 1.1022 w_k1_AXP_s007 + 1.22237 w_k1_DIS_s007
                                 + 1.14343 w_k1_FDX_s007
                                 + 0.998854 w_k1_IBM_s007
                                 + 1.2651 w_k1_INTC_s007 + 1.04425 w_k1_KO_s007
                                 + 0.885046 w_k1_MAT_s007
                                 + 1.02919 w_k1_MSFT_s007
                                 + 1.10173 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.846456 w_k1_ADM_s007
                                 + 0.792094 w_k1_AXP_s007
                                 + 0.574185 w_k1_DIS_s007
                                 + 0.775007 w_k1_FDX_s007
                                 + 0.71373 w_k1_IBM_s007
                                 + 0.62544 w_k1_INTC_s007
                                 + 0.630385 w_k1_KO_s007
                                 + 0.679027 w_k1_MAT_s007
                                 + 0.726219 w_k1_MSFT_s007
                                 + 0.752409 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.975027 w_k1_ADM_s007
                                 + 1.39171 w_k1_AXP_s007 + 1.2347 w_k1_DIS_s007
                                 + 1.02328 w_k1_FDX_s007
                                 + 1.82597 w_k1_IBM_s007
                                 + 1.32887 w_k1_INTC_s007 + 1.4096 w_k1_KO_s007
                                 + 1.64655 w_k1_MAT_s007
                                 + 1.06178 w_k1_MSFT_s007
                                 + 1.05696 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.13142 w_k1_ADM_s007
                                 + 1.08583 w_k1_AXP_s007
                                 + 0.825314 w_k1_DIS_s007
                                 + 1.00455 w_k1_FDX_s007
                                 + 0.717103 w_k1_IBM_s007
                                 + 1.19166 w_k1_INTC_s007
                                 + 0.757384 w_k1_KO_s007
                                 + 0.561696 w_k1_MAT_s007
                                 + 1.01387 w_k1_MSFT_s007
                                 + 0.85733 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.972029 w_k1_ADM_s007
                                 + 1.42125 w_k1_AXP_s007
                                 + 0.965292 w_k1_DIS_s007
                                 + 0.986008 w_k1_FDX_s007
                                 + 1.24409 w_k1_IBM_s007
                                 + 1.1315 w_k1_INTC_s007 + 1.04416 w_k1_KO_s007
                                 + 1.82467 w_k1_MAT_s007
                                 + 1.04364 w_k1_MSFT_s007
                                 + 0.909837 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.711496 w_k1_ADM_s007
                                 + 0.68323 w_k1_AXP_s007
                                 + 0.818448 w_k1_DIS_s007
                                 + 0.748824 w_k1_FDX_s007
                                 + 1.10981 w_k1_IBM_s007
                                 + 0.893883 w_k1_INTC_s007
                                 + 1.07803 w_k1_KO_s007
                                 + 0.892677 w_k1_MAT_s007
                                 + 0.884149 w_k1_MSFT_s007
                                 + 0.887413 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.23476 w_k1_ADM_s010
                                 + 0.999763 w_k1_AXP_s010
                                 + 0.782877 w_k1_DIS_s010
                                 + 1.02735 w_k1_FDX_s010
                                 + 0.794565 w_k1_IBM_s010
                                 + 0.890813 w_k1_INTC_s010
                                 + 0.696833 w_k1_KO_s010
                                 + 0.725731 w_k1_MAT_s010
                                 + 0.916674 w_k1_MSFT_s010
                                 + 1.02506 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.0749 w_k1_ADM_s010
                                 + 1.22959 w_k1_AXP_s010
                                 + 1.35043 w_k1_DIS_s010
                                 + 1.25817 w_k1_FDX_s010
                                 + 0.965289 w_k1_IBM_s010
                                 + 1.13679 w_k1_INTC_s010
                                 + 0.944398 w_k1_KO_s010
                                 + 1.26969 w_k1_MAT_s010
                                 + 1.10706 w_k1_MSFT_s010
                                 + 1.06431 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.15441 w_k1_ADM_s010
                                 + 1.19555 w_k1_AXP_s010
                                 + 1.13168 w_k1_DIS_s010
                                 + 1.12106 w_k1_FDX_s010
                                 + 1.42599 w_k1_IBM_s010
                                 + 1.14525 w_k1_INTC_s010 + 1.0766 w_k1_KO_s010
                                 + 1.51218 w_k1_MAT_s010
                                 + 1.14164 w_k1_MSFT_s010
                                 + 1.16857 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.980927 w_k1_ADM_s010
                                 + 0.743942 w_k1_AXP_s010
                                 + 0.844963 w_k1_DIS_s010
                                 + 0.844603 w_k1_FDX_s010
                                 + 1.15139 w_k1_IBM_s010
                                 + 1.08024 w_k1_INTC_s010
                                 + 0.853848 w_k1_KO_s010
                                 + 0.99498 w_k1_MAT_s010 + 1.006 w_k1_MSFT_s010
                                 + 1.09511 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.54623 w_k1_ADM_s010
                                 + 1.12334 w_k1_AXP_s010
                                 + 1.40213 w_k1_DIS_s010
                                 + 1.32185 w_k1_FDX_s010 + 1.1556 w_k1_IBM_s010
                                 + 1.41229 w_k1_INTC_s010 + 1.1549 w_k1_KO_s010
                                 + 0.842708 w_k1_MAT_s010
                                 + 1.43926 w_k1_MSFT_s010
                                 + 1.21951 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.97179 w_k1_ADM_s010
                                 + 0.693854 w_k1_AXP_s010
                                 + 0.811261 w_k1_DIS_s010
                                 + 0.916919 w_k1_FDX_s010
                                 + 0.741214 w_k1_IBM_s010
                                 + 1.0738 w_k1_INTC_s010 + 0.67717 w_k1_KO_s010
                                 + 0.460543 w_k1_MAT_s010
                                 + 0.853704 w_k1_MSFT_s010
                                 + 1.01632 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.2725 w_k1_ADM_s010
                                 + 1.00119 w_k1_AXP_s010
                                 + 1.26506 w_k1_DIS_s010
                                 + 1.09686 w_k1_FDX_s010
                                 + 1.03045 w_k1_IBM_s010
                                 + 0.965477 w_k1_INTC_s010
                                 + 1.02378 w_k1_KO_s010
                                 + 0.914611 w_k1_MAT_s010
                                 + 1.03957 w_k1_MSFT_s010
                                 + 1.08891 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.13295 w_k1_ADM_s010
                                 + 1.22561 w_k1_AXP_s010
                                 + 1.76594 w_k1_DIS_s010 + 1.4836 w_k1_FDX_s010
                                 + 1.03101 w_k1_IBM_s010
                                 + 1.6244 w_k1_INTC_s010 + 1.23063 w_k1_KO_s010
                                 + 0.939553 w_k1_MAT_s010
                                 + 1.33609 w_k1_MSFT_s010
                                 + 1.21615 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.648812 w_k1_ADM_s011
                                 + 0.649101 w_k1_AXP_s011
                                 + 0.611578 w_k1_DIS_s011
                                 + 0.734951 w_k1_FDX_s011
                                 + 0.689686 w_k1_IBM_s011
                                 + 0.763426 w_k1_INTC_s011
                                 + 0.693407 w_k1_KO_s011
                                 + 0.653471 w_k1_MAT_s011
                                 + 0.753757 w_k1_MSFT_s011
                                 + 0.805763 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.23737 w_k1_ADM_s011
                                 + 1.74979 w_k1_AXP_s011
                                 + 1.67834 w_k1_DIS_s011
                                 + 1.41799 w_k1_FDX_s011
                                 + 1.31577 w_k1_IBM_s011
                                 + 1.18563 w_k1_INTC_s011
                                 + 1.19027 w_k1_KO_s011 + 2.68909 w_k1_MAT_s011
                                 + 1.35978 w_k1_MSFT_s011
                                 + 1.26643 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.07093 w_k1_ADM_s011
                                 + 1.26834 w_k1_AXP_s011
                                 + 1.33993 w_k1_DIS_s011
                                 + 0.933911 w_k1_FDX_s011
                                 + 2.01403 w_k1_IBM_s011
                                 + 0.941731 w_k1_INTC_s011
                                 + 1.48598 w_k1_KO_s011 + 2.23275 w_k1_MAT_s011
                                 + 1.00869 w_k1_MSFT_s011
                                 + 1.03534 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.70307 w_k1_ADM_s011
                                 + 1.11148 w_k1_AXP_s011
                                 + 1.06196 w_k1_DIS_s011
                                 + 0.872334 w_k1_FDX_s011
                                 + 1.61634 w_k1_IBM_s011
                                 + 1.10722 w_k1_INTC_s011
                                 + 1.15677 w_k1_KO_s011 + 1.71749 w_k1_MAT_s011
                                 + 1.04269 w_k1_MSFT_s011
                                 + 1.07569 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.95067 w_k1_ADM_s011
                                 + 1.27354 w_k1_AXP_s011
                                 + 1.08678 w_k1_DIS_s011
                                 + 1.03801 w_k1_FDX_s011 + 1.3071 w_k1_IBM_s011
                                 + 1.15246 w_k1_INTC_s011 + 1.0971 w_k1_KO_s011
                                 + 1.15318 w_k1_MAT_s011
                                 + 1.08874 w_k1_MSFT_s011
                                 + 1.14895 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.07932 w_k1_ADM_s011
                                 + 1.07355 w_k1_AXP_s011
                                 + 1.18857 w_k1_DIS_s011 + 1.0471 w_k1_FDX_s011
                                 + 0.975151 w_k1_IBM_s011
                                 + 1.1017 w_k1_INTC_s011
                                 + 0.877847 w_k1_KO_s011
                                 + 0.719798 w_k1_MAT_s011
                                 + 0.900053 w_k1_MSFT_s011
                                 + 1.04864 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.839639 w_k1_ADM_s011
                                 + 0.97367 w_k1_AXP_s011
                                 + 0.841326 w_k1_DIS_s011
                                 + 0.802841 w_k1_FDX_s011
                                 + 1.11076 w_k1_IBM_s011
                                 + 0.964763 w_k1_INTC_s011
                                 + 0.859329 w_k1_KO_s011
                                 + 1.17515 w_k1_MAT_s011
                                 + 0.911931 w_k1_MSFT_s011
                                 + 0.897851 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.798227 w_k1_ADM_s011
                                 + 0.934188 w_k1_AXP_s011
                                 + 0.927085 w_k1_DIS_s011
                                 + 0.943667 w_k1_FDX_s011
                                 + 0.649167 w_k1_IBM_s011
                                 + 0.960239 w_k1_INTC_s011
                                 + 0.730353 w_k1_KO_s011
                                 + 0.551809 w_k1_MAT_s011
                                 + 0.927675 w_k1_MSFT_s011
                                 + 0.883482 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.810286 w_k1_ADM_s012
                                 + 1.06312 w_k1_AXP_s012
                                 + 1.43503 w_k1_DIS_s012
                                 + 1.08775 w_k1_FDX_s012
                                 + 1.34631 w_k1_IBM_s012
                                 + 0.901003 w_k1_INTC_s012
                                 + 1.06809 w_k1_KO_s012 + 1.44065 w_k1_MAT_s012
                                 + 1.07102 w_k1_MSFT_s012 + 1.082 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.929469 w_k1_ADM_s012
                                 + 1.32509 w_k1_AXP_s012
                                 + 1.12511 w_k1_DIS_s012
                                 + 1.15645 w_k1_FDX_s012
                                 + 0.976529 w_k1_IBM_s012
                                 + 0.871142 w_k1_INTC_s012
                                 + 0.828685 w_k1_KO_s012
                                 + 0.969796 w_k1_MAT_s012
                                 + 0.952274 w_k1_MSFT_s012
                                 + 1.0184 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.932372 w_k1_ADM_s012
                                 + 0.970948 w_k1_AXP_s012
                                 + 0.835914 w_k1_DIS_s012
                                 + 0.865522 w_k1_FDX_s012
                                 + 1.0653 w_k1_IBM_s012
                                 + 0.715133 w_k1_INTC_s012
                                 + 0.707082 w_k1_KO_s012
                                 + 1.31254 w_k1_MAT_s012
                                 + 0.806111 w_k1_MSFT_s012
                                 + 0.974263 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.800637 w_k1_ADM_s012
                                 + 0.940195 w_k1_AXP_s012
                                 + 1.08187 w_k1_DIS_s012
                                 + 1.04574 w_k1_FDX_s012
                                 + 1.00288 w_k1_IBM_s012
                                 + 0.842754 w_k1_INTC_s012
                                 + 0.769973 w_k1_KO_s012
                                 + 1.20016 w_k1_MAT_s012
                                 + 1.05523 w_k1_MSFT_s012
                                 + 0.888463 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.16129 w_k1_ADM_s012
                                 + 0.771211 w_k1_AXP_s012
                                 + 1.12515 w_k1_DIS_s012
                                 + 1.08236 w_k1_FDX_s012
                                 + 0.727507 w_k1_IBM_s012
                                 + 0.909073 w_k1_INTC_s012
                                 + 0.857546 w_k1_KO_s012
                                 + 0.723179 w_k1_MAT_s012
                                 + 1.03192 w_k1_MSFT_s012
                                 + 1.03917 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.869548 w_k1_ADM_s012
                                 + 1.64612 w_k1_AXP_s012
                                 + 1.47579 w_k1_DIS_s012
                                 + 1.24688 w_k1_FDX_s012
                                 + 1.44886 w_k1_IBM_s012
                                 + 1.10846 w_k1_INTC_s012 + 1.1123 w_k1_KO_s012
                                 + 1.83245 w_k1_MAT_s012
                                 + 1.12811 w_k1_MSFT_s012
                                 + 1.02042 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.781832 w_k1_ADM_s012
                                 + 0.767605 w_k1_AXP_s012
                                 + 0.650182 w_k1_DIS_s012
                                 + 0.775136 w_k1_FDX_s012
                                 + 0.761696 w_k1_IBM_s012
                                 + 0.689704 w_k1_INTC_s012
                                 + 0.689843 w_k1_KO_s012
                                 + 0.952408 w_k1_MAT_s012
                                 + 0.788606 w_k1_MSFT_s012
                                 + 0.800381 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.755579 w_k1_ADM_s012
                                 + 0.73851 w_k1_AXP_s012
                                 + 0.882574 w_k1_DIS_s012
                                 + 0.863719 w_k1_FDX_s012
                                 + 0.894832 w_k1_IBM_s012
                                 + 0.882189 w_k1_INTC_s012
                                 + 0.797191 w_k1_KO_s012
                                 + 0.897956 w_k1_MAT_s012
                                 + 0.98578 w_k1_MSFT_s012
                                 + 0.821327 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.848354 w_k1_ADM_s013
                                 + 0.695125 w_k1_AXP_s013
                                 + 0.6933 w_k1_DIS_s013
                                 + 0.764853 w_k1_FDX_s013
                                 + 0.788113 w_k1_IBM_s013
                                 + 0.759891 w_k1_INTC_s013
                                 + 0.775988 w_k1_KO_s013
                                 + 0.85538 w_k1_MAT_s013
                                 + 0.746994 w_k1_MSFT_s013
                                 + 0.841991 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.894105 w_k1_ADM_s013
                                 + 0.913023 w_k1_AXP_s013
                                 + 0.814866 w_k1_DIS_s013
                                 + 0.857459 w_k1_FDX_s013
                                 + 1.07341 w_k1_IBM_s013
                                 + 0.959035 w_k1_INTC_s013
                                 + 0.91937 w_k1_KO_s013 + 1.51762 w_k1_MAT_s013
                                 + 0.916516 w_k1_MSFT_s013
                                 + 0.945353 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.898122 w_k1_ADM_s013
                                 + 0.736097 w_k1_AXP_s013
                                 + 0.830642 w_k1_DIS_s013
                                 + 0.970415 w_k1_FDX_s013
                                 + 0.60862 w_k1_IBM_s013
                                 + 0.878306 w_k1_INTC_s013
                                 + 0.713552 w_k1_KO_s013
                                 + 0.636698 w_k1_MAT_s013
                                 + 0.962924 w_k1_MSFT_s013
                                 + 0.852241 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.20986 w_k1_ADM_s013
                                 + 1.17033 w_k1_AXP_s013 + 1.2504 w_k1_DIS_s013
                                 + 1.2979 w_k1_FDX_s013
                                 + 0.801013 w_k1_IBM_s013
                                 + 1.08578 w_k1_INTC_s013
                                 + 0.889002 w_k1_KO_s013
                                 + 0.81491 w_k1_MAT_s013
                                 + 1.07568 w_k1_MSFT_s013
                                 + 1.00593 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.13308 w_k1_ADM_s013
                                 + 0.926606 w_k1_AXP_s013
                                 + 1.14526 w_k1_DIS_s013
                                 + 0.873258 w_k1_FDX_s013
                                 + 1.81473 w_k1_IBM_s013
                                 + 1.10386 w_k1_INTC_s013 + 1.2987 w_k1_KO_s013
                                 + 1.39858 w_k1_MAT_s013
                                 + 1.02866 w_k1_MSFT_s013
                                 + 0.895805 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.843602 w_k1_ADM_s013
                                 + 1.04968 w_k1_AXP_s013
                                 + 1.07891 w_k1_DIS_s013
                                 + 0.969898 w_k1_FDX_s013
                                 + 1.45047 w_k1_IBM_s013
                                 + 0.972116 w_k1_INTC_s013
                                 + 1.04892 w_k1_KO_s013 + 1.6806 w_k1_MAT_s013
                                 + 1.01845 w_k1_MSFT_s013
                                 + 0.992992 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.16842 w_k1_ADM_s013
                                 + 1.18543 w_k1_AXP_s013
                                 + 1.32345 w_k1_DIS_s013
                                 + 1.15273 w_k1_FDX_s013
                                 + 1.23492 w_k1_IBM_s013
                                 + 1.10044 w_k1_INTC_s013
                                 + 1.15454 w_k1_KO_s013 + 1.77218 w_k1_MAT_s013
                                 + 1.25957 w_k1_MSFT_s013
                                 + 1.11048 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.879374 w_k1_ADM_s013
                                 + 0.664702 w_k1_AXP_s013
                                 + 0.848624 w_k1_DIS_s013
                                 + 0.816319 w_k1_FDX_s013
                                 + 0.877492 w_k1_IBM_s013
                                 + 0.887384 w_k1_INTC_s013
                                 + 0.851413 w_k1_KO_s013
                                 + 0.944017 w_k1_MAT_s013
                                 + 0.97436 w_k1_MSFT_s013
                                 + 0.997269 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.00989 w_k1_ADM_s014
                                 + 1.01534 w_k1_AXP_s014
                                 + 0.729074 w_k1_DIS_s014
                                 + 0.807284 w_k1_FDX_s014
                                 + 1.23082 w_k1_IBM_s014
                                 + 0.854257 w_k1_INTC_s014
                                 + 0.868452 w_k1_KO_s014
                                 + 1.03367 w_k1_MAT_s014
                                 + 0.775441 w_k1_MSFT_s014
                                 + 0.848367 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.854 w_k1_ADM_s014
                                 + 0.728718 w_k1_AXP_s014
                                 + 0.669362 w_k1_DIS_s014
                                 + 0.85259 w_k1_FDX_s014
                                 + 0.55521 w_k1_IBM_s014
                                 + 0.709431 w_k1_INTC_s014
                                 + 0.64406 w_k1_KO_s014
                                 + 0.396986 w_k1_MAT_s014
                                 + 0.790018 w_k1_MSFT_s014
                                 + 0.821625 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.26471 w_k1_ADM_s014
                                 + 1.09626 w_k1_AXP_s014
                                 + 0.824145 w_k1_DIS_s014
                                 + 1.07567 w_k1_FDX_s014
                                 + 0.597256 w_k1_IBM_s014
                                 + 1.08589 w_k1_INTC_s014
                                 + 0.750494 w_k1_KO_s014
                                 + 0.559303 w_k1_MAT_s014
                                 + 0.977554 w_k1_MSFT_s014
                                 + 1.00202 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.933975 w_k1_ADM_s014
                                 + 0.913033 w_k1_AXP_s014
                                 + 0.897947 w_k1_DIS_s014
                                 + 0.969372 w_k1_FDX_s014
                                 + 0.772043 w_k1_IBM_s014
                                 + 1.18471 w_k1_INTC_s014
                                 + 0.799378 w_k1_KO_s014
                                 + 0.447937 w_k1_MAT_s014
                                 + 0.916933 w_k1_MSFT_s014
                                 + 0.913943 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.20131 w_k1_ADM_s014
                                 + 1.39358 w_k1_AXP_s014
                                 + 1.27152 w_k1_DIS_s014
                                 + 1.22425 w_k1_FDX_s014
                                 + 1.01233 w_k1_IBM_s014
                                 + 1.11821 w_k1_INTC_s014
                                 + 0.932531 w_k1_KO_s014
                                 + 1.13639 w_k1_MAT_s014
                                 + 1.04658 w_k1_MSFT_s014
                                 + 1.17704 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.56587 w_k1_ADM_s014
                                 + 1.62597 w_k1_AXP_s014
                                 + 1.80114 w_k1_DIS_s014
                                 + 1.49801 w_k1_FDX_s014
                                 + 1.32234 w_k1_IBM_s014
                                 + 1.18239 w_k1_INTC_s014
                                 + 1.34002 w_k1_KO_s014 + 1.53698 w_k1_MAT_s014
                                 + 1.40189 w_k1_MSFT_s014
                                 + 1.63356 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.935351 w_k1_ADM_s014
                                 + 1.0145 w_k1_AXP_s014
                                 + 0.837321 w_k1_DIS_s014
                                 + 0.933874 w_k1_FDX_s014
                                 + 0.999965 w_k1_IBM_s014
                                 + 0.864493 w_k1_INTC_s014
                                 + 0.969718 w_k1_KO_s014 + 1.1989 w_k1_MAT_s014
                                 + 0.964489 w_k1_MSFT_s014
                                 + 0.974892 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.07628 w_k1_ADM_s014
                                 + 1.22596 w_k1_AXP_s014 + 1.3708 w_k1_DIS_s014
                                 + 1.09752 w_k1_FDX_s014
                                 + 1.33375 w_k1_IBM_s014
                                 + 1.00276 w_k1_INTC_s014 + 1.2928 w_k1_KO_s014
                                 + 0.934915 w_k1_MAT_s014
                                 + 0.98618 w_k1_MSFT_s014
                                 + 1.01349 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.15289 w_k1_ADM_s015
                                 + 1.4451 w_k1_AXP_s015 + 1.43152 w_k1_DIS_s015
                                 + 1.47275 w_k1_FDX_s015
                                 + 0.904237 w_k1_IBM_s015
                                 + 1.49134 w_k1_INTC_s015
                                 + 1.00255 w_k1_KO_s015
                                 + 0.843222 w_k1_MAT_s015
                                 + 1.45965 w_k1_MSFT_s015
                                 + 1.01074 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.733214 w_k1_ADM_s015
                                 + 0.974118 w_k1_AXP_s015
                                 + 0.93926 w_k1_DIS_s015
                                 + 0.854513 w_k1_FDX_s015
                                 + 1.26141 w_k1_IBM_s015
                                 + 0.939783 w_k1_INTC_s015
                                 + 1.03901 w_k1_KO_s015 + 1.15775 w_k1_MAT_s015
                                 + 0.907062 w_k1_MSFT_s015
                                 + 0.875417 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.04232 w_k1_ADM_s015
                                 + 0.854916 w_k1_AXP_s015
                                 + 0.548059 w_k1_DIS_s015
                                 + 0.765729 w_k1_FDX_s015
                                 + 0.863081 w_k1_IBM_s015
                                 + 0.808789 w_k1_INTC_s015
                                 + 0.787755 w_k1_KO_s015
                                 + 1.00415 w_k1_MAT_s015
                                 + 0.759402 w_k1_MSFT_s015
                                 + 0.727294 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.17732 w_k1_ADM_s015
                                 + 1.27127 w_k1_AXP_s015 + 1.2339 w_k1_DIS_s015
                                 + 1.18932 w_k1_FDX_s015
                                 + 1.09548 w_k1_IBM_s015
                                 + 0.908029 w_k1_INTC_s015
                                 + 1.18925 w_k1_KO_s015 + 1.44382 w_k1_MAT_s015
                                 + 1.16178 w_k1_MSFT_s015
                                 + 1.10104 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.23533 w_k1_ADM_s015
                                 + 1.40367 w_k1_AXP_s015
                                 + 1.30502 w_k1_DIS_s015
                                 + 1.04832 w_k1_FDX_s015 + 1.7605 w_k1_IBM_s015
                                 + 1.00677 w_k1_INTC_s015
                                 + 1.34012 w_k1_KO_s015 + 1.67751 w_k1_MAT_s015
                                 + 1.20013 w_k1_MSFT_s015
                                 + 1.07774 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.943927 w_k1_ADM_s015
                                 + 0.819526 w_k1_AXP_s015
                                 + 0.763227 w_k1_DIS_s015
                                 + 0.823529 w_k1_FDX_s015
                                 + 0.998071 w_k1_IBM_s015
                                 + 1.00371 w_k1_INTC_s015
                                 + 1.04727 w_k1_KO_s015
                                 + 0.731136 w_k1_MAT_s015
                                 + 0.914663 w_k1_MSFT_s015
                                 + 0.886592 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.4046 w_k1_ADM_s015
                                 + 1.35335 w_k1_AXP_s015
                                 + 1.17749 w_k1_DIS_s015
                                 + 1.31242 w_k1_FDX_s015
                                 + 0.71372 w_k1_IBM_s015
                                 + 1.04413 w_k1_INTC_s015
                                 + 0.987663 w_k1_KO_s015
                                 + 0.862336 w_k1_MAT_s015
                                 + 1.14364 w_k1_MSFT_s015 + 1.0146 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.933973 w_k1_ADM_s015
                                 + 0.944671 w_k1_AXP_s015
                                 + 0.981654 w_k1_DIS_s015
                                 + 1.10302 w_k1_FDX_s015
                                 + 0.728706 w_k1_IBM_s015
                                 + 1.22246 w_k1_INTC_s015
                                 + 1.02162 w_k1_KO_s015
                                 + 0.795386 w_k1_MAT_s015
                                 + 0.995931 w_k1_MSFT_s015
                                 + 0.890411 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.19457 w_k1_ADM_s016
                                 + 1.22135 w_k1_AXP_s016
                                 + 1.40824 w_k1_DIS_s016
                                 + 1.08593 w_k1_FDX_s016
                                 + 1.61748 w_k1_IBM_s016
                                 + 1.01587 w_k1_INTC_s016
                                 + 1.45381 w_k1_KO_s016 + 1.99083 w_k1_MAT_s016
                                 + 1.20616 w_k1_MSFT_s016
                                 + 1.19683 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.958219 w_k1_ADM_s016
                                 + 0.714714 w_k1_AXP_s016
                                 + 0.727887 w_k1_DIS_s016
                                 + 0.807035 w_k1_FDX_s016
                                 + 1.10361 w_k1_IBM_s016
                                 + 0.69936 w_k1_INTC_s016
                                 + 0.898197 w_k1_KO_s016
                                 + 0.758808 w_k1_MAT_s016
                                 + 0.771578 w_k1_MSFT_s016
                                 + 0.966833 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.730003 w_k1_ADM_s016
                                 + 0.528295 w_k1_AXP_s016
                                 + 0.712243 w_k1_DIS_s016
                                 + 0.782256 w_k1_FDX_s016
                                 + 0.673401 w_k1_IBM_s016
                                 + 0.706043 w_k1_INTC_s016
                                 + 0.698234 w_k1_KO_s016
                                 + 0.681966 w_k1_MAT_s016
                                 + 0.798299 w_k1_MSFT_s016
                                 + 0.763093 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.33074 w_k1_ADM_s016
                                 + 1.07274 w_k1_AXP_s016
                                 + 1.49003 w_k1_DIS_s016
                                 + 1.37755 w_k1_FDX_s016
                                 + 0.913735 w_k1_IBM_s016
                                 + 1.15569 w_k1_INTC_s016
                                 + 1.12202 w_k1_KO_s016 + 1.21915 w_k1_MAT_s016
                                 + 1.31502 w_k1_MSFT_s016
                                 + 1.18339 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.12919 w_k1_ADM_s016
                                 + 0.742321 w_k1_AXP_s016
                                 + 0.795315 w_k1_DIS_s016
                                 + 0.920311 w_k1_FDX_s016
                                 + 0.640064 w_k1_IBM_s016
                                 + 0.801437 w_k1_INTC_s016
                                 + 0.78419 w_k1_KO_s016
                                 + 0.679957 w_k1_MAT_s016
                                 + 0.890568 w_k1_MSFT_s016
                                 + 0.964209 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.24364 w_k1_ADM_s016
                                 + 0.872215 w_k1_AXP_s016
                                 + 1.08096 w_k1_DIS_s016 + 1.114 w_k1_FDX_s016
                                 + 0.800195 w_k1_IBM_s016
                                 + 1.07146 w_k1_INTC_s016
                                 + 1.15309 w_k1_KO_s016 + 1.02764 w_k1_MAT_s016
                                 + 1.17197 w_k1_MSFT_s016
                                 + 1.06727 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.30851 w_k1_ADM_s016
                                 + 1.66087 w_k1_AXP_s016
                                 + 1.63456 w_k1_DIS_s016
                                 + 1.34832 w_k1_FDX_s016 + 1.3409 w_k1_IBM_s016
                                 + 1.40424 w_k1_INTC_s016 + 1.4507 w_k1_KO_s016
                                 + 1.73201 w_k1_MAT_s016
                                 + 1.3883 w_k1_MSFT_s016 + 1.23167 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.10105 w_k1_ADM_s016
                                 + 1.11325 w_k1_AXP_s016
                                 + 1.11068 w_k1_DIS_s016
                                 + 1.06727 w_k1_FDX_s016
                                 + 1.12776 w_k1_IBM_s016
                                 + 1.09211 w_k1_INTC_s016 + 1.1602 w_k1_KO_s016
                                 + 1.53818 w_k1_MAT_s016
                                 + 1.02775 w_k1_MSFT_s016
                                 + 1.08097 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.09926 w_k1_ADM_s017
                                 + 1.41544 w_k1_AXP_s017
                                 + 0.844906 w_k1_DIS_s017
                                 + 1.00966 w_k1_FDX_s017
                                 + 1.27879 w_k1_IBM_s017
                                 + 1.04034 w_k1_INTC_s017
                                 + 0.927121 w_k1_KO_s017
                                 + 1.63721 w_k1_MAT_s017
                                 + 0.90756 w_k1_MSFT_s017
                                 + 0.907182 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.98304 w_k1_ADM_s017
                                 + 1.13021 w_k1_AXP_s017
                                 + 0.968074 w_k1_DIS_s017
                                 + 1.08723 w_k1_FDX_s017
                                 + 0.666744 w_k1_IBM_s017
                                 + 1.28659 w_k1_INTC_s017
                                 + 0.827185 w_k1_KO_s017
                                 + 0.746165 w_k1_MAT_s017
                                 + 1.07183 w_k1_MSFT_s017
                                 + 0.897671 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.999476 w_k1_ADM_s017
                                 + 0.837448 w_k1_AXP_s017
                                 + 0.765272 w_k1_DIS_s017
                                 + 0.834845 w_k1_FDX_s017
                                 + 0.858229 w_k1_IBM_s017
                                 + 0.839737 w_k1_INTC_s017
                                 + 0.862301 w_k1_KO_s017
                                 + 0.778617 w_k1_MAT_s017
                                 + 0.915138 w_k1_MSFT_s017
                                 + 0.855179 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.26307 w_k1_ADM_s017
                                 + 1.08865 w_k1_AXP_s017
                                 + 1.35265 w_k1_DIS_s017
                                 + 1.12079 w_k1_FDX_s017
                                 + 0.850323 w_k1_IBM_s017
                                 + 1.1462 w_k1_INTC_s017 + 1.18825 w_k1_KO_s017
                                 + 0.909906 w_k1_MAT_s017
                                 + 1.31777 w_k1_MSFT_s017
                                 + 1.12673 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.983629 w_k1_ADM_s017
                                 + 1.52723 w_k1_AXP_s017
                                 + 1.48951 w_k1_DIS_s017
                                 + 1.25074 w_k1_FDX_s017
                                 + 1.78625 w_k1_IBM_s017
                                 + 1.17538 w_k1_INTC_s017
                                 + 1.44668 w_k1_KO_s017 + 1.9481 w_k1_MAT_s017
                                 + 1.26952 w_k1_MSFT_s017
                                 + 1.25472 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.764126 w_k1_ADM_s017
                                 + 0.70683 w_k1_AXP_s017
                                 + 0.47309 w_k1_DIS_s017
                                 + 0.65196 w_k1_FDX_s017
                                 + 0.738641 w_k1_IBM_s017
                                 + 0.713913 w_k1_INTC_s017
                                 + 0.6561 w_k1_KO_s017 + 0.644768 w_k1_MAT_s017
                                 + 0.68131 w_k1_MSFT_s017
                                 + 0.722075 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.47059 w_k1_ADM_s017
                                 + 1.59632 w_k1_AXP_s017
                                 + 1.39458 w_k1_DIS_s017
                                 + 1.38902 w_k1_FDX_s017
                                 + 0.977923 w_k1_IBM_s017
                                 + 1.11936 w_k1_INTC_s017
                                 + 1.19243 w_k1_KO_s017 + 1.30444 w_k1_MAT_s017
                                 + 1.34974 w_k1_MSFT_s017
                                 + 1.06901 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.11066 w_k1_ADM_s017
                                 + 1.12428 w_k1_AXP_s017
                                 + 1.24496 w_k1_DIS_s017
                                 + 1.08212 w_k1_FDX_s017
                                 + 1.19983 w_k1_IBM_s017
                                 + 0.983099 w_k1_INTC_s017
                                 + 1.18747 w_k1_KO_s017 + 1.33973 w_k1_MAT_s017
                                 + 1.25629 w_k1_MSFT_s017
                                 + 1.07067 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.900252 w_k1_ADM_s100
                                 + 0.958722 w_k1_AXP_s100
                                 + 1.05331 w_k1_DIS_s100
                                 + 1.00948 w_k1_FDX_s100
                                 + 1.28959 w_k1_IBM_s100
                                 + 1.16192 w_k1_INTC_s100
                                 + 1.28051 w_k1_KO_s100
                                 + 0.813786 w_k1_MAT_s100
                                 + 1.18305 w_k1_MSFT_s100
                                 + 0.975137 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.18328 w_k1_ADM_s100
                                 + 1.12442 w_k1_AXP_s100 + 1.0296 w_k1_DIS_s100
                                 + 1.11635 w_k1_FDX_s100 + 1.015 w_k1_IBM_s100
                                 + 0.993311 w_k1_INTC_s100
                                 + 1.02758 w_k1_KO_s100
                                 + 0.841397 w_k1_MAT_s100
                                 + 1.0035 w_k1_MSFT_s100
                                 + 0.963623 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.20783 w_k1_ADM_s100
                                 + 1.35831 w_k1_AXP_s100
                                 + 1.25645 w_k1_DIS_s100
                                 + 1.02674 w_k1_FDX_s100
                                 + 1.63069 w_k1_IBM_s100
                                 + 1.12191 w_k1_INTC_s100
                                 + 1.48421 w_k1_KO_s100 + 1.98664 w_k1_MAT_s100
                                 + 1.21924 w_k1_MSFT_s100
                                 + 1.10327 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.20291 w_k1_ADM_s100
                                 + 1.2254 w_k1_AXP_s100 + 1.42544 w_k1_DIS_s100
                                 + 1.38921 w_k1_FDX_s100
                                 + 0.976755 w_k1_IBM_s100
                                 + 1.39472 w_k1_INTC_s100
                                 + 1.39446 w_k1_KO_s100
                                 + 0.907843 w_k1_MAT_s100
                                 + 1.45314 w_k1_MSFT_s100
                                 + 1.13903 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.712897 w_k1_ADM_s100
                                 + 0.842762 w_k1_AXP_s100
                                 + 0.811489 w_k1_DIS_s100
                                 + 0.801009 w_k1_FDX_s100
                                 + 0.848054 w_k1_IBM_s100
                                 + 0.816781 w_k1_INTC_s100
                                 + 0.948201 w_k1_KO_s100
                                 + 0.884527 w_k1_MAT_s100
                                 + 0.820716 w_k1_MSFT_s100
                                 + 0.8446 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.652539 w_k1_ADM_s100
                                 + 0.781743 w_k1_AXP_s100
                                 + 0.73032 w_k1_DIS_s100
                                 + 0.774733 w_k1_FDX_s100
                                 + 1.16223 w_k1_IBM_s100
                                 + 0.880069 w_k1_INTC_s100
                                 + 1.07552 w_k1_KO_s100
                                 + 0.986001 w_k1_MAT_s100
                                 + 0.87222 w_k1_MSFT_s100
                                 + 0.912387 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.03232 w_k1_ADM_s100
                                 + 1.46666 w_k1_AXP_s100
                                 + 1.38652 w_k1_DIS_s100
                                 + 1.31583 w_k1_FDX_s100
                                 + 1.35066 w_k1_IBM_s100
                                 + 1.62541 w_k1_INTC_s100
                                 + 1.51018 w_k1_KO_s100 + 1.16135 w_k1_MAT_s100
                                 + 1.28775 w_k1_MSFT_s100
                                 + 1.33556 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.987316 w_k1_ADM_s100
                                 + 0.712246 w_k1_AXP_s100
                                 + 0.790444 w_k1_DIS_s100
                                 + 0.994227 w_k1_FDX_s100
                                 + 0.69108 w_k1_IBM_s100
                                 + 1.27762 w_k1_INTC_s100
                                 + 0.798686 w_k1_KO_s100
                                 + 0.480101 w_k1_MAT_s100
                                 + 1.14661 w_k1_MSFT_s100
                                 + 0.875675 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.17444 w_k1_ADM_s101
                                 + 1.07635 w_k1_AXP_s101
                                 + 1.27825 w_k1_DIS_s101
                                 + 1.26642 w_k1_FDX_s101
                                 + 0.754692 w_k1_IBM_s101
                                 + 1.01189 w_k1_INTC_s101
                                 + 1.01502 w_k1_KO_s101
                                 + 0.985558 w_k1_MAT_s101
                                 + 1.28124 w_k1_MSFT_s101
                                 + 1.02883 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.31526 w_k1_ADM_s101
                                 + 1.20351 w_k1_AXP_s101
                                 + 1.36873 w_k1_DIS_s101
                                 + 1.22503 w_k1_FDX_s101
                                 + 1.33291 w_k1_IBM_s101
                                 + 0.972307 w_k1_INTC_s101
                                 + 1.13796 w_k1_KO_s101 + 1.43758 w_k1_MAT_s101
                                 + 1.22067 w_k1_MSFT_s101
                                 + 1.12016 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.810384 w_k1_ADM_s101
                                 + 1.09622 w_k1_AXP_s101
                                 + 1.04707 w_k1_DIS_s101
                                 + 0.993443 w_k1_FDX_s101
                                 + 1.00577 w_k1_IBM_s101
                                 + 0.876196 w_k1_INTC_s101
                                 + 1.03446 w_k1_KO_s101 + 1.06901 w_k1_MAT_s101
                                 + 1.03213 w_k1_MSFT_s101
                                 + 1.06132 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.744568 w_k1_ADM_s101
                                 + 0.60233 w_k1_AXP_s101
                                 + 0.557839 w_k1_DIS_s101
                                 + 0.745252 w_k1_FDX_s101
                                 + 0.64296 w_k1_IBM_s101
                                 + 0.791029 w_k1_INTC_s101
                                 + 0.630995 w_k1_KO_s101
                                 + 0.543362 w_k1_MAT_s101
                                 + 0.730604 w_k1_MSFT_s101
                                 + 0.748455 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.05745 w_k1_ADM_s101
                                 + 0.997078 w_k1_AXP_s101
                                 + 0.865713 w_k1_DIS_s101
                                 + 1.03608 w_k1_FDX_s101
                                 + 0.798838 w_k1_IBM_s101
                                 + 0.962077 w_k1_INTC_s101
                                 + 0.821947 w_k1_KO_s101
                                 + 0.895811 w_k1_MAT_s101
                                 + 1.13532 w_k1_MSFT_s101
                                 + 0.916366 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.770445 w_k1_ADM_s101
                                 + 0.842966 w_k1_AXP_s101
                                 + 0.609302 w_k1_DIS_s101
                                 + 0.756493 w_k1_FDX_s101
                                 + 0.966382 w_k1_IBM_s101
                                 + 0.69095 w_k1_INTC_s101
                                 + 0.789955 w_k1_KO_s101
                                 + 1.13618 w_k1_MAT_s101
                                 + 0.779689 w_k1_MSFT_s101
                                 + 0.789281 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.48501 w_k1_ADM_s101
                                 + 1.4781 w_k1_AXP_s101 + 1.87033 w_k1_DIS_s101
                                 + 1.4733 w_k1_FDX_s101 + 1.21738 w_k1_IBM_s101
                                 + 1.13135 w_k1_INTC_s101
                                 + 1.43696 w_k1_KO_s101 + 1.3401 w_k1_MAT_s101
                                 + 1.5586 w_k1_MSFT_s101 + 1.26897 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.724197 w_k1_ADM_s101
                                 + 0.831662 w_k1_AXP_s101
                                 + 0.872078 w_k1_DIS_s101
                                 + 0.827246 w_k1_FDX_s101
                                 + 1.40183 w_k1_IBM_s101
                                 + 0.904385 w_k1_INTC_s101
                                 + 1.0158 w_k1_KO_s101 + 1.08028 w_k1_MAT_s101
                                 + 0.989664 w_k1_MSFT_s101
                                 + 0.883474 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.00255 w_k1_ADM_s102
                                 + 0.775168 w_k1_AXP_s102
                                 + 0.62342 w_k1_DIS_s102
                                 + 0.761955 w_k1_FDX_s102
                                 + 1.05079 w_k1_IBM_s102
                                 + 0.687497 w_k1_INTC_s102
                                 + 0.872248 w_k1_KO_s102
                                 + 0.940057 w_k1_MAT_s102
                                 + 0.781765 w_k1_MSFT_s102
                                 + 0.872461 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.30019 w_k1_ADM_s102
                                 + 1.16936 w_k1_AXP_s102
                                 + 1.31335 w_k1_DIS_s102
                                 + 1.26339 w_k1_FDX_s102
                                 + 0.89937 w_k1_IBM_s102
                                 + 1.34345 w_k1_INTC_s102
                                 + 1.27247 w_k1_KO_s102
                                 + 0.750226 w_k1_MAT_s102
                                 + 1.22601 w_k1_MSFT_s102
                                 + 1.13535 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.10607 w_k1_ADM_s102
                                 + 0.693324 w_k1_AXP_s102
                                 + 0.750901 w_k1_DIS_s102
                                 + 0.867766 w_k1_FDX_s102
                                 + 0.595642 w_k1_IBM_s102
                                 + 0.749055 w_k1_INTC_s102
                                 + 0.764119 w_k1_KO_s102
                                 + 0.511057 w_k1_MAT_s102
                                 + 0.821353 w_k1_MSFT_s102
                                 + 0.833434 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.18328 w_k1_ADM_s102
                                 + 1.29901 w_k1_AXP_s102
                                 + 1.06522 w_k1_DIS_s102
                                 + 0.948602 w_k1_FDX_s102
                                 + 1.56322 w_k1_IBM_s102
                                 + 0.962412 w_k1_INTC_s102
                                 + 1.15966 w_k1_KO_s102 + 1.17288 w_k1_MAT_s102
                                 + 0.82185 w_k1_MSFT_s102
                                 + 0.987805 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.05468 w_k1_ADM_s102
                                 + 0.867885 w_k1_AXP_s102
                                 + 0.956885 w_k1_DIS_s102
                                 + 0.88424 w_k1_FDX_s102
                                 + 1.11485 w_k1_IBM_s102
                                 + 0.911882 w_k1_INTC_s102
                                 + 1.12886 w_k1_KO_s102
                                 + 0.837981 w_k1_MAT_s102
                                 + 0.944446 w_k1_MSFT_s102
                                 + 0.996423 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.3876 w_k1_ADM_s102
                                 + 1.25253 w_k1_AXP_s102
                                 + 1.82498 w_k1_DIS_s102
                                 + 1.48936 w_k1_FDX_s102
                                 + 1.23099 w_k1_IBM_s102
                                 + 1.38911 w_k1_INTC_s102
                                 + 1.62733 w_k1_KO_s102 + 1.13423 w_k1_MAT_s102
                                 + 1.52302 w_k1_MSFT_s102
                                 + 1.28153 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.2611 w_k1_ADM_s102
                                 + 0.892303 w_k1_AXP_s102
                                 + 0.902363 w_k1_DIS_s102
                                 + 1.06715 w_k1_FDX_s102
                                 + 0.811169 w_k1_IBM_s102
                                 + 0.939457 w_k1_INTC_s102
                                 + 0.911511 w_k1_KO_s102
                                 + 0.666764 w_k1_MAT_s102
                                 + 0.890124 w_k1_MSFT_s102
                                 + 0.932606 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06435 w_k1_ADM_s102
                                 + 1.19788 w_k1_AXP_s102
                                 + 1.50408 w_k1_DIS_s102
                                 + 1.11507 w_k1_FDX_s102
                                 + 1.25459 w_k1_IBM_s102
                                 + 1.11323 w_k1_INTC_s102
                                 + 1.47199 w_k1_KO_s102 + 1.14232 w_k1_MAT_s102
                                 + 1.17916 w_k1_MSFT_s102
                                 + 1.00774 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.52251 w_k1_ADM_s103
                                 + 1.39512 w_k1_AXP_s103
                                 + 1.73417 w_k1_DIS_s103
                                 + 1.41363 w_k1_FDX_s103
                                 + 1.29098 w_k1_IBM_s103
                                 + 1.44236 w_k1_INTC_s103
                                 + 1.19071 w_k1_KO_s103 + 1.34429 w_k1_MAT_s103
                                 + 1.44012 w_k1_MSFT_s103
                                 + 1.19485 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.0734 w_k1_ADM_s103
                                 + 0.821135 w_k1_AXP_s103
                                 + 0.73393 w_k1_DIS_s103
                                 + 0.865959 w_k1_FDX_s103
                                 + 0.993263 w_k1_IBM_s103
                                 + 0.791229 w_k1_INTC_s103
                                 + 0.835183 w_k1_KO_s103
                                 + 1.17742 w_k1_MAT_s103
                                 + 0.854494 w_k1_MSFT_s103
                                 + 0.893397 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.927676 w_k1_ADM_s103
                                 + 0.763111 w_k1_AXP_s103
                                 + 0.838264 w_k1_DIS_s103
                                 + 0.925844 w_k1_FDX_s103
                                 + 0.742293 w_k1_IBM_s103
                                 + 0.672317 w_k1_INTC_s103
                                 + 0.835468 w_k1_KO_s103
                                 + 0.745025 w_k1_MAT_s103
                                 + 0.790866 w_k1_MSFT_s103
                                 + 0.883084 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.729984 w_k1_ADM_s103
                                 + 0.573813 w_k1_AXP_s103
                                 + 0.721529 w_k1_DIS_s103
                                 + 0.831138 w_k1_FDX_s103
                                 + 0.718142 w_k1_IBM_s103
                                 + 0.813144 w_k1_INTC_s103
                                 + 0.771852 w_k1_KO_s103
                                 + 0.614947 w_k1_MAT_s103
                                 + 0.974436 w_k1_MSFT_s103
                                 + 0.881767 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.984744 w_k1_ADM_s103
                                 + 1.09132 w_k1_AXP_s103
                                 + 1.05374 w_k1_DIS_s103 + 1.1403 w_k1_FDX_s103
                                 + 0.853108 w_k1_IBM_s103
                                 + 1.05495 w_k1_INTC_s103
                                 + 1.00143 w_k1_KO_s103 + 1.10556 w_k1_MAT_s103
                                 + 1.18554 w_k1_MSFT_s103
                                 + 0.997447 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.02925 w_k1_ADM_s103
                                 + 1.00492 w_k1_AXP_s103
                                 + 1.04477 w_k1_DIS_s103
                                 + 0.911363 w_k1_FDX_s103
                                 + 1.3431 w_k1_IBM_s103
                                 + 1.03128 w_k1_INTC_s103
                                 + 1.11627 w_k1_KO_s103 + 1.36041 w_k1_MAT_s103
                                 + 1.09256 w_k1_MSFT_s103
                                 + 1.02658 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.19011 w_k1_ADM_s103
                                 + 1.09163 w_k1_AXP_s103
                                 + 1.37761 w_k1_DIS_s103
                                 + 1.09757 w_k1_FDX_s103
                                 + 1.07546 w_k1_IBM_s103
                                 + 0.979256 w_k1_INTC_s103
                                 + 1.17356 w_k1_KO_s103 + 1.2318 w_k1_MAT_s103
                                 + 1.13968 w_k1_MSFT_s103
                                 + 1.09223 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.40669 w_k1_ADM_s103
                                 + 1.24921 w_k1_AXP_s103
                                 + 1.55611 w_k1_DIS_s103 + 1.2304 w_k1_FDX_s103
                                 + 1.38458 w_k1_IBM_s103
                                 + 0.96978 w_k1_INTC_s103
                                 + 1.31297 w_k1_KO_s103 + 1.61427 w_k1_MAT_s103
                                 + 1.12701 w_k1_MSFT_s103
                                 + 1.13463 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.01517 w_k1_ADM_s104
                                 + 0.978059 w_k1_AXP_s104
                                 + 0.935093 w_k1_DIS_s104
                                 + 1.10246 w_k1_FDX_s104
                                 + 0.767324 w_k1_IBM_s104
                                 + 1.03193 w_k1_INTC_s104
                                 + 0.825391 w_k1_KO_s104
                                 + 0.609843 w_k1_MAT_s104
                                 + 1.00779 w_k1_MSFT_s104
                                 + 0.988806 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.747836 w_k1_ADM_s104
                                 + 0.781577 w_k1_AXP_s104
                                 + 0.762761 w_k1_DIS_s104
                                 + 0.847142 w_k1_FDX_s104
                                 + 1.05383 w_k1_IBM_s104
                                 + 0.812933 w_k1_INTC_s104
                                 + 0.71993 w_k1_KO_s104 + 0.85944 w_k1_MAT_s104
                                 + 0.850253 w_k1_MSFT_s104
                                 + 0.961678 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.02009 w_k1_ADM_s104
                                 + 0.947393 w_k1_AXP_s104
                                 + 1.16584 w_k1_DIS_s104
                                 + 1.02345 w_k1_FDX_s104
                                 + 1.20421 w_k1_IBM_s104
                                 + 1.0337 w_k1_INTC_s104 + 1.07395 w_k1_KO_s104
                                 + 0.923063 w_k1_MAT_s104
                                 + 1.01361 w_k1_MSFT_s104
                                 + 1.16126 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.1205 w_k1_ADM_s104
                                 + 1.46175 w_k1_AXP_s104
                                 + 1.15151 w_k1_DIS_s104
                                 + 1.14807 w_k1_FDX_s104
                                 + 0.924474 w_k1_IBM_s104
                                 + 1.02413 w_k1_INTC_s104
                                 + 0.955839 w_k1_KO_s104
                                 + 1.16049 w_k1_MAT_s104
                                 + 1.05436 w_k1_MSFT_s104
                                 + 1.16389 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.19219 w_k1_ADM_s104
                                 + 1.25905 w_k1_AXP_s104
                                 + 1.78214 w_k1_DIS_s104
                                 + 1.36168 w_k1_FDX_s104 + 1.2522 w_k1_IBM_s104
                                 + 1.23643 w_k1_INTC_s104
                                 + 1.26126 w_k1_KO_s104 + 1.01754 w_k1_MAT_s104
                                 + 1.3287 w_k1_MSFT_s104 + 1.21639 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.902886 w_k1_ADM_s104
                                 + 0.628044 w_k1_AXP_s104
                                 + 0.677035 w_k1_DIS_s104
                                 + 0.844469 w_k1_FDX_s104
                                 + 0.748656 w_k1_IBM_s104
                                 + 0.714946 w_k1_INTC_s104
                                 + 0.689293 w_k1_KO_s104
                                 + 0.621758 w_k1_MAT_s104
                                 + 0.770708 w_k1_MSFT_s104
                                 + 0.932503 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.25003 w_k1_ADM_s104
                                 + 1.02702 w_k1_AXP_s104
                                 + 1.46317 w_k1_DIS_s104
                                 + 1.39138 w_k1_FDX_s104
                                 + 0.861751 w_k1_IBM_s104
                                 + 1.21221 w_k1_INTC_s104
                                 + 1.01865 w_k1_KO_s104
                                 + 0.849988 w_k1_MAT_s104
                                 + 1.24029 w_k1_MSFT_s104
                                 + 1.19777 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.919436 w_k1_ADM_s104
                                 + 1.37881 w_k1_AXP_s104
                                 + 1.13445 w_k1_DIS_s104
                                 + 1.09347 w_k1_FDX_s104 + 1.5426 w_k1_IBM_s104
                                 + 1.19141 w_k1_INTC_s104
                                 + 1.24465 w_k1_KO_s104 + 1.59329 w_k1_MAT_s104
                                 + 1.16261 w_k1_MSFT_s104
                                 + 1.19715 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.04102 w_k1_ADM_s105
                                 + 0.930129 w_k1_AXP_s105
                                 + 1.17518 w_k1_DIS_s105
                                 + 1.14321 w_k1_FDX_s105
                                 + 0.89452 w_k1_IBM_s105
                                 + 1.08915 w_k1_INTC_s105
                                 + 1.01645 w_k1_KO_s105
                                 + 0.483241 w_k1_MAT_s105
                                 + 1.12171 w_k1_MSFT_s105
                                 + 0.974991 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.994134 w_k1_ADM_s105
                                 + 0.940058 w_k1_AXP_s105
                                 + 1.16493 w_k1_DIS_s105
                                 + 1.02695 w_k1_FDX_s105
                                 + 1.14476 w_k1_IBM_s105
                                 + 1.09383 w_k1_INTC_s105
                                 + 1.20521 w_k1_KO_s105 + 0.93256 w_k1_MAT_s105
                                 + 1.06565 w_k1_MSFT_s105
                                 + 1.04179 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.946705 w_k1_ADM_s105
                                 + 0.837103 w_k1_AXP_s105
                                 + 0.768923 w_k1_DIS_s105
                                 + 0.962622 w_k1_FDX_s105
                                 + 0.65506 w_k1_IBM_s105
                                 + 0.955075 w_k1_INTC_s105
                                 + 0.716401 w_k1_KO_s105
                                 + 0.566467 w_k1_MAT_s105
                                 + 1.00135 w_k1_MSFT_s105
                                 + 0.909651 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.847274 w_k1_ADM_s105
                                 + 1.18791 w_k1_AXP_s105
                                 + 0.907968 w_k1_DIS_s105
                                 + 0.979533 w_k1_FDX_s105
                                 + 1.13024 w_k1_IBM_s105
                                 + 1.08962 w_k1_INTC_s105
                                 + 0.964616 w_k1_KO_s105
                                 + 1.00727 w_k1_MAT_s105
                                 + 0.897011 w_k1_MSFT_s105
                                 + 1.00013 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.18183 w_k1_ADM_s105
                                 + 1.30384 w_k1_AXP_s105
                                 + 1.63254 w_k1_DIS_s105
                                 + 1.36742 w_k1_FDX_s105
                                 + 1.04296 w_k1_IBM_s105
                                 + 1.52677 w_k1_INTC_s105
                                 + 1.17352 w_k1_KO_s105
                                 + 0.862959 w_k1_MAT_s105
                                 + 1.35496 w_k1_MSFT_s105
                                 + 1.07118 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.863122 w_k1_ADM_s105
                                 + 0.615793 w_k1_AXP_s105
                                 + 0.802628 w_k1_DIS_s105
                                 + 0.800674 w_k1_FDX_s105
                                 + 0.973529 w_k1_IBM_s105
                                 + 0.786549 w_k1_INTC_s105
                                 + 0.911698 w_k1_KO_s105
                                 + 0.717986 w_k1_MAT_s105
                                 + 0.853097 w_k1_MSFT_s105
                                 + 0.836226 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.97032 w_k1_ADM_s105
                                 + 0.999488 w_k1_AXP_s105
                                 + 0.863264 w_k1_DIS_s105
                                 + 0.851438 w_k1_FDX_s105
                                 + 0.916801 w_k1_IBM_s105
                                 + 0.839815 w_k1_INTC_s105
                                 + 0.862279 w_k1_KO_s105
                                 + 0.901095 w_k1_MAT_s105
                                 + 0.821161 w_k1_MSFT_s105
                                 + 0.932901 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.981086 w_k1_ADM_s105
                                 + 1.21128 w_k1_AXP_s105
                                 + 1.40443 w_k1_DIS_s105
                                 + 1.05323 w_k1_FDX_s105 + 1.5753 w_k1_IBM_s105
                                 + 1.07441 w_k1_INTC_s105
                                 + 1.31504 w_k1_KO_s105 + 1.24163 w_k1_MAT_s105
                                 + 1.12983 w_k1_MSFT_s105
                                 + 1.11243 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.94235 w_k1_ADM_s106
                                 + 0.863905 w_k1_AXP_s106
                                 + 0.750614 w_k1_DIS_s106
                                 + 0.94121 w_k1_FDX_s106
                                 + 0.681506 w_k1_IBM_s106
                                 + 1.00641 w_k1_INTC_s106
                                 + 0.721055 w_k1_KO_s106
                                 + 0.596198 w_k1_MAT_s106
                                 + 0.980677 w_k1_MSFT_s106
                                 + 0.830786 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.09815 w_k1_ADM_s106
                                 + 1.1293 w_k1_AXP_s106 + 1.15 w_k1_DIS_s106
                                 + 1.20861 w_k1_FDX_s106
                                 + 0.817164 w_k1_IBM_s106
                                 + 1.42303 w_k1_INTC_s106
                                 + 0.899427 w_k1_KO_s106
                                 + 0.710476 w_k1_MAT_s106
                                 + 1.13894 w_k1_MSFT_s106
                                 + 0.962274 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.860849 w_k1_ADM_s106
                                 + 0.905332 w_k1_AXP_s106
                                 + 0.766687 w_k1_DIS_s106
                                 + 0.852164 w_k1_FDX_s106
                                 + 1.0639 w_k1_IBM_s106
                                 + 0.912712 w_k1_INTC_s106
                                 + 0.851895 w_k1_KO_s106
                                 + 1.07767 w_k1_MAT_s106
                                 + 0.831956 w_k1_MSFT_s106
                                 + 0.952572 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.75493 w_k1_ADM_s106
                                 + 0.80623 w_k1_AXP_s106
                                 + 0.626022 w_k1_DIS_s106
                                 + 0.782614 w_k1_FDX_s106
                                 + 1.19896 w_k1_IBM_s106
                                 + 0.865636 w_k1_INTC_s106
                                 + 0.930923 w_k1_KO_s106
                                 + 1.15289 w_k1_MAT_s106
                                 + 0.73324 w_k1_MSFT_s106
                                 + 0.824673 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.32783 w_k1_ADM_s106
                                 + 1.59669 w_k1_AXP_s106
                                 + 1.91624 w_k1_DIS_s106
                                 + 1.61845 w_k1_FDX_s106
                                 + 1.44471 w_k1_IBM_s106
                                 + 1.25317 w_k1_INTC_s106
                                 + 1.39508 w_k1_KO_s106 + 1.82095 w_k1_MAT_s106
                                 + 1.38606 w_k1_MSFT_s106 + 1.324 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.01293 w_k1_ADM_s106
                                 + 0.853709 w_k1_AXP_s106
                                 + 1.00712 w_k1_DIS_s106
                                 + 1.00605 w_k1_FDX_s106
                                 + 1.13732 w_k1_IBM_s106
                                 + 0.980091 w_k1_INTC_s106
                                 + 1.05635 w_k1_KO_s106
                                 + 0.974466 w_k1_MAT_s106
                                 + 1.05326 w_k1_MSFT_s106
                                 + 1.01231 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.06984 w_k1_ADM_s106
                                 + 1.23668 w_k1_AXP_s106
                                 + 1.21008 w_k1_DIS_s106
                                 + 1.10025 w_k1_FDX_s106 + 1.4681 w_k1_IBM_s106
                                 + 1.25357 w_k1_INTC_s106
                                 + 1.18006 w_k1_KO_s106 + 1.07921 w_k1_MAT_s106
                                 + 1.08031 w_k1_MSFT_s106 + 1.0819 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.746436 w_k1_ADM_s106
                                 + 0.588552 w_k1_AXP_s106
                                 + 0.538469 w_k1_DIS_s106
                                 + 0.77443 w_k1_FDX_s106
                                 + 0.655577 w_k1_IBM_s106
                                 + 0.805534 w_k1_INTC_s106
                                 + 0.627734 w_k1_KO_s106
                                 + 0.661902 w_k1_MAT_s106
                                 + 0.753336 w_k1_MSFT_s106
                                 + 0.732582 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.596513 w_k1_ADM_s107
                                 + 0.660274 w_k1_AXP_s107
                                 + 0.761263 w_k1_DIS_s107
                                 + 0.828407 w_k1_FDX_s107
                                 + 0.708558 w_k1_IBM_s107
                                 + 0.878227 w_k1_INTC_s107
                                 + 0.782887 w_k1_KO_s107
                                 + 0.703401 w_k1_MAT_s107
                                 + 0.789908 w_k1_MSFT_s107
                                 + 0.773878 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.36158 w_k1_ADM_s107
                                 + 1.07401 w_k1_AXP_s107
                                 + 1.26168 w_k1_DIS_s107
                                 + 1.27773 w_k1_FDX_s107
                                 + 0.708385 w_k1_IBM_s107
                                 + 0.923683 w_k1_INTC_s107
                                 + 0.947282 w_k1_KO_s107
                                 + 0.98442 w_k1_MAT_s107
                                 + 1.04093 w_k1_MSFT_s107
                                 + 1.05375 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.917719 w_k1_ADM_s107
                                 + 1.1635 w_k1_AXP_s107
                                 + 0.935741 w_k1_DIS_s107
                                 + 0.962568 w_k1_FDX_s107
                                 + 1.3659 w_k1_IBM_s107
                                 + 0.87018 w_k1_INTC_s107
                                 + 0.982821 w_k1_KO_s107
                                 + 1.40419 w_k1_MAT_s107
                                 + 0.871536 w_k1_MSFT_s107
                                 + 0.980764 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.43817 w_k1_ADM_s107
                                 + 1.51883 w_k1_AXP_s107
                                 + 1.45301 w_k1_DIS_s107
                                 + 1.44342 w_k1_FDX_s107
                                 + 1.02167 w_k1_IBM_s107
                                 + 1.25909 w_k1_INTC_s107
                                 + 1.06546 w_k1_KO_s107 + 1.11625 w_k1_MAT_s107
                                 + 1.25295 w_k1_MSFT_s107
                                 + 1.13452 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.86138 w_k1_ADM_s107
                                 + 0.99881 w_k1_AXP_s107
                                 + 0.774902 w_k1_DIS_s107
                                 + 0.953281 w_k1_FDX_s107
                                 + 0.83167 w_k1_IBM_s107
                                 + 0.657258 w_k1_INTC_s107
                                 + 0.792364 w_k1_KO_s107
                                 + 1.05808 w_k1_MAT_s107
                                 + 0.728086 w_k1_MSFT_s107
                                 + 0.924681 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.894503 w_k1_ADM_s107
                                 + 0.811416 w_k1_AXP_s107
                                 + 0.941885 w_k1_DIS_s107
                                 + 0.8798 w_k1_FDX_s107 + 1.46477 w_k1_IBM_s107
                                 + 0.888458 w_k1_INTC_s107
                                 + 1.06178 w_k1_KO_s107 + 1.30791 w_k1_MAT_s107
                                 + 0.879054 w_k1_MSFT_s107
                                 + 0.954133 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.800852 w_k1_ADM_s107
                                 + 0.975437 w_k1_AXP_s107
                                 + 1.22595 w_k1_DIS_s107
                                 + 1.00439 w_k1_FDX_s107
                                 + 1.03327 w_k1_IBM_s107
                                 + 1.06952 w_k1_INTC_s107
                                 + 1.06848 w_k1_KO_s107 + 1.05689 w_k1_MAT_s107
                                 + 1.19675 w_k1_MSFT_s107
                                 + 0.954681 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.06187 w_k1_ADM_s107
                                 + 1.33625 w_k1_AXP_s107
                                 + 1.78287 w_k1_DIS_s107
                                 + 1.43275 w_k1_FDX_s107
                                 + 1.35632 w_k1_IBM_s107
                                 + 1.09064 w_k1_INTC_s107
                                 + 1.37003 w_k1_KO_s107 + 1.41951 w_k1_MAT_s107
                                 + 1.23951 w_k1_MSFT_s107
                                 + 1.04325 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.83408 w_k1_ADM_s110
                                 + 0.998146 w_k1_AXP_s110
                                 + 0.893839 w_k1_DIS_s110
                                 + 0.909669 w_k1_FDX_s110
                                 + 1.25599 w_k1_IBM_s110
                                 + 0.868188 w_k1_INTC_s110
                                 + 1.02821 w_k1_KO_s110 + 1.24814 w_k1_MAT_s110
                                 + 0.926216 w_k1_MSFT_s110
                                 + 0.965402 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.914817 w_k1_ADM_s110
                                 + 0.615903 w_k1_AXP_s110
                                 + 1.02643 w_k1_DIS_s110
                                 + 0.862063 w_k1_FDX_s110
                                 + 0.83214 w_k1_IBM_s110
                                 + 0.965428 w_k1_INTC_s110
                                 + 0.882362 w_k1_KO_s110
                                 + 0.606872 w_k1_MAT_s110
                                 + 1.10549 w_k1_MSFT_s110
                                 + 0.864448 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.762327 w_k1_ADM_s110
                                 + 0.787912 w_k1_AXP_s110
                                 + 0.599649 w_k1_DIS_s110
                                 + 0.786408 w_k1_FDX_s110
                                 + 0.732983 w_k1_IBM_s110
                                 + 0.850884 w_k1_INTC_s110
                                 + 0.639408 w_k1_KO_s110
                                 + 0.492231 w_k1_MAT_s110
                                 + 0.813335 w_k1_MSFT_s110
                                 + 0.693324 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.976354 w_k1_ADM_s110
                                 + 0.953859 w_k1_AXP_s110
                                 + 0.666016 w_k1_DIS_s110
                                 + 0.800216 w_k1_FDX_s110
                                 + 0.982616 w_k1_IBM_s110
                                 + 0.71473 w_k1_INTC_s110
                                 + 0.738311 w_k1_KO_s110
                                 + 1.19971 w_k1_MAT_s110
                                 + 0.760223 w_k1_MSFT_s110
                                 + 0.814317 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.849194 w_k1_ADM_s110
                                 + 0.825754 w_k1_AXP_s110
                                 + 1.16058 w_k1_DIS_s110
                                 + 0.898697 w_k1_FDX_s110
                                 + 1.41495 w_k1_IBM_s110
                                 + 0.870794 w_k1_INTC_s110
                                 + 1.19241 w_k1_KO_s110 + 1.33998 w_k1_MAT_s110
                                 + 1.08047 w_k1_MSFT_s110
                                 + 0.935958 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.1015 w_k1_ADM_s110
                                 + 1.20652 w_k1_AXP_s110
                                 + 1.39651 w_k1_DIS_s110
                                 + 1.24144 w_k1_FDX_s110
                                 + 1.13115 w_k1_IBM_s110
                                 + 1.28844 w_k1_INTC_s110 + 1.1994 w_k1_KO_s110
                                 + 0.890209 w_k1_MAT_s110
                                 + 1.27737 w_k1_MSFT_s110
                                 + 1.08415 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.01329 w_k1_ADM_s110
                                 + 1.46329 w_k1_AXP_s110 + 1.2437 w_k1_DIS_s110
                                 + 1.24 w_k1_FDX_s110 + 1.32603 w_k1_IBM_s110
                                 + 1.21478 w_k1_INTC_s110
                                 + 1.17825 w_k1_KO_s110 + 1.83234 w_k1_MAT_s110
                                 + 1.10439 w_k1_MSFT_s110
                                 + 1.05641 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.957943 w_k1_ADM_s110
                                 + 0.997279 w_k1_AXP_s110
                                 + 0.863617 w_k1_DIS_s110
                                 + 1.07115 w_k1_FDX_s110
                                 + 0.724166 w_k1_IBM_s110
                                 + 0.979778 w_k1_INTC_s110
                                 + 0.730633 w_k1_KO_s110
                                 + 0.728015 w_k1_MAT_s110
                                 + 1.00938 w_k1_MSFT_s110
                                 + 0.917809 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.047 w_k1_ADM_s111
                                 + 1.12128 w_k1_AXP_s111
                                 + 1.18769 w_k1_DIS_s111
                                 + 1.07085 w_k1_FDX_s111
                                 + 1.20587 w_k1_IBM_s111
                                 + 0.914289 w_k1_INTC_s111
                                 + 1.1072 w_k1_KO_s111 + 1.32247 w_k1_MAT_s111
                                 + 1.01975 w_k1_MSFT_s111
                                 + 1.17228 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.666746 w_k1_ADM_s111
                                 + 0.553703 w_k1_AXP_s111
                                 + 0.814437 w_k1_DIS_s111
                                 + 0.813229 w_k1_FDX_s111
                                 + 0.665259 w_k1_IBM_s111
                                 + 0.896679 w_k1_INTC_s111
                                 + 0.840042 w_k1_KO_s111
                                 + 0.718208 w_k1_MAT_s111
                                 + 0.811089 w_k1_MSFT_s111
                                 + 0.879313 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.718904 w_k1_ADM_s111
                                 + 0.688949 w_k1_AXP_s111
                                 + 0.806127 w_k1_DIS_s111
                                 + 0.979418 w_k1_FDX_s111
                                 + 0.626558 w_k1_IBM_s111
                                 + 0.944939 w_k1_INTC_s111
                                 + 0.701217 w_k1_KO_s111
                                 + 0.594181 w_k1_MAT_s111
                                 + 0.869656 w_k1_MSFT_s111
                                 + 0.785913 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.30953 w_k1_ADM_s111
                                 + 1.10813 w_k1_AXP_s111
                                 + 1.43228 w_k1_DIS_s111
                                 + 1.29394 w_k1_FDX_s111
                                 + 0.958775 w_k1_IBM_s111
                                 + 1.25854 w_k1_INTC_s111
                                 + 1.12671 w_k1_KO_s111 + 1.02382 w_k1_MAT_s111
                                 + 1.15595 w_k1_MSFT_s111
                                 + 1.11454 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.962444 w_k1_ADM_s111
                                 + 0.962768 w_k1_AXP_s111
                                 + 0.919249 w_k1_DIS_s111
                                 + 0.922295 w_k1_FDX_s111
                                 + 1.0819 w_k1_IBM_s111
                                 + 0.875092 w_k1_INTC_s111
                                 + 0.995339 w_k1_KO_s111
                                 + 0.98773 w_k1_MAT_s111
                                 + 0.855132 w_k1_MSFT_s111
                                 + 0.853054 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.838029 w_k1_ADM_s111
                                 + 0.846116 w_k1_AXP_s111
                                 + 0.684629 w_k1_DIS_s111
                                 + 0.802201 w_k1_FDX_s111
                                 + 0.712379 w_k1_IBM_s111
                                 + 0.766318 w_k1_INTC_s111
                                 + 0.667786 w_k1_KO_s111
                                 + 0.725021 w_k1_MAT_s111
                                 + 0.715299 w_k1_MSFT_s111
                                 + 0.849437 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.09444 w_k1_ADM_s111
                                 + 1.20319 w_k1_AXP_s111
                                 + 1.60424 w_k1_DIS_s111
                                 + 1.17474 w_k1_FDX_s111
                                 + 1.62493 w_k1_IBM_s111
                                 + 1.03616 w_k1_INTC_s111
                                 + 1.37462 w_k1_KO_s111 + 1.65509 w_k1_MAT_s111
                                 + 1.23787 w_k1_MSFT_s111 + 1.2185 w_k1_PG_s111
                                 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.11615 w_k1_ADM_s111
                                 + 0.82276 w_k1_AXP_s111 + 1.0194 w_k1_DIS_s111
                                 + 1.01296 w_k1_FDX_s111
                                 + 0.801165 w_k1_IBM_s111
                                 + 0.952302 w_k1_INTC_s111
                                 + 1.02341 w_k1_KO_s111 + 0.93088 w_k1_MAT_s111
                                 + 1.07375 w_k1_MSFT_s111
                                 + 0.96596 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.703887 w_k1_ADM_s112
                                 + 0.757638 w_k1_AXP_s112
                                 + 0.791235 w_k1_DIS_s112
                                 + 0.825744 w_k1_FDX_s112
                                 + 1.03915 w_k1_IBM_s112
                                 + 0.920333 w_k1_INTC_s112
                                 + 0.910382 w_k1_KO_s112
                                 + 0.774584 w_k1_MAT_s112
                                 + 0.817513 w_k1_MSFT_s112
                                 + 0.876454 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.866104 w_k1_ADM_s112
                                 + 0.870559 w_k1_AXP_s112
                                 + 0.908166 w_k1_DIS_s112
                                 + 0.985343 w_k1_FDX_s112
                                 + 0.806081 w_k1_IBM_s112
                                 + 1.0572 w_k1_INTC_s112
                                 + 0.945232 w_k1_KO_s112
                                 + 0.819747 w_k1_MAT_s112
                                 + 0.978151 w_k1_MSFT_s112
                                 + 0.964326 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.13266 w_k1_ADM_s112
                                 + 1.19245 w_k1_AXP_s112
                                 + 1.20675 w_k1_DIS_s112
                                 + 1.24453 w_k1_FDX_s112
                                 + 0.821665 w_k1_IBM_s112
                                 + 1.25196 w_k1_INTC_s112
                                 + 1.05528 w_k1_KO_s112
                                 + 0.942388 w_k1_MAT_s112
                                 + 1.11253 w_k1_MSFT_s112
                                 + 1.04929 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.799484 w_k1_ADM_s112
                                 + 0.56799 w_k1_AXP_s112
                                 + 0.629338 w_k1_DIS_s112
                                 + 0.709897 w_k1_FDX_s112
                                 + 0.721244 w_k1_IBM_s112
                                 + 0.756072 w_k1_INTC_s112
                                 + 0.730512 w_k1_KO_s112
                                 + 0.581658 w_k1_MAT_s112
                                 + 0.783734 w_k1_MSFT_s112
                                 + 0.814631 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.0174 w_k1_ADM_s112
                                 + 1.11643 w_k1_AXP_s112
                                 + 1.11505 w_k1_DIS_s112
                                 + 1.04877 w_k1_FDX_s112
                                 + 1.49525 w_k1_IBM_s112
                                 + 0.87959 w_k1_INTC_s112
                                 + 1.00246 w_k1_KO_s112 + 1.39375 w_k1_MAT_s112
                                 + 0.903078 w_k1_MSFT_s112
                                 + 1.0879 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.951022 w_k1_ADM_s112
                                 + 1.3984 w_k1_AXP_s112 + 1.55397 w_k1_DIS_s112
                                 + 1.14119 w_k1_FDX_s112
                                 + 1.79952 w_k1_IBM_s112
                                 + 1.12938 w_k1_INTC_s112 + 1.4941 w_k1_KO_s112
                                 + 1.63054 w_k1_MAT_s112
                                 + 1.1054 w_k1_MSFT_s112 + 1.31956 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.30703 w_k1_ADM_s112
                                 + 1.52769 w_k1_AXP_s112
                                 + 1.92148 w_k1_DIS_s112
                                 + 1.57275 w_k1_FDX_s112
                                 + 1.28311 w_k1_IBM_s112
                                 + 1.57998 w_k1_INTC_s112
                                 + 1.55463 w_k1_KO_s112 + 1.294 w_k1_MAT_s112
                                 + 1.27194 w_k1_MSFT_s112
                                 + 1.29808 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.09323 w_k1_ADM_s112
                                 + 0.972932 w_k1_AXP_s112
                                 + 1.20133 w_k1_DIS_s112
                                 + 1.01783 w_k1_FDX_s112
                                 + 1.21684 w_k1_IBM_s112
                                 + 0.937694 w_k1_INTC_s112
                                 + 1.23122 w_k1_KO_s112 + 1.16565 w_k1_MAT_s112
                                 + 1.01182 w_k1_MSFT_s112
                                 + 1.03719 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.945857 w_k1_ADM_s113
                                 + 1.26943 w_k1_AXP_s113
                                 + 1.47037 w_k1_DIS_s113
                                 + 1.28318 w_k1_FDX_s113 + 1.5017 w_k1_IBM_s113
                                 + 1.15634 w_k1_INTC_s113
                                 + 1.44637 w_k1_KO_s113 + 1.66303 w_k1_MAT_s113
                                 + 1.16153 w_k1_MSFT_s113
                                 + 1.17018 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.15631 w_k1_ADM_s113
                                 + 1.27125 w_k1_AXP_s113
                                 + 1.23765 w_k1_DIS_s113
                                 + 1.39079 w_k1_FDX_s113
                                 + 0.720699 w_k1_IBM_s113
                                 + 1.32625 w_k1_INTC_s113
                                 + 1.03613 w_k1_KO_s113
                                 + 0.872132 w_k1_MAT_s113
                                 + 1.21962 w_k1_MSFT_s113
                                 + 1.10443 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.916709 w_k1_ADM_s113
                                 + 0.984947 w_k1_AXP_s113
                                 + 0.746647 w_k1_DIS_s113
                                 + 1.02753 w_k1_FDX_s113
                                 + 0.794442 w_k1_IBM_s113
                                 + 0.966244 w_k1_INTC_s113
                                 + 0.789338 w_k1_KO_s113
                                 + 0.644359 w_k1_MAT_s113
                                 + 0.761789 w_k1_MSFT_s113
                                 + 0.863042 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.11102 w_k1_ADM_s113
                                 + 1.14966 w_k1_AXP_s113 + 1.0639 w_k1_DIS_s113
                                 + 1.21058 w_k1_FDX_s113
                                 + 1.06951 w_k1_IBM_s113
                                 + 1.08852 w_k1_INTC_s113 + 1.091 w_k1_KO_s113
                                 + 1.21674 w_k1_MAT_s113
                                 + 1.04412 w_k1_MSFT_s113
                                 + 0.974479 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.855661 w_k1_ADM_s113
                                 + 0.628769 w_k1_AXP_s113
                                 + 0.57942 w_k1_DIS_s113
                                 + 0.781256 w_k1_FDX_s113
                                 + 0.602225 w_k1_IBM_s113
                                 + 0.741397 w_k1_INTC_s113
                                 + 0.738524 w_k1_KO_s113
                                 + 0.503053 w_k1_MAT_s113
                                 + 0.669029 w_k1_MSFT_s113
                                 + 0.777339 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.05654 w_k1_ADM_s113
                                 + 0.659291 w_k1_AXP_s113
                                 + 0.893857 w_k1_DIS_s113
                                 + 0.971419 w_k1_FDX_s113
                                 + 0.662419 w_k1_IBM_s113
                                 + 0.895467 w_k1_INTC_s113
                                 + 0.90339 w_k1_KO_s113
                                 + 0.488152 w_k1_MAT_s113
                                 + 0.952578 w_k1_MSFT_s113
                                 + 0.853951 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.904396 w_k1_ADM_s113
                                 + 0.669558 w_k1_AXP_s113
                                 + 0.655948 w_k1_DIS_s113
                                 + 0.773411 w_k1_FDX_s113
                                 + 0.864835 w_k1_IBM_s113
                                 + 0.73672 w_k1_INTC_s113
                                 + 0.82924 w_k1_KO_s113 + 0.804 w_k1_MAT_s113
                                 + 0.709224 w_k1_MSFT_s113
                                 + 0.798989 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.14576 w_k1_ADM_s113
                                 + 0.975845 w_k1_AXP_s113
                                 + 0.839113 w_k1_DIS_s113
                                 + 0.910216 w_k1_FDX_s113
                                 + 1.41656 w_k1_IBM_s113
                                 + 1.14081 w_k1_INTC_s113
                                 + 1.10522 w_k1_KO_s113 + 1.10077 w_k1_MAT_s113
                                 + 0.894087 w_k1_MSFT_s113
                                 + 0.996145 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.31869 w_k1_ADM_s114
                                 + 1.10943 w_k1_AXP_s114
                                 + 1.35457 w_k1_DIS_s114 + 1.2876 w_k1_FDX_s114
                                 + 1.05336 w_k1_IBM_s114
                                 + 1.34032 w_k1_INTC_s114
                                 + 1.20712 w_k1_KO_s114
                                 + 0.946479 w_k1_MAT_s114
                                 + 1.50144 w_k1_MSFT_s114
                                 + 1.08949 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.06169 w_k1_ADM_s114
                                 + 0.78745 w_k1_AXP_s114
                                 + 0.976552 w_k1_DIS_s114
                                 + 0.881238 w_k1_FDX_s114
                                 + 1.19138 w_k1_IBM_s114
                                 + 0.679665 w_k1_INTC_s114
                                 + 0.99501 w_k1_KO_s114
                                 + 0.952848 w_k1_MAT_s114
                                 + 0.933368 w_k1_MSFT_s114
                                 + 0.940914 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.07469 w_k1_ADM_s114
                                 + 1.08509 w_k1_AXP_s114
                                 + 0.860102 w_k1_DIS_s114
                                 + 1.08427 w_k1_FDX_s114
                                 + 0.946071 w_k1_IBM_s114
                                 + 0.928995 w_k1_INTC_s114
                                 + 0.81334 w_k1_KO_s114
                                 + 0.857476 w_k1_MAT_s114
                                 + 0.96242 w_k1_MSFT_s114
                                 + 0.844577 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.664129 w_k1_ADM_s114
                                 + 0.734783 w_k1_AXP_s114
                                 + 0.53074 w_k1_DIS_s114
                                 + 0.699754 w_k1_FDX_s114
                                 + 0.955754 w_k1_IBM_s114
                                 + 0.769574 w_k1_INTC_s114
                                 + 0.72723 w_k1_KO_s114 + 1.02062 w_k1_MAT_s114
                                 + 0.73149 w_k1_MSFT_s114
                                 + 0.79146 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.879304 w_k1_ADM_s114
                                 + 0.820796 w_k1_AXP_s114
                                 + 0.792003 w_k1_DIS_s114
                                 + 0.911043 w_k1_FDX_s114
                                 + 0.664668 w_k1_IBM_s114
                                 + 0.881917 w_k1_INTC_s114
                                 + 0.750153 w_k1_KO_s114
                                 + 0.602056 w_k1_MAT_s114
                                 + 0.914243 w_k1_MSFT_s114
                                 + 0.918301 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.1007 w_k1_ADM_s114
                                 + 1.09556 w_k1_AXP_s114
                                 + 1.22085 w_k1_DIS_s114
                                 + 1.21504 w_k1_FDX_s114
                                 + 0.836833 w_k1_IBM_s114
                                 + 1.22341 w_k1_INTC_s114
                                 + 1.04386 w_k1_KO_s114
                                 + 0.680023 w_k1_MAT_s114
                                 + 1.18617 w_k1_MSFT_s114
                                 + 1.08258 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.19 w_k1_ADM_s114
                                 + 1.26436 w_k1_AXP_s114
                                 + 1.44057 w_k1_DIS_s114 + 1.1349 w_k1_FDX_s114
                                 + 1.48882 w_k1_IBM_s114
                                 + 1.0542 w_k1_INTC_s114 + 1.41934 w_k1_KO_s114
                                 + 1.42913 w_k1_MAT_s114
                                 + 1.16804 w_k1_MSFT_s114
                                 + 1.14264 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.859959 w_k1_ADM_s114
                                 + 1.03759 w_k1_AXP_s114
                                 + 1.48677 w_k1_DIS_s114
                                 + 1.18132 w_k1_FDX_s114
                                 + 1.27419 w_k1_IBM_s114
                                 + 1.15396 w_k1_INTC_s114
                                 + 1.29647 w_k1_KO_s114 + 1.4283 w_k1_MAT_s114
                                 + 1.17857 w_k1_MSFT_s114
                                 + 1.05282 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.905649 w_k1_ADM_s115
                                 + 0.633644 w_k1_AXP_s115
                                 + 0.583259 w_k1_DIS_s115
                                 + 0.689366 w_k1_FDX_s115
                                 + 0.717027 w_k1_IBM_s115
                                 + 0.848103 w_k1_INTC_s115
                                 + 0.641558 w_k1_KO_s115
                                 + 0.412363 w_k1_MAT_s115
                                 + 0.714039 w_k1_MSFT_s115
                                 + 0.814011 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.928392 w_k1_ADM_s115
                                 + 0.74016 w_k1_AXP_s115
                                 + 0.745678 w_k1_DIS_s115
                                 + 0.844865 w_k1_FDX_s115
                                 + 0.787619 w_k1_IBM_s115
                                 + 0.802313 w_k1_INTC_s115
                                 + 0.803962 w_k1_KO_s115
                                 + 0.69426 w_k1_MAT_s115
                                 + 0.858377 w_k1_MSFT_s115
                                 + 0.838837 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1593 w_k1_ADM_s115
                                 + 1.05569 w_k1_AXP_s115
                                 + 1.14989 w_k1_DIS_s115
                                 + 1.16701 w_k1_FDX_s115
                                 + 0.887519 w_k1_IBM_s115
                                 + 1.33683 w_k1_INTC_s115
                                 + 1.09711 w_k1_KO_s115
                                 + 0.652747 w_k1_MAT_s115
                                 + 1.16842 w_k1_MSFT_s115
                                 + 1.04813 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.97082 w_k1_ADM_s115
                                 + 1.43103 w_k1_AXP_s115
                                 + 1.03944 w_k1_DIS_s115
                                 + 1.19402 w_k1_FDX_s115
                                 + 0.860647 w_k1_IBM_s115
                                 + 1.14753 w_k1_INTC_s115
                                 + 0.908753 w_k1_KO_s115 + 1.1611 w_k1_MAT_s115
                                 + 1.16452 w_k1_MSFT_s115
                                 + 1.11978 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.846695 w_k1_ADM_s115
                                 + 0.883944 w_k1_AXP_s115
                                 + 0.726141 w_k1_DIS_s115
                                 + 0.744259 w_k1_FDX_s115
                                 + 1.15582 w_k1_IBM_s115
                                 + 0.957142 w_k1_INTC_s115
                                 + 0.987306 w_k1_KO_s115
                                 + 1.22143 w_k1_MAT_s115
                                 + 0.883219 w_k1_MSFT_s115
                                 + 0.95697 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.910137 w_k1_ADM_s115
                                 + 1.2041 w_k1_AXP_s115 + 1.27921 w_k1_DIS_s115
                                 + 1.05774 w_k1_FDX_s115
                                 + 1.52248 w_k1_IBM_s115
                                 + 1.1799 w_k1_INTC_s115 + 1.56014 w_k1_KO_s115
                                 + 1.23688 w_k1_MAT_s115
                                 + 1.11144 w_k1_MSFT_s115 + 1.2153 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.979865 w_k1_ADM_s115
                                 + 0.840914 w_k1_AXP_s115
                                 + 0.920898 w_k1_DIS_s115
                                 + 0.924612 w_k1_FDX_s115
                                 + 0.933212 w_k1_IBM_s115
                                 + 1.00443 w_k1_INTC_s115
                                 + 0.999834 w_k1_KO_s115
                                 + 0.702911 w_k1_MAT_s115
                                 + 0.963493 w_k1_MSFT_s115
                                 + 0.977204 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.02469 w_k1_ADM_s115
                                 + 1.13599 w_k1_AXP_s115
                                 + 0.83599 w_k1_DIS_s115
                                 + 0.943957 w_k1_FDX_s115
                                 + 1.27313 w_k1_IBM_s115
                                 + 0.955002 w_k1_INTC_s115
                                 + 1.04652 w_k1_KO_s115 + 1.05669 w_k1_MAT_s115
                                 + 0.902992 w_k1_MSFT_s115
                                 + 1.08649 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.27232 w_k1_ADM_s116
                                 + 1.32618 w_k1_AXP_s116
                                 + 1.57281 w_k1_DIS_s116
                                 + 1.26048 w_k1_FDX_s116
                                 + 1.12945 w_k1_IBM_s116
                                 + 1.24998 w_k1_INTC_s116
                                 + 1.37087 w_k1_KO_s116 + 1.38457 w_k1_MAT_s116
                                 + 1.48201 w_k1_MSFT_s116
                                 + 1.11919 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.2032 w_k1_ADM_s116
                                 + 1.51822 w_k1_AXP_s116
                                 + 1.36565 w_k1_DIS_s116
                                 + 1.29163 w_k1_FDX_s116
                                 + 1.20585 w_k1_IBM_s116
                                 + 1.30841 w_k1_INTC_s116 + 1.3296 w_k1_KO_s116
                                 + 1.70623 w_k1_MAT_s116
                                 + 1.32438 w_k1_MSFT_s116
                                 + 0.989256 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.927626 w_k1_ADM_s116
                                 + 0.793137 w_k1_AXP_s116
                                 + 0.69712 w_k1_DIS_s116
                                 + 0.798689 w_k1_FDX_s116
                                 + 1.13765 w_k1_IBM_s116
                                 + 0.89881 w_k1_INTC_s116
                                 + 0.965879 w_k1_KO_s116
                                 + 1.26518 w_k1_MAT_s116
                                 + 0.973376 w_k1_MSFT_s116
                                 + 0.86352 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.17444 w_k1_ADM_s116
                                 + 0.961823 w_k1_AXP_s116
                                 + 1.02591 w_k1_DIS_s116
                                 + 1.03157 w_k1_FDX_s116
                                 + 1.19381 w_k1_IBM_s116
                                 + 1.11855 w_k1_INTC_s116
                                 + 1.20943 w_k1_KO_s116 + 1.02068 w_k1_MAT_s116
                                 + 1.12596 w_k1_MSFT_s116
                                 + 1.00337 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.963657 w_k1_ADM_s116
                                 + 0.65941 w_k1_AXP_s116
                                 + 0.691256 w_k1_DIS_s116
                                 + 0.775797 w_k1_FDX_s116
                                 + 0.766994 w_k1_IBM_s116
                                 + 0.886624 w_k1_INTC_s116
                                 + 0.830172 w_k1_KO_s116
                                 + 0.697048 w_k1_MAT_s116
                                 + 0.881264 w_k1_MSFT_s116
                                 + 0.888155 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.13414 w_k1_ADM_s116
                                 + 1.03337 w_k1_AXP_s116
                                 + 0.916946 w_k1_DIS_s116
                                 + 0.977739 w_k1_FDX_s116
                                 + 0.924169 w_k1_IBM_s116
                                 + 0.952061 w_k1_INTC_s116
                                 + 0.873318 w_k1_KO_s116
                                 + 1.18142 w_k1_MAT_s116
                                 + 1.12626 w_k1_MSFT_s116
                                 + 0.934051 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.21184 w_k1_ADM_s116
                                 + 1.03818 w_k1_AXP_s116
                                 + 1.37164 w_k1_DIS_s116
                                 + 1.07057 w_k1_FDX_s116 + 1.6795 w_k1_IBM_s116
                                 + 1.17584 w_k1_INTC_s116 + 1.432 w_k1_KO_s116
                                 + 2.03595 w_k1_MAT_s116
                                 + 1.25874 w_k1_MSFT_s116 + 1.0802 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.29803 w_k1_ADM_s116
                                 + 1.09652 w_k1_AXP_s116 + 1.0851 w_k1_DIS_s116
                                 + 1.16814 w_k1_FDX_s116
                                 + 0.831239 w_k1_IBM_s116
                                 + 1.03766 w_k1_INTC_s116
                                 + 1.00356 w_k1_KO_s116 + 1.26484 w_k1_MAT_s116
                                 + 1.16911 w_k1_MSFT_s116
                                 + 1.03443 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.07875 w_k1_ADM_s117
                                 + 1.36699 w_k1_AXP_s117
                                 + 1.13008 w_k1_DIS_s117
                                 + 1.00753 w_k1_FDX_s117
                                 + 1.56735 w_k1_IBM_s117
                                 + 1.15838 w_k1_INTC_s117
                                 + 1.11765 w_k1_KO_s117 + 1.27217 w_k1_MAT_s117
                                 + 0.854952 w_k1_MSFT_s117
                                 + 0.979774 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.85067 w_k1_ADM_s117
                                 + 0.856788 w_k1_AXP_s117
                                 + 0.810145 w_k1_DIS_s117
                                 + 0.8942 w_k1_FDX_s117
                                 + 0.648918 w_k1_IBM_s117
                                 + 0.918229 w_k1_INTC_s117
                                 + 0.753438 w_k1_KO_s117
                                 + 0.522863 w_k1_MAT_s117
                                 + 0.878095 w_k1_MSFT_s117
                                 + 0.763969 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.649723 w_k1_ADM_s117
                                 + 0.629384 w_k1_AXP_s117
                                 + 0.644229 w_k1_DIS_s117
                                 + 0.649228 w_k1_FDX_s117
                                 + 0.974288 w_k1_IBM_s117
                                 + 0.879025 w_k1_INTC_s117
                                 + 0.73549 w_k1_KO_s117
                                 + 0.871676 w_k1_MAT_s117
                                 + 0.697898 w_k1_MSFT_s117
                                 + 0.854245 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.774869 w_k1_ADM_s117
                                 + 0.840996 w_k1_AXP_s117
                                 + 0.732609 w_k1_DIS_s117
                                 + 0.796831 w_k1_FDX_s117
                                 + 0.886181 w_k1_IBM_s117
                                 + 1.14057 w_k1_INTC_s117
                                 + 0.888655 w_k1_KO_s117
                                 + 0.568657 w_k1_MAT_s117
                                 + 0.76544 w_k1_MSFT_s117
                                 + 0.795662 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.825983 w_k1_ADM_s117
                                 + 1.00695 w_k1_AXP_s117
                                 + 0.949685 w_k1_DIS_s117
                                 + 0.886895 w_k1_FDX_s117 + 1.074 w_k1_IBM_s117
                                 + 0.949984 w_k1_INTC_s117
                                 + 0.919116 w_k1_KO_s117 + 1.3794 w_k1_MAT_s117
                                 + 0.916286 w_k1_MSFT_s117
                                 + 0.896053 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.936934 w_k1_ADM_s117
                                 + 1.38407 w_k1_AXP_s117
                                 + 1.50765 w_k1_DIS_s117
                                 + 1.16128 w_k1_FDX_s117
                                 + 1.23824 w_k1_IBM_s117
                                 + 1.09887 w_k1_INTC_s117
                                 + 1.23856 w_k1_KO_s117 + 1.31605 w_k1_MAT_s117
                                 + 1.17129 w_k1_MSFT_s117
                                 + 1.16403 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.30706 w_k1_ADM_s117
                                 + 1.535 w_k1_AXP_s117 + 1.59714 w_k1_DIS_s117
                                 + 1.51049 w_k1_FDX_s117
                                 + 1.03086 w_k1_IBM_s117
                                 + 1.13555 w_k1_INTC_s117
                                 + 1.20286 w_k1_KO_s117 + 1.41466 w_k1_MAT_s117
                                 + 1.2108 w_k1_MSFT_s117 + 1.14115 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.974697 w_k1_ADM_s117
                                 + 1.15266 w_k1_AXP_s117
                                 + 1.08126 w_k1_DIS_s117
                                 + 1.09684 w_k1_FDX_s117
                                 + 0.857474 w_k1_IBM_s117
                                 + 1.19042 w_k1_INTC_s117
                                 + 0.90198 w_k1_KO_s117
                                 + 0.845908 w_k1_MAT_s117
                                 + 0.938691 w_k1_MSFT_s117
                                 + 0.91606 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0229444674936012 w_k1_ADM_s000
                                 + 0.013315689582792 w_k1_ADM_s001
                                 + 0.0140909594352077 w_k1_ADM_s002
                                 + 0.026098570599236 w_k1_ADM_s003
                                 + 0.0125171649327147 w_k1_ADM_s004
                                 + 0.0111480288956597 w_k1_ADM_s005
                                 + 0.0542937846372634 w_k1_ADM_s006
                                 + 0.0192388593010162 w_k1_ADM_s007
                                 + 0.0111190218128626 w_k1_ADM_s010
                                 + 0.00721597444454032 w_k1_ADM_s011
                                 + 0.0164331954713049 w_k1_ADM_s012
                                 + 0.00806124614192401 w_k1_ADM_s013
                                 + 0.0145253823679166 w_k1_ADM_s014
                                 + 0.020759350712457 w_k1_ADM_s015
                                 + 0.017211535629513 w_k1_ADM_s016
                                 + 0.0197650844393628 w_k1_ADM_s017
                                 + 0.0207861114348885 w_k1_ADM_s100
                                 + 0.0377851957458004 w_k1_ADM_s101
                                 + 0.0606307708429044 w_k1_ADM_s102
                                 + 0.0370853536785841 w_k1_ADM_s103
                                 + 0.0663474626299402 w_k1_ADM_s104
                                 + 0.030242104252836 w_k1_ADM_s105
                                 + 0.0840365367996681 w_k1_ADM_s106
                                 + 0.0351972951952947 w_k1_ADM_s107
                                 + 0.0506313157460875 w_k1_ADM_s110
                                 + 0.0877699507734273 w_k1_ADM_s111
                                 + 0.0587394775898982 w_k1_ADM_s112
                                 + 0.0303161690669923 w_k1_ADM_s113
                                 + 0.0337852846690696 w_k1_ADM_s114
                                 + 0.0339132004652852 w_k1_ADM_s115
                                 + 0.0288997361895191 w_k1_ADM_s116
                                 + 0.0318418029676586 w_k1_ADM_s117
                                 + 0.0217761804945225 w_k1_AXP_s000
                                 + 0.0138431635399156 w_k1_AXP_s001
                                 + 0.013260934014167 w_k1_AXP_s002
                                 + 0.0298616986993042 w_k1_AXP_s003
                                 + 0.011501760773961 w_k1_AXP_s004
                                 + 0.011325492808678 w_k1_AXP_s005
                                 + 0.0497938159228035 w_k1_AXP_s006
                                 + 0.0209111258420143 w_k1_AXP_s007
                                 + 0.00935855815640613 w_k1_AXP_s010
                                 + 0.00859086805884105 w_k1_AXP_s011
                                 + 0.0193293872265811 w_k1_AXP_s012
                                 + 0.00765965175306543 w_k1_AXP_s013
                                 + 0.0151562051542281 w_k1_AXP_s014
                                 + 0.0212309378276301 w_k1_AXP_s015
                                 + 0.0142473850376876 w_k1_AXP_s016
                                 + 0.0204831023199289 w_k1_AXP_s017
                                 + 0.0216938190296743 w_k1_AXP_s100
                                 + 0.037825695856817 w_k1_AXP_s101
                                 + 0.0522672327825787 w_k1_AXP_s102
                                 + 0.0335858128305472 w_k1_AXP_s103
                                 + 0.0673046503379285 w_k1_AXP_s104
                                 + 0.0312439615991417 w_k1_AXP_s105
                                 + 0.0863453931090021 w_k1_AXP_s106
                                 + 0.0390430594807725 w_k1_AXP_s107
                                 + 0.0567334530978201 w_k1_AXP_s110
                                 + 0.0815291043295133 w_k1_AXP_s111
                                 + 0.0593658005344406 w_k1_AXP_s112
                                 + 0.0289815936008391 w_k1_AXP_s113
                                 + 0.0325659533181999 w_k1_AXP_s114
                                 + 0.0337852483725568 w_k1_AXP_s115
                                 + 0.0257805468319573 w_k1_AXP_s116
                                 + 0.0376009506429498 w_k1_AXP_s117
                                 + 0.0209428383823712 w_k1_DIS_s000
                                 + 0.0145035087800191 w_k1_DIS_s001
                                 + 0.0137200980132049 w_k1_DIS_s002
                                 + 0.0259902504802995 w_k1_DIS_s003
                                 + 0.0140449190967952 w_k1_DIS_s004
                                 + 0.0107744447393169 w_k1_DIS_s005
                                 + 0.0501957188797788 w_k1_DIS_s006
                                 + 0.0191200589514833 w_k1_DIS_s007
                                 + 0.0104216431633608 w_k1_DIS_s010
                                 + 0.00834641716921744 w_k1_DIS_s011
                                 + 0.0194990790738397 w_k1_DIS_s012
                                 + 0.0082482424397657 w_k1_DIS_s013
                                 + 0.0140881924185051 w_k1_DIS_s014
                                 + 0.0196082986281585 w_k1_DIS_s015
                                 + 0.0161806305414153 w_k1_DIS_s016
                                 + 0.0182973153679215 w_k1_DIS_s017
                                 + 0.0215887712980392 w_k1_DIS_s100
                                 + 0.0386004923157595 w_k1_DIS_s101
                                 + 0.0563366017406479 w_k1_DIS_s102
                                 + 0.037002791423822 w_k1_DIS_s103
                                 + 0.0718328993125194 w_k1_DIS_s104
                                 + 0.0341271135217883 w_k1_DIS_s105
                                 + 0.0844530292458108 w_k1_DIS_s106
                                 + 0.0412982259908572 w_k1_DIS_s107
                                 + 0.05423291891932 w_k1_DIS_s110
                                 + 0.0926356486766539 w_k1_DIS_s111
                                 + 0.066069752034783 w_k1_DIS_s112
                                 + 0.02800561979479 w_k1_DIS_s113
                                 + 0.0342588705788913 w_k1_DIS_s114
                                 + 0.0331950035084954 w_k1_DIS_s115
                                 + 0.0258704006497118 w_k1_DIS_s116
                                 + 0.0355395706951396 w_k1_DIS_s117
                                 + 0.0205441444477653 w_k1_FDX_s000
                                 + 0.0145677332356958 w_k1_FDX_s001
                                 + 0.0138602997136914 w_k1_FDX_s002
                                 + 0.0272267544987532 w_k1_FDX_s003
                                 + 0.0134725689519638 w_k1_FDX_s004
                                 + 0.0110141583380937 w_k1_FDX_s005
                                 + 0.0525092339205679 w_k1_FDX_s006
                                 + 0.0189525586633776 w_k1_FDX_s007
                                 + 0.010455493873298 w_k1_FDX_s010
                                 + 0.00771001047038116 w_k1_FDX_s011
                                 + 0.0188697833889297 w_k1_FDX_s012
                                 + 0.00807120362255371 w_k1_FDX_s013
                                 + 0.0141857044500068 w_k1_FDX_s014
                                 + 0.0202163112767031 w_k1_FDX_s015
                                 + 0.0158610928710738 w_k1_FDX_s016
                                 + 0.0184490455988334 w_k1_FDX_s017
                                 + 0.0220265258119824 w_k1_FDX_s100
                                 + 0.0384255502268207 w_k1_FDX_s101
                                 + 0.0541541754065941 w_k1_FDX_s102
                                 + 0.0350038135913711 w_k1_FDX_s103
                                 + 0.0705720838737774 w_k1_FDX_s104
                                 + 0.0318135153443804 w_k1_FDX_s105
                                 + 0.0876675832236282 w_k1_FDX_s106
                                 + 0.0397946989837245 w_k1_FDX_s107
                                 + 0.0553794159252941 w_k1_FDX_s110
                                 + 0.0881662950850113 w_k1_FDX_s111
                                 + 0.0623985805195211 w_k1_FDX_s112
                                 + 0.0307764847782905 w_k1_FDX_s113
                                 + 0.034158245873889 w_k1_FDX_s114
                                 + 0.0336201127186509 w_k1_FDX_s115
                                 + 0.0258657055008677 w_k1_FDX_s116
                                 + 0.0342032618266957 w_k1_FDX_s117
                                 + 0.0198431525459429 w_k1_IBM_s000
                                 + 0.0135683848866518 w_k1_IBM_s001
                                 + 0.0134459083581827 w_k1_IBM_s002
                                 + 0.0264926141910907 w_k1_IBM_s003
                                 + 0.0135139181045895 w_k1_IBM_s004
                                 + 0.0111639277694205 w_k1_IBM_s005
                                 + 0.0507754482066461 w_k1_IBM_s006
                                 + 0.0206003273020637 w_k1_IBM_s007
                                 + 0.00977273711685671 w_k1_IBM_s010
                                 + 0.00867282872803426 w_k1_IBM_s011
                                 + 0.0188929299022882 w_k1_IBM_s012
                                 + 0.0083312452271701 w_k1_IBM_s013
                                 + 0.0133043955149401 w_k1_IBM_s014
                                 + 0.0204371788774667 w_k1_IBM_s015
                                 + 0.0149977327026222 w_k1_IBM_s016
                                 + 0.0184618706842096 w_k1_IBM_s017
                                 + 0.0223918633803572 w_k1_IBM_s100
                                 + 0.0393510006974824 w_k1_IBM_s101
                                 + 0.0538078705199147 w_k1_IBM_s102
                                 + 0.0353690011696523 w_k1_IBM_s103
                                 + 0.0712372226624362 w_k1_IBM_s104
                                 + 0.0325268801824074 w_k1_IBM_s105
                                 + 0.0919012824379063 w_k1_IBM_s106
                                 + 0.0400850861775996 w_k1_IBM_s107
                                 + 0.0565280090614214 w_k1_IBM_s110
                                 + 0.0834023492188282 w_k1_IBM_s111
                                 + 0.0626944306783866 w_k1_IBM_s112
                                 + 0.0312409505369532 w_k1_IBM_s113
                                 + 0.0343883288113548 w_k1_IBM_s114
                                 + 0.0344896271489715 w_k1_IBM_s115
                                 + 0.0271051054304878 w_k1_IBM_s116
                                 + 0.0350617835688897 w_k1_IBM_s117
                                 + 0.0213298876159452 w_k1_INTC_s000
                                 + 0.0154086958706663 w_k1_INTC_s001
                                 + 0.0126699495487098 w_k1_INTC_s002
                                 + 0.0277785960040651 w_k1_INTC_s003
                                 + 0.0121955696476819 w_k1_INTC_s004
                                 + 0.0119327018210653 w_k1_INTC_s005
                                 + 0.0538577432580742 w_k1_INTC_s006
                                 + 0.0207518962493778 w_k1_INTC_s007
                                 + 0.0108864892904609 w_k1_INTC_s010
                                 + 0.00802224184891943 w_k1_INTC_s011
                                 + 0.0157351907704381 w_k1_INTC_s012
                                 + 0.00791280421480047 w_k1_INTC_s013
                                 + 0.0138394409556445 w_k1_INTC_s014
                                 + 0.0199463440067434 w_k1_INTC_s015
                                 + 0.0148164329448637 w_k1_INTC_s016
                                 + 0.0182805835073766 w_k1_INTC_s017
                                 + 0.0240476807424632 w_k1_INTC_s100
                                 + 0.0340702539543876 w_k1_INTC_s101
                                 + 0.0523582545199916 w_k1_INTC_s102
                                 + 0.031921570659777 w_k1_INTC_s103
                                 + 0.0672255095288348 w_k1_INTC_s104
                                 + 0.0328547599863754 w_k1_INTC_s105
                                 + 0.0935504023698845 w_k1_INTC_s106
                                 + 0.0353365339354561 w_k1_INTC_s107
                                 + 0.0545424168562934 w_k1_INTC_s110
                                 + 0.0827657850934438 w_k1_INTC_s111
                                 + 0.0624525406064476 w_k1_INTC_s112
                                 + 0.030361310195027 w_k1_INTC_s113
                                 + 0.0321874617242846 w_k1_INTC_s114
                                 + 0.0368197822571879 w_k1_INTC_s115
                                 + 0.0267489951611553 w_k1_INTC_s116
                                 + 0.0374024588795166 w_k1_INTC_s117
                                 + 0.0209862562153654 w_k1_KO_s000
                                 + 0.0155265214740328 w_k1_KO_s001
                                 + 0.0131803787118409 w_k1_KO_s002
                                 + 0.0274442635698318 w_k1_KO_s003
                                 + 0.0143643893780533 w_k1_KO_s004
                                 + 0.011520518387722 w_k1_KO_s005
                                 + 0.0543551077130928 w_k1_KO_s006
                                 + 0.0196755159996776 w_k1_KO_s007
                                 + 0.00877399582944974 w_k1_KO_s010
                                 + 0.00756069544933171 w_k1_KO_s011
                                 + 0.0154702481872274 w_k1_KO_s012
                                 + 0.00761587659361186 w_k1_KO_s013
                                 + 0.012750153761863 w_k1_KO_s014
                                 + 0.020879698723293 w_k1_KO_s015
                                 + 0.0163310467565685 w_k1_KO_s016
                                 + 0.0182482051728142 w_k1_KO_s017
                                 + 0.0239375780834509 w_k1_KO_s100
                                 + 0.0364542247288108 w_k1_KO_s101
                                 + 0.0582773252188947 w_k1_KO_s102
                                 + 0.0346891256600583 w_k1_KO_s103
                                 + 0.0644275465890473 w_k1_KO_s104
                                 + 0.0317299952212778 w_k1_KO_s105
                                 + 0.0819417872025633 w_k1_KO_s106
                                 + 0.0373246094830728 w_k1_KO_s107
                                 + 0.0518092213735406 w_k1_KO_s110
                                 + 0.0857656841284139 w_k1_KO_s111
                                 + 0.0646745073785075 w_k1_KO_s112
                                 + 0.0304274244600182 w_k1_KO_s113
                                 + 0.0331540608566103 w_k1_KO_s114
                                 + 0.0358860835863818 w_k1_KO_s115
                                 + 0.0273893743406733 w_k1_KO_s116
                                 + 0.0330209874632434 w_k1_KO_s117
                                 + 0.0237383307988161 w_k1_MAT_s000
                                 + 0.0130794772756528 w_k1_MAT_s001
                                 + 0.0144926494852113 w_k1_MAT_s002
                                 + 0.029508631349787 w_k1_MAT_s003
                                 + 0.0139293273780815 w_k1_MAT_s004
                                 + 0.00998466884051228 w_k1_MAT_s005
                                 + 0.0457591146836726 w_k1_MAT_s006
                                 + 0.0205373111726729 w_k1_MAT_s007
                                 + 0.00864943118971432 w_k1_MAT_s010
                                 + 0.00958287104023295 w_k1_MAT_s011
                                 + 0.0212310546581287 w_k1_MAT_s012
                                 + 0.00975168071949592 w_k1_MAT_s013
                                 + 0.0121269881465631 w_k1_MAT_s014
                                 + 0.0214219181510291 w_k1_MAT_s015
                                 + 0.0174381729543537 w_k1_MAT_s016
                                 + 0.0202077079165491 w_k1_MAT_s017
                                 + 0.0198532660611604 w_k1_MAT_s100
                                 + 0.0409029336103394 w_k1_MAT_s101
                                 + 0.0449232095952581 w_k1_MAT_s102
                                 + 0.0393861847739519 w_k1_MAT_s103
                                 + 0.0625611976287769 w_k1_MAT_s104
                                 + 0.0259736498789008 w_k1_MAT_s105
                                 + 0.0829853566789574 w_k1_MAT_s106
                                 + 0.0421236097402167 w_k1_MAT_s107
                                 + 0.0577363185381558 w_k1_MAT_s110
                                 + 0.086590541888775 w_k1_MAT_s111
                                 + 0.060198554532857 w_k1_MAT_s112
                                 + 0.0294703546335331 w_k1_MAT_s113
                                 + 0.0317541823239451 w_k1_MAT_s114
                                 + 0.0285724459499623 w_k1_MAT_s115
                                 + 0.0311494554265939 w_k1_MAT_s116
                                 + 0.0328203606867851 w_k1_MAT_s117
                                 + 0.0209699994353834 w_k1_MSFT_s000
                                 + 0.0159196435265459 w_k1_MSFT_s001
                                 + 0.0133765579828177 w_k1_MSFT_s002
                                 + 0.0288823262140386 w_k1_MSFT_s003
                                 + 0.0143457809747148 w_k1_MSFT_s004
                                 + 0.0105622244215992 w_k1_MSFT_s005
                                 + 0.0477402004333946 w_k1_MSFT_s006
                                 + 0.0190660468840194 w_k1_MSFT_s007
                                 + 0.0101650361301713 w_k1_MSFT_s010
                                 + 0.00780802164837032 w_k1_MSFT_s011
                                 + 0.017880151201199 w_k1_MSFT_s012
                                 + 0.00822103407808874 w_k1_MSFT_s013
                                 + 0.0131437612585067 w_k1_MSFT_s014
                                 + 0.0202400439823409 w_k1_MSFT_s015
                                 + 0.0159972475308428 w_k1_MSFT_s016
                                 + 0.0193451553054683 w_k1_MSFT_s017
                                 + 0.0231889619655475 w_k1_MSFT_s100
                                 + 0.0403450467358972 w_k1_MSFT_s101
                                 + 0.0520677124210887 w_k1_MSFT_s102
                                 + 0.035569993013534 w_k1_MSFT_s103
                                 + 0.06815934780805 w_k1_MSFT_s104
                                 + 0.0320186442335657 w_k1_MSFT_s105
                                 + 0.0852945745421396 w_k1_MSFT_s106
                                 + 0.0366137632794218 w_k1_MSFT_s107
                                 + 0.0552901861871946 w_k1_MSFT_s110
                                 + 0.0838365902480008 w_k1_MSFT_s111
                                 + 0.0594460134728528 w_k1_MSFT_s112
                                 + 0.0274882280741894 w_k1_MSFT_s113
                                 + 0.0344501811702485 w_k1_MSFT_s114
                                 + 0.0343108204141993 w_k1_MSFT_s115
                                 + 0.0289720482432334 w_k1_MSFT_s116
                                 + 0.0315087867601496 w_k1_MSFT_s117
                                 + 0.0196700976804868 w_k1_PG_s000
                                 + 0.0147038715621131 w_k1_PG_s001
                                 + 0.0138183794976265 w_k1_PG_s002
                                 + 0.0257265583975884 w_k1_PG_s003
                                 + 0.0138853691652533 w_k1_PG_s004
                                 + 0.0113180989048766 w_k1_PG_s005
                                 + 0.0522477514372303 w_k1_PG_s006
                                 + 0.0185594039222872 w_k1_PG_s007
                                 + 0.0107127063939291 w_k1_PG_s010
                                 + 0.00793664770021407 w_k1_PG_s011
                                 + 0.0178789459038131 w_k1_PG_s012
                                 + 0.0078622104072114 w_k1_PG_s013
                                 + 0.0139220253836245 w_k1_PG_s014
                                 + 0.0185886270277438 w_k1_PG_s015
                                 + 0.0159856549619496 w_k1_PG_s016
                                 + 0.0175534867885427 w_k1_PG_s017
                                 + 0.0211545830346619 w_k1_PG_s100
                                 + 0.0363455042894178 w_k1_PG_s101
                                 + 0.0516107342130218 w_k1_PG_s102
                                 + 0.0339944540812552 w_k1_PG_s103
                                 + 0.0730359801448184 w_k1_PG_s104
                                 + 0.0304728213478471 w_k1_PG_s105
                                 + 0.0831127259289084 w_k1_PG_s106
                                 + 0.0359091952865458 w_k1_PG_s107
                                 + 0.0508446346705622 w_k1_PG_s110
                                 + 0.0844372942770865 w_k1_PG_s111
                                 + 0.0610476615288972 w_k1_PG_s112
                                 + 0.028217615790487 w_k1_PG_s113
                                 + 0.0325449159943329 w_k1_PG_s114
                                 + 0.0346284176061896 w_k1_PG_s115
                                 + 0.0248879576648225 w_k1_PG_s116
                                 + 0.0317871166916401 w_k1_PG_s117 >= 1.09
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
