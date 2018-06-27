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
 w_k1_ADM_s0_return_constraint:  1.10681 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.923149 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.16596 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.840232 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.944837 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.21097 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.663973 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.771495 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.987511 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.951097 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.771771 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.17823 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.979514 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.06393 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.16292 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.2933 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.903425 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.10671 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.907636 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.780234 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.874482 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.23924 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.727244 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.08957 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.17119 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.955818 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.13405 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.872724 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.831027 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.02344 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.15906 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.31983 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.2059 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.849506 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.16551 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.804649 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.03873 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.07179 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20624 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.896924 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.19369 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.775757 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.861313 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.29034 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.600499 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.682713 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.880747 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.998133 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.857757 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.48148 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.10957 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.12377 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.35416 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.13892 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.09647 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.29401 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.800929 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.664416 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.869197 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.23745 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.630567 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.51404 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.29103 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.833886 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.10485 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.801821 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.914306 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.99249 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.03157 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.34508 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.965645 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.944489 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.40707 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.748246 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.744201 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.18255 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28519 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.8259 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.16613 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.719134 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.828721 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.43616 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.513346 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.732726 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.00541 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.01578 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.810471 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.62483 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.28215 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.32326 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.37104 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.05803 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.887897 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.3342 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.754395 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.578687 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.899456 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.49119 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.606447 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.64733 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.45522 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.829239 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.2184 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.750546 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.915342 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.9006 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.875414 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.57186 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.06997 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.847691 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.51174 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.746852 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.05296 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.08355 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14267 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.900161 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.14665 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.784829 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.88714 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.22315 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.70257 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.868753 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.13901 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.974938 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.824686 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.32099 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.996037 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.2842 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.10779 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.23987 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.02175 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.23187 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.956677 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.694056 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.883303 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.2108 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.707182 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.26461 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.21889 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.89769 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.14729 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.772234 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.889259 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.10202 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.10817 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.45402 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.14149 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.860665 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.16568 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.798762 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.01342 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.05642 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17853 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.884347 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.07525 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.964268 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.938567 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.25316 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.551081 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.648541 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.588459 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.897926 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.13004 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.36591 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.96577 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.884249 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.91578 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.694034 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.00398 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.936164 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.663813 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.850522 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.31579 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.18595 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.667334 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.77387 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.23885 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.867979 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.98215 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.46049 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.1025 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.531546 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.618615 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.954277 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.726894 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.47276 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.54022 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.871579 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.683219 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.05907 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.14454 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.896909 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.870569 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.769534 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.935283 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.19862 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.838666 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.08676 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.2584 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.26561 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.16376 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.82704 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.20721 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.43828 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.34018 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.05791 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.18444 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.33531 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.0588 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.735373 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.0266 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.27648 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.722182 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.05072 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.31434 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.03416 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.11935 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.971775 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.944031 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.08937 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.94699 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.397 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.952936 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.93463 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.1443 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.780473 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.038 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.18352 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18664 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.870523 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.19762 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.817125 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.905635 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.33683 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.527076 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.685548 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.72333 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.936884 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.823536 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.23302 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.33432 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.994073 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.41448 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.883005 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.08999 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.1328 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.77675 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.823874 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.16781 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.32245 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.702181 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.3966 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.20575 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.916098 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.10357 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.04221 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.975228 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.773131 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.744458 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.22987 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.95325 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.03222 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.37622 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.813831 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.836323 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.993218 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15384 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.815487 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.13966 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.939535 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.914797 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.34409 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.501118 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.782105 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.659489 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.1847 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.2836 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.56901 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 2.03702 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.878359 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.88157 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.807966 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.10225 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.13142 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.699685 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.844631 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.31284 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.23536 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.586145 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 2.25451 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.29761 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.879872 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.08606 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.17936 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.26281 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.582719 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.609587 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.923008 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.914683 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.80475 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.57091 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.90766 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.60831 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.24833 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12231 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.878455 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.07329 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.808622 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.927369 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.24906 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.683284 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.871243 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.136 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.05689 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.996947 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.38701 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.00964 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.2441 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.16726 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.07225 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.01152 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.26304 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.987175 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.751755 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.01427 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.31373 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.748031 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.23666 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.1995 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.929898 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.13668 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.793589 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.871577 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.08907 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.01508 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.32577 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.14966 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.861837 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.12063 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.812779 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.03711 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.05497 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12849 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.918781 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.07705 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.889969 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.922076 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.1982 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.712885 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.828724 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.94903 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.998503 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.935122 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.21951 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.16417 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08275 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.03272 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.925255 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.04573 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.06219 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.812147 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.755095 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.973375 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.03699 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.80309 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.29564 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.13794 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.936674 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.07986 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.9916 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.979381 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.912282 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.839346 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.2104 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.980409 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.963909 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.06616 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.83976 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.939681 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.00808 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.210000053217333 z_k1_s0000
                                 - 0.332000274179852 z_k1_s0001
                                 - 0.136999945892228 z_k1_s0002
                                 - 0.335000002265716 z_k1_s0003
                                 - 0.398000799071198 z_k1_s0004
                                 - 0.143999962126144 z_k1_s0005
                                 - 0.427000339568661 z_k1_s0006
                                 - 0.186000059529642 z_k1_s0007
                                 - 0.757001531344822 z_k1_s0010
                                 - 0.532002875719734 z_k1_s0011
                                 - 0.397001360337192 z_k1_s0012
                                 - 0.415000788385702 z_k1_s0013
                                 - 0.641001777243334 z_k1_s0014
                                 - 0.286000888358488 z_k1_s0015
                                 - 0.416003121618246 z_k1_s0016
                                 - 0.426003233868805 z_k1_s0017
                                 - 0.482000351532775 z_k1_s0020
                                 - 0.135000026425162 z_k1_s0021
                                 - 0.337999166012719 z_k1_s0022
                                 - 0.344999585308796 z_k1_s0023
                                 - 0.255001014358543 z_k1_s0024
                                 - 0.368999782895472 z_k1_s0025
                                 - 0.125999952330159 z_k1_s0026
                                 - 0.11999990293349 z_k1_s0027
                                 - 0.395000312520692 z_k1_s0030
                                 - 0.890002730735595 z_k1_s0031
                                 - 0.669999432370239 z_k1_s0032
                                 - 1.0290001653507 z_k1_s0033
                                 - 0.755002222831747 z_k1_s0034
                                 - 0.700001398240408 z_k1_s0035
                                 - 0.877000211156139 z_k1_s0036
                                 - 0.243000358326132 z_k1_s0037
                                 - 1.29399731125346 z_k1_s0040
                                 - 1.11900179887911 z_k1_s0041
                                 - 1.66299676684245 z_k1_s0042
                                 - 0.562999083609325 z_k1_s0043
                                 - 0.411999647473543 z_k1_s0044
                                 - 0.453999433482247 z_k1_s0045
                                 - 0.753001403320538 z_k1_s0046
                                 - 0.934002227746211 z_k1_s0047
                                 - 0.78200124147502 z_k1_s0050
                                 - 0.932997263288442 z_k1_s0051
                                 - 0.296999849124524 z_k1_s0052
                                 - 0.405999766621431 z_k1_s0053
                                 - 0.731998716534102 z_k1_s0054
                                 - 0.583001076680072 z_k1_s0055
                                 - 0.368999617203396 z_k1_s0056
                                 - 1.26999859516625 z_k1_s0057
                                 - 0.411999275413264 z_k1_s0060
                                 - 0.454999867829505 z_k1_s0061
                                 - 0.25799953650152 z_k1_s0062
                                 - 0.17299983715397 z_k1_s0063
                                 - 0.144999829833708 z_k1_s0064
                                 - 0.15599989728446 z_k1_s0065
                                 - 0.137999878228592 z_k1_s0066
                                 - 0.272000243310938 z_k1_s0067
                                 - 0.401998949324018 z_k1_s0070
                                 - 0.280999644851555 z_k1_s0071
                                 - 0.355999650704379 z_k1_s0072
                                 - 0.50899865437296 z_k1_s0073
                                 - 0.507999605184816 z_k1_s0074
                                 - 0.217999910446963 z_k1_s0075
                                 - 0.400999900135874 z_k1_s0076
                                 - 0.398998727762084 z_k1_s0077
                                 - 0.404999195616053 z_k1_s0100
                                 - 0.463000703591032 z_k1_s0101
                                 - 0.262999406298805 z_k1_s0102
                                 - 0.410000612648644 z_k1_s0103
                                 - 0.114000041136426 z_k1_s0104
                                 - 0.35499947329312 z_k1_s0105
                                 - 0.188000063614455 z_k1_s0106
                                 - 0.210000037756566 z_k1_s0107
                                 - 0.155000427131606 z_k1_s0110
                                 - 0.337000254645234 z_k1_s0111
                                 - 0.47100029426664 z_k1_s0112
                                 - 0.097000276681736 z_k1_s0113
                                 - 0.217001304165869 z_k1_s0114
                                 - 0.343000721420394 z_k1_s0115
                                 - 0.302999686590761 z_k1_s0116
                                 - 0.154000349335746 z_k1_s0117
                                 - 0.1460000562794 z_k1_s0120
                                 - 0.108999923217505 z_k1_s0121
                                 - 0.218000170634989 z_k1_s0122
                                 - 0.395000390971649 z_k1_s0123
                                 - 0.125000002663188 z_k1_s0124
                                 - 0.190000639480004 z_k1_s0125
                                 - 0.293000593692874 z_k1_s0126
                                 - 0.145000061445294 z_k1_s0127
                                 - 0.151999913162643 z_k1_s0130
                                 - 0.123999841585431 z_k1_s0131
                                 - 0.569999080110708 z_k1_s0132
                                 - 0.273000460178129 z_k1_s0133
                                 - 0.131999862036063 z_k1_s0134
                                 - 0.545000204700887 z_k1_s0135
                                 - 0.360999258936995 z_k1_s0136
                                 - 0.219999611593652 z_k1_s0137
                                 - 0.70600094891682 z_k1_s0140
                                 - 0.309000471454898 z_k1_s0141
                                 - 1.09400194541291 z_k1_s0142
                                 - 0.92000283425949 z_k1_s0143
                                 - 0.254000594725702 z_k1_s0144
                                 - 0.744001806043888 z_k1_s0145
                                 - 0.679000473616214 z_k1_s0146
                                 - 0.375000730010555 z_k1_s0147
                                 - 0.15300009902112 z_k1_s0150
                                 - 0.192999839268391 z_k1_s0151
                                 - 0.133000069397439 z_k1_s0152
                                 - 0.460999789625585 z_k1_s0153
                                 - 0.0949999812124376 z_k1_s0154
                                 - 0.228000529110012 z_k1_s0155
                                 - 0.111999974492557 z_k1_s0156
                                 - 0.220000006860396 z_k1_s0157
                                 - 0.189000254717286 z_k1_s0160
                                 - 0.297000611970257 z_k1_s0161
                                 - 0.284999325596315 z_k1_s0162
                                 - 0.262000211966662 z_k1_s0163
                                 - 0.307001331114815 z_k1_s0164
                                 - 0.215000642591486 z_k1_s0165
                                 - 0.251000267708592 z_k1_s0166
                                 - 0.311000348571223 z_k1_s0167
                                 - 0.167000181118811 z_k1_s0170
                                 - 0.258000230602671 z_k1_s0171
                                 - 0.701001893844043 z_k1_s0172
                                 - 0.237000286914148 z_k1_s0173
                                 - 0.281000378571239 z_k1_s0174
                                 - 0.644001208117306 z_k1_s0175
                                 - 0.290000270580531 z_k1_s0176
                                 - 0.356999042704885 z_k1_s0177
                                 - 0.557999936910414 z_k1_s1000
                                 - 0.216999942565191 z_k1_s1001
                                 - 0.628998729625924 z_k1_s1002
                                 - 0.130999849343609 z_k1_s1003
                                 - 0.303999668467595 z_k1_s1004
                                 - 0.417000207561849 z_k1_s1005
                                 - 0.24199964257758 z_k1_s1006
                                 - 0.462999300264971 z_k1_s1007
                                 - 0.251999790287099 z_k1_s1010
                                 - 0.355999080493121 z_k1_s1011
                                 - 0.426999653667025 z_k1_s1012
                                 - 0.570999719259583 z_k1_s1013
                                 - 0.361999083226144 z_k1_s1014
                                 - 0.788998736893169 z_k1_s1015
                                 - 0.25599990027574 z_k1_s1016
                                 - 0.231999889343647 z_k1_s1017
                                 - 0.515998237131846 z_k1_s1020
                                 - 1.36399787236113 z_k1_s1021
                                 - 0.291999181272527 z_k1_s1022
                                 - 0.783996088309424 z_k1_s1023
                                 - 0.652999150577076 z_k1_s1024
                                 - 0.638997160291974 z_k1_s1025
                                 - 0.878998317277641 z_k1_s1026
                                 - 0.944994688660457 z_k1_s1027
                                 - 1.05400312914983 z_k1_s1030
                                 - 0.945002762146654 z_k1_s1031
                                 - 0.687000410626678 z_k1_s1032
                                 - 1.01900133262527 z_k1_s1033
                                 - 0.7310000551709 z_k1_s1034
                                 - 1.81400184109721 z_k1_s1035
                                 - 0.831001049353866 z_k1_s1036
                                 - 0.543000534299764 z_k1_s1037
                                 - 0.716000455703508 z_k1_s1040
                                 - 0.612002978800013 z_k1_s1041
                                 - 1.51000331058987 z_k1_s1042
                                 - 0.321000586302737 z_k1_s1043
                                 - 0.998003847819703 z_k1_s1044
                                 - 0.398000489862294 z_k1_s1045
                                 - 0.524000800214869 z_k1_s1046
                                 - 0.642999090623189 z_k1_s1047
                                 - 0.140999955042178 z_k1_s1050
                                 - 0.148999935845996 z_k1_s1051
                                 - 0.427998738304393 z_k1_s1052
                                 - 0.431000550030052 z_k1_s1053
                                 - 0.297000988916042 z_k1_s1054
                                 - 0.192000008691446 z_k1_s1055
                                 - 0.138000020915721 z_k1_s1056
                                 - 0.570999040597375 z_k1_s1057
                                 - 0.256000269575399 z_k1_s1060
                                 - 0.151999905623033 z_k1_s1061
                                 - 0.193000694132747 z_k1_s1062
                                 - 0.279999686410927 z_k1_s1063
                                 - 0.244999725609561 z_k1_s1064
                                 - 0.176999902909709 z_k1_s1065
                                 - 0.106999981306977 z_k1_s1066
                                 - 0.360000102528057 z_k1_s1067
                                 - 0.835000106866196 z_k1_s1070
                                 - 0.535000089399279 z_k1_s1071
                                 - 0.415998952421855 z_k1_s1072
                                 - 0.555998028574008 z_k1_s1073
                                 - 0.206999907202277 z_k1_s1074
                                 - 0.534000520390628 z_k1_s1075
                                 - 0.171999963414412 z_k1_s1076
                                 - 0.239999664473881 z_k1_s1077
                                 - 0.384998668002341 z_k1_s1100
                                 - 0.100999691379702 z_k1_s1101
                                 - 0.400999411719308 z_k1_s1102
                                 - 0.0939996554525828 z_k1_s1103
                                 - 0.171998897987118 z_k1_s1104
                                 - 0.133999529951485 z_k1_s1105
                                 - 0.181998949311573 z_k1_s1106
                                 - 0.184999295507829 z_k1_s1107
                                 - 0.17299952024797 z_k1_s1110
                                 - 0.298000768550419 z_k1_s1111
                                 - 0.15700009626613 z_k1_s1112
                                 - 0.120000207869176 z_k1_s1113
                                 - 0.126000097987492 z_k1_s1114
                                 - 0.155000698894029 z_k1_s1115
                                 - 0.21300020270509 z_k1_s1116
                                 - 0.17299952024797 z_k1_s1117
                                 - 0.161000199057396 z_k1_s1120
                                 - 0.0669999898548829 z_k1_s1121
                                 - 0.210000587814201 z_k1_s1122
                                 - 0.212999661696675 z_k1_s1123
                                 - 0.165999494195378 z_k1_s1124
                                 - 0.0640000353718568 z_k1_s1125
                                 - 0.257999607942462 z_k1_s1126
                                 - 0.119000081494556 z_k1_s1127
                                 - 0.115999676502858 z_k1_s1130
                                 - 0.263998729180134 z_k1_s1131
                                 - 0.429999866387479 z_k1_s1132
                                 - 0.170999586331655 z_k1_s1133
                                 - 0.262999416454414 z_k1_s1134
                                 - 0.157999721917316 z_k1_s1135
                                 - 0.286000369107839 z_k1_s1136
                                 - 0.406998913734054 z_k1_s1137
                                 - 0.402999456011428 z_k1_s1140
                                 - 0.15499994037354 z_k1_s1141
                                 - 0.27300025884609 z_k1_s1142
                                 - 0.144999961330076 z_k1_s1143
                                 - 0.279999766816789 z_k1_s1144
                                 - 0.128999924140574 z_k1_s1145
                                 - 0.196999286596415 z_k1_s1146
                                 - 0.186000564927882 z_k1_s1147
                                 - 0.28400028577478 z_k1_s1150
                                 - 0.329000241625637 z_k1_s1151
                                 - 0.336000079060083 z_k1_s1152
                                 - 0.314000402308809 z_k1_s1153
                                 - 1.10900020620779 z_k1_s1154
                                 - 1.1020025582752 z_k1_s1155
                                 - 0.598999538830412 z_k1_s1156
                                 - 0.306000400426426 z_k1_s1157
                                 - 0.0960001253516438 z_k1_s1160
                                 - 0.0900001500172058 z_k1_s1161
                                 - 0.0560001164551785 z_k1_s1162
                                 - 0.23099988237688 z_k1_s1163
                                 - 0.128999677690671 z_k1_s1164
                                 - 0.200000564482962 z_k1_s1165
                                 - 0.191000393481051 z_k1_s1166
                                 - 0.0470000494533944 z_k1_s1167
                                 - 0.334999315255768 z_k1_s1170
                                 - 0.106000066313767 z_k1_s1171
                                 - 0.223000064501604 z_k1_s1172
                                 - 0.139000014835977 z_k1_s1173
                                 - 0.179999997212057 z_k1_s1174
                                 - 0.205000676380403 z_k1_s1175
                                 - 0.151000065616782 z_k1_s1176
                                 - 0.189999657279391 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.28849 w_k1_ADM_s000
                                 + 1.46864 w_k1_AXP_s000
                                 + 1.62586 w_k1_DIS_s000
                                 + 1.39438 w_k1_FDX_s000
                                 + 1.00949 w_k1_IBM_s000
                                 + 1.28237 w_k1_INTC_s000
                                 + 1.28503 w_k1_KO_s000 + 1.61144 w_k1_MAT_s000
                                 + 1.40479 w_k1_MSFT_s000
                                 + 1.15047 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.07331 w_k1_ADM_s000
                                 + 0.815329 w_k1_AXP_s000
                                 + 1.19127 w_k1_DIS_s000
                                 + 1.06596 w_k1_FDX_s000
                                 + 0.795137 w_k1_IBM_s000
                                 + 1.09538 w_k1_INTC_s000
                                 + 1.05606 w_k1_KO_s000
                                 + 0.692424 w_k1_MAT_s000
                                 + 1.22303 w_k1_MSFT_s000
                                 + 0.952723 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.04831 w_k1_ADM_s000
                                 + 1.05298 w_k1_AXP_s000
                                 + 1.04166 w_k1_DIS_s000
                                 + 0.943486 w_k1_FDX_s000
                                 + 1.55655 w_k1_IBM_s000
                                 + 1.10171 w_k1_INTC_s000
                                 + 1.15182 w_k1_KO_s000 + 1.69847 w_k1_MAT_s000
                                 + 0.974833 w_k1_MSFT_s000
                                 + 1.18168 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.16685 w_k1_ADM_s000
                                 + 1.33503 w_k1_AXP_s000
                                 + 1.25414 w_k1_DIS_s000
                                 + 1.18741 w_k1_FDX_s000
                                 + 0.956819 w_k1_IBM_s000
                                 + 1.26572 w_k1_INTC_s000
                                 + 1.09183 w_k1_KO_s000 + 1.23035 w_k1_MAT_s000
                                 + 1.31567 w_k1_MSFT_s000
                                 + 1.16135 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.964431 w_k1_ADM_s000
                                 + 0.662863 w_k1_AXP_s000
                                 + 0.723071 w_k1_DIS_s000
                                 + 0.828541 w_k1_FDX_s000
                                 + 0.673335 w_k1_IBM_s000
                                 + 0.971697 w_k1_INTC_s000
                                 + 0.722599 w_k1_KO_s000
                                 + 0.620345 w_k1_MAT_s000
                                 + 0.822523 w_k1_MSFT_s000
                                 + 0.832843 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.973838 w_k1_ADM_s000
                                 + 0.759398 w_k1_AXP_s000
                                 + 0.748553 w_k1_DIS_s000
                                 + 0.751997 w_k1_FDX_s000
                                 + 1.35355 w_k1_IBM_s000
                                 + 1.02498 w_k1_INTC_s000
                                 + 0.934736 w_k1_KO_s000
                                 + 1.14189 w_k1_MAT_s000
                                 + 0.966214 w_k1_MSFT_s000
                                 + 0.977138 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.02887 w_k1_ADM_s000
                                 + 0.908386 w_k1_AXP_s000
                                 + 0.860325 w_k1_DIS_s000
                                 + 0.877652 w_k1_FDX_s000
                                 + 1.06242 w_k1_IBM_s000
                                 + 0.935439 w_k1_INTC_s000
                                 + 0.90157 w_k1_KO_s000 + 1.0363 w_k1_MAT_s000
                                 + 0.954938 w_k1_MSFT_s000
                                 + 0.965427 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.226 w_k1_ADM_s000
                                 + 1.13704 w_k1_AXP_s000
                                 + 1.59866 w_k1_DIS_s000
                                 + 1.17918 w_k1_FDX_s000
                                 + 1.66192 w_k1_IBM_s000
                                 + 1.10985 w_k1_INTC_s000
                                 + 1.39376 w_k1_KO_s000 + 1.71967 w_k1_MAT_s000
                                 + 1.2613 w_k1_MSFT_s000 + 1.12483 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.02093 w_k1_ADM_s001
                                 + 1.04234 w_k1_AXP_s001
                                 + 0.792708 w_k1_DIS_s001
                                 + 0.943041 w_k1_FDX_s001
                                 + 0.833763 w_k1_IBM_s001
                                 + 1.00743 w_k1_INTC_s001
                                 + 0.865211 w_k1_KO_s001
                                 + 0.744313 w_k1_MAT_s001
                                 + 0.968468 w_k1_MSFT_s001
                                 + 0.915184 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.28063 w_k1_ADM_s001
                                 + 1.24792 w_k1_AXP_s001
                                 + 1.64548 w_k1_DIS_s001
                                 + 1.24386 w_k1_FDX_s001
                                 + 0.982737 w_k1_IBM_s001
                                 + 1.16986 w_k1_INTC_s001
                                 + 1.40787 w_k1_KO_s001
                                 + 0.940237 w_k1_MAT_s001
                                 + 1.35174 w_k1_MSFT_s001 + 1.2506 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.0687 w_k1_ADM_s001
                                 + 1.42899 w_k1_AXP_s001
                                 + 1.15183 w_k1_DIS_s001
                                 + 1.05688 w_k1_FDX_s001
                                 + 1.42817 w_k1_IBM_s001
                                 + 0.921955 w_k1_INTC_s001
                                 + 1.30122 w_k1_KO_s001 + 1.50699 w_k1_MAT_s001
                                 + 1.11561 w_k1_MSFT_s001
                                 + 1.12539 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02422 w_k1_ADM_s001
                                 + 0.968764 w_k1_AXP_s001
                                 + 0.897836 w_k1_DIS_s001
                                 + 0.785914 w_k1_FDX_s001
                                 + 1.37387 w_k1_IBM_s001
                                 + 0.761543 w_k1_INTC_s001
                                 + 1.09235 w_k1_KO_s001 + 1.06353 w_k1_MAT_s001
                                 + 0.876459 w_k1_MSFT_s001
                                 + 0.924699 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.27323 w_k1_ADM_s001
                                 + 1.33153 w_k1_AXP_s001
                                 + 1.21608 w_k1_DIS_s001
                                 + 1.25648 w_k1_FDX_s001
                                 + 1.06489 w_k1_IBM_s001
                                 + 1.1438 w_k1_INTC_s001 + 1.13202 w_k1_KO_s001
                                 + 1.00033 w_k1_MAT_s001
                                 + 1.30108 w_k1_MSFT_s001
                                 + 1.14443 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.892068 w_k1_ADM_s001
                                 + 0.844034 w_k1_AXP_s001
                                 + 1.07032 w_k1_DIS_s001
                                 + 0.935309 w_k1_FDX_s001
                                 + 0.929814 w_k1_IBM_s001
                                 + 1.03897 w_k1_INTC_s001
                                 + 0.963685 w_k1_KO_s001
                                 + 0.539042 w_k1_MAT_s001
                                 + 1.12596 w_k1_MSFT_s001
                                 + 1.05804 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05163 w_k1_ADM_s001
                                 + 1.07782 w_k1_AXP_s001
                                 + 0.949291 w_k1_DIS_s001
                                 + 1.03905 w_k1_FDX_s001
                                 + 0.975885 w_k1_IBM_s001
                                 + 1.06148 w_k1_INTC_s001
                                 + 1.06754 w_k1_KO_s001
                                 + 0.836274 w_k1_MAT_s001
                                 + 1.0347 w_k1_MSFT_s001 + 1.00298 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.847261 w_k1_ADM_s001
                                 + 0.909688 w_k1_AXP_s001
                                 + 0.626612 w_k1_DIS_s001
                                 + 0.804046 w_k1_FDX_s001
                                 + 0.703529 w_k1_IBM_s001
                                 + 0.765759 w_k1_INTC_s001
                                 + 0.66924 w_k1_KO_s001
                                 + 0.520007 w_k1_MAT_s001
                                 + 0.811928 w_k1_MSFT_s001
                                 + 0.820085 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.07094 w_k1_ADM_s002
                                 + 1.36437 w_k1_AXP_s002
                                 + 0.96853 w_k1_DIS_s002
                                 + 1.02311 w_k1_FDX_s002
                                 + 1.19463 w_k1_IBM_s002
                                 + 1.04784 w_k1_INTC_s002
                                 + 1.16754 w_k1_KO_s002 + 1.45946 w_k1_MAT_s002
                                 + 0.947627 w_k1_MSFT_s002
                                 + 0.972413 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.12042 w_k1_ADM_s002
                                 + 1.22953 w_k1_AXP_s002
                                 + 1.69913 w_k1_DIS_s002
                                 + 1.24564 w_k1_FDX_s002
                                 + 1.07894 w_k1_IBM_s002
                                 + 1.07587 w_k1_INTC_s002
                                 + 1.42741 w_k1_KO_s002 + 1.24111 w_k1_MAT_s002
                                 + 1.34394 w_k1_MSFT_s002
                                 + 1.13889 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.12932 w_k1_ADM_s002
                                 + 1.00989 w_k1_AXP_s002
                                 + 0.920826 w_k1_DIS_s002
                                 + 1.05149 w_k1_FDX_s002
                                 + 0.589445 w_k1_IBM_s002
                                 + 0.893929 w_k1_INTC_s002
                                 + 0.872906 w_k1_KO_s002
                                 + 0.776126 w_k1_MAT_s002
                                 + 1.02803 w_k1_MSFT_s002
                                 + 0.981646 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.981344 w_k1_ADM_s002
                                 + 0.775874 w_k1_AXP_s002
                                 + 0.914002 w_k1_DIS_s002
                                 + 0.859869 w_k1_FDX_s002
                                 + 1.02363 w_k1_IBM_s002
                                 + 1.09147 w_k1_INTC_s002
                                 + 1.16809 w_k1_KO_s002
                                 + 0.639158 w_k1_MAT_s002
                                 + 0.944562 w_k1_MSFT_s002
                                 + 0.912226 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.975837 w_k1_ADM_s002
                                 + 0.840142 w_k1_AXP_s002
                                 + 1.13078 w_k1_DIS_s002 + 1.0228 w_k1_FDX_s002
                                 + 0.947473 w_k1_IBM_s002
                                 + 0.988123 w_k1_INTC_s002
                                 + 1.13131 w_k1_KO_s002
                                 + 0.734314 w_k1_MAT_s002
                                 + 1.15258 w_k1_MSFT_s002
                                 + 0.924875 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.92934 w_k1_ADM_s002
                                 + 0.795346 w_k1_AXP_s002
                                 + 0.650093 w_k1_DIS_s002
                                 + 0.7686 w_k1_FDX_s002
                                 + 0.763369 w_k1_IBM_s002
                                 + 0.707135 w_k1_INTC_s002
                                 + 0.827755 w_k1_KO_s002
                                 + 0.916861 w_k1_MAT_s002
                                 + 0.785636 w_k1_MSFT_s002
                                 + 0.857001 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.33641 w_k1_ADM_s002
                                 + 1.56906 w_k1_AXP_s002
                                 + 1.34185 w_k1_DIS_s002
                                 + 1.41473 w_k1_FDX_s002
                                 + 1.02951 w_k1_IBM_s002
                                 + 1.05996 w_k1_INTC_s002
                                 + 1.17555 w_k1_KO_s002 + 1.77406 w_k1_MAT_s002
                                 + 1.27552 w_k1_MSFT_s002
                                 + 1.08762 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.10665 w_k1_ADM_s002
                                 + 1.21243 w_k1_AXP_s002
                                 + 1.40338 w_k1_DIS_s002
                                 + 1.14176 w_k1_FDX_s002
                                 + 1.27207 w_k1_IBM_s002
                                 + 0.953878 w_k1_INTC_s002
                                 + 1.23062 w_k1_KO_s002 + 1.46645 w_k1_MAT_s002
                                 + 1.11638 w_k1_MSFT_s002 + 1.1545 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.974403 w_k1_ADM_s003
                                 + 0.862562 w_k1_AXP_s003
                                 + 0.817276 w_k1_DIS_s003
                                 + 0.978923 w_k1_FDX_s003
                                 + 0.697066 w_k1_IBM_s003
                                 + 0.904281 w_k1_INTC_s003
                                 + 0.72231 w_k1_KO_s003
                                 + 0.601177 w_k1_MAT_s003
                                 + 0.914769 w_k1_MSFT_s003
                                 + 0.946572 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02949 w_k1_ADM_s003
                                 + 0.890505 w_k1_AXP_s003
                                 + 0.948709 w_k1_DIS_s003
                                 + 0.873648 w_k1_FDX_s003
                                 + 1.18731 w_k1_IBM_s003
                                 + 0.908354 w_k1_INTC_s003
                                 + 1.01461 w_k1_KO_s003
                                 + 0.822387 w_k1_MAT_s003
                                 + 0.887188 w_k1_MSFT_s003
                                 + 0.954548 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.04534 w_k1_ADM_s003
                                 + 1.11949 w_k1_AXP_s003
                                 + 1.11536 w_k1_DIS_s003
                                 + 1.18657 w_k1_FDX_s003 + 0.8217 w_k1_IBM_s003
                                 + 0.904449 w_k1_INTC_s003
                                 + 0.874519 w_k1_KO_s003
                                 + 0.889398 w_k1_MAT_s003
                                 + 1.0286 w_k1_MSFT_s003
                                 + 0.994597 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.06074 w_k1_ADM_s003
                                 + 1.12583 w_k1_AXP_s003
                                 + 0.972897 w_k1_DIS_s003
                                 + 1.08605 w_k1_FDX_s003
                                 + 1.09627 w_k1_IBM_s003
                                 + 1.2162 w_k1_INTC_s003 + 1.06213 w_k1_KO_s003
                                 + 0.914506 w_k1_MAT_s003
                                 + 1.13469 w_k1_MSFT_s003
                                 + 1.03751 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.786729 w_k1_ADM_s003
                                 + 0.542788 w_k1_AXP_s003
                                 + 0.667193 w_k1_DIS_s003
                                 + 0.781759 w_k1_FDX_s003
                                 + 0.742024 w_k1_IBM_s003
                                 + 0.734462 w_k1_INTC_s003
                                 + 0.684225 w_k1_KO_s003
                                 + 0.544732 w_k1_MAT_s003
                                 + 0.805684 w_k1_MSFT_s003
                                 + 0.734378 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.857977 w_k1_ADM_s003
                                 + 0.806017 w_k1_AXP_s003
                                 + 1.1753 w_k1_DIS_s003
                                 + 0.969687 w_k1_FDX_s003
                                 + 0.952412 w_k1_IBM_s003
                                 + 0.950601 w_k1_INTC_s003
                                 + 0.964236 w_k1_KO_s003
                                 + 0.765758 w_k1_MAT_s003
                                 + 1.08121 w_k1_MSFT_s003 + 1.0028 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.1399 w_k1_ADM_s003
                                 + 1.4172 w_k1_AXP_s003 + 1.54119 w_k1_DIS_s003
                                 + 1.34814 w_k1_FDX_s003
                                 + 1.39797 w_k1_IBM_s003
                                 + 1.09495 w_k1_INTC_s003
                                 + 1.26745 w_k1_KO_s003 + 1.10996 w_k1_MAT_s003
                                 + 1.23097 w_k1_MSFT_s003
                                 + 1.17038 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.881504 w_k1_ADM_s003
                                 + 0.9117 w_k1_AXP_s003
                                 + 0.722914 w_k1_DIS_s003
                                 + 0.923466 w_k1_FDX_s003
                                 + 0.855528 w_k1_IBM_s003
                                 + 0.832225 w_k1_INTC_s003
                                 + 0.746344 w_k1_KO_s003
                                 + 0.820798 w_k1_MAT_s003
                                 + 0.816259 w_k1_MSFT_s003
                                 + 0.928287 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.969932 w_k1_ADM_s004
                                 + 0.769441 w_k1_AXP_s004
                                 + 0.883044 w_k1_DIS_s004
                                 + 0.823837 w_k1_FDX_s004
                                 + 0.991343 w_k1_IBM_s004
                                 + 0.918532 w_k1_INTC_s004
                                 + 1.03223 w_k1_KO_s004
                                 + 0.739539 w_k1_MAT_s004
                                 + 0.867896 w_k1_MSFT_s004
                                 + 0.863715 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.1639 w_k1_ADM_s004
                                 + 0.90151 w_k1_AXP_s004 + 1.0033 w_k1_DIS_s004
                                 + 1.01978 w_k1_FDX_s004
                                 + 0.75838 w_k1_IBM_s004
                                 + 0.996717 w_k1_INTC_s004
                                 + 0.95953 w_k1_KO_s004
                                 + 0.656078 w_k1_MAT_s004
                                 + 0.956233 w_k1_MSFT_s004
                                 + 0.958774 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.27918 w_k1_ADM_s004
                                 + 1.24131 w_k1_AXP_s004
                                 + 1.37592 w_k1_DIS_s004
                                 + 1.15202 w_k1_FDX_s004
                                 + 1.31121 w_k1_IBM_s004
                                 + 1.22607 w_k1_INTC_s004
                                 + 1.40758 w_k1_KO_s004 + 1.31635 w_k1_MAT_s004
                                 + 1.01032 w_k1_MSFT_s004
                                 + 1.05433 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.994628 w_k1_ADM_s004
                                 + 1.16511 w_k1_AXP_s004
                                 + 1.14491 w_k1_DIS_s004
                                 + 0.965707 w_k1_FDX_s004
                                 + 1.67097 w_k1_IBM_s004
                                 + 0.979324 w_k1_INTC_s004
                                 + 1.40127 w_k1_KO_s004 + 1.64863 w_k1_MAT_s004
                                 + 0.989943 w_k1_MSFT_s004
                                 + 1.1611 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.3185 w_k1_ADM_s004
                                 + 1.43636 w_k1_AXP_s004
                                 + 1.07079 w_k1_DIS_s004
                                 + 1.23972 w_k1_FDX_s004
                                 + 0.925702 w_k1_IBM_s004
                                 + 0.935334 w_k1_INTC_s004
                                 + 0.963733 w_k1_KO_s004
                                 + 0.985193 w_k1_MAT_s004
                                 + 0.899274 w_k1_MSFT_s004
                                 + 1.01081 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.871149 w_k1_ADM_s004
                                 + 0.720739 w_k1_AXP_s004
                                 + 0.574827 w_k1_DIS_s004
                                 + 0.739364 w_k1_FDX_s004
                                 + 0.586961 w_k1_IBM_s004
                                 + 0.69952 w_k1_INTC_s004
                                 + 0.674938 w_k1_KO_s004
                                 + 0.622948 w_k1_MAT_s004
                                 + 0.678857 w_k1_MSFT_s004
                                 + 0.840592 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.5074 w_k1_ADM_s004
                                 + 1.33192 w_k1_AXP_s004
                                 + 1.49418 w_k1_DIS_s004
                                 + 1.22518 w_k1_FDX_s004
                                 + 1.04391 w_k1_IBM_s004
                                 + 0.917523 w_k1_INTC_s004
                                 + 1.31937 w_k1_KO_s004 + 1.0339 w_k1_MAT_s004
                                 + 1.16158 w_k1_MSFT_s004
                                 + 1.17613 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.23961 w_k1_ADM_s004
                                 + 1.08948 w_k1_AXP_s004
                                 + 0.756204 w_k1_DIS_s004
                                 + 0.878847 w_k1_FDX_s004
                                 + 0.952225 w_k1_IBM_s004
                                 + 0.86502 w_k1_INTC_s004
                                 + 0.849022 w_k1_KO_s004
                                 + 1.00475 w_k1_MAT_s004
                                 + 0.70854 w_k1_MSFT_s004
                                 + 0.921704 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.837665 w_k1_ADM_s005
                                 + 0.810941 w_k1_AXP_s005
                                 + 0.810262 w_k1_DIS_s005
                                 + 0.850228 w_k1_FDX_s005
                                 + 1.17933 w_k1_IBM_s005
                                 + 0.747948 w_k1_INTC_s005
                                 + 0.994989 w_k1_KO_s005
                                 + 0.947042 w_k1_MAT_s005
                                 + 0.764339 w_k1_MSFT_s005
                                 + 0.87042 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.018 w_k1_ADM_s005
                                 + 1.15452 w_k1_AXP_s005
                                 + 1.13439 w_k1_DIS_s005
                                 + 1.06476 w_k1_FDX_s005
                                 + 1.20092 w_k1_IBM_s005
                                 + 0.887542 w_k1_INTC_s005
                                 + 1.22864 w_k1_KO_s005 + 1.03092 w_k1_MAT_s005
                                 + 1.00256 w_k1_MSFT_s005
                                 + 1.08155 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.856179 w_k1_ADM_s005
                                 + 1.24061 w_k1_AXP_s005
                                 + 1.16983 w_k1_DIS_s005
                                 + 1.01213 w_k1_FDX_s005
                                 + 1.79633 w_k1_IBM_s005
                                 + 0.958908 w_k1_INTC_s005
                                 + 1.38936 w_k1_KO_s005 + 1.95745 w_k1_MAT_s005
                                 + 1.01598 w_k1_MSFT_s005 + 1.0709 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.95911 w_k1_ADM_s005
                                 + 1.04029 w_k1_AXP_s005
                                 + 1.54869 w_k1_DIS_s005
                                 + 1.25276 w_k1_FDX_s005
                                 + 0.868147 w_k1_IBM_s005
                                 + 0.871977 w_k1_INTC_s005
                                 + 1.14828 w_k1_KO_s005
                                 + 0.935947 w_k1_MAT_s005
                                 + 1.13583 w_k1_MSFT_s005
                                 + 1.12664 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.44571 w_k1_ADM_s005
                                 + 1.14903 w_k1_AXP_s005
                                 + 1.08393 w_k1_DIS_s005
                                 + 1.23459 w_k1_FDX_s005
                                 + 0.732176 w_k1_IBM_s005
                                 + 0.898462 w_k1_INTC_s005
                                 + 0.889907 w_k1_KO_s005
                                 + 0.756719 w_k1_MAT_s005
                                 + 1.11457 w_k1_MSFT_s005
                                 + 1.02204 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.963284 w_k1_ADM_s005
                                 + 1.27838 w_k1_AXP_s005
                                 + 0.858041 w_k1_DIS_s005
                                 + 0.989224 w_k1_FDX_s005
                                 + 0.996284 w_k1_IBM_s005
                                 + 0.798711 w_k1_INTC_s005
                                 + 1.00035 w_k1_KO_s005 + 1.25085 w_k1_MAT_s005
                                 + 0.810344 w_k1_MSFT_s005
                                 + 0.976022 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.74068 w_k1_ADM_s005
                                 + 1.76695 w_k1_AXP_s005
                                 + 2.05254 w_k1_DIS_s005
                                 + 1.55285 w_k1_FDX_s005
                                 + 1.40657 w_k1_IBM_s005 + 1.284 w_k1_INTC_s005
                                 + 1.81437 w_k1_KO_s005 + 1.83732 w_k1_MAT_s005
                                 + 1.54253 w_k1_MSFT_s005
                                 + 1.40628 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.02825 w_k1_ADM_s005
                                 + 0.761185 w_k1_AXP_s005
                                 + 0.67439 w_k1_DIS_s005
                                 + 0.779156 w_k1_FDX_s005
                                 + 0.742318 w_k1_IBM_s005
                                 + 0.775702 w_k1_INTC_s005
                                 + 0.829848 w_k1_KO_s005
                                 + 0.734567 w_k1_MAT_s005
                                 + 0.806694 w_k1_MSFT_s005
                                 + 0.896247 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.29128 w_k1_ADM_s006
                                 + 1.10118 w_k1_AXP_s006 + 1.0903 w_k1_DIS_s006
                                 + 1.11207 w_k1_FDX_s006
                                 + 0.838304 w_k1_IBM_s006
                                 + 0.956602 w_k1_INTC_s006
                                 + 0.949969 w_k1_KO_s006
                                 + 0.783523 w_k1_MAT_s006
                                 + 1.07691 w_k1_MSFT_s006
                                 + 0.991432 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.845296 w_k1_ADM_s006
                                 + 1.00747 w_k1_AXP_s006
                                 + 0.805839 w_k1_DIS_s006
                                 + 0.83118 w_k1_FDX_s006
                                 + 1.12237 w_k1_IBM_s006
                                 + 1.01656 w_k1_INTC_s006 + 1.0395 w_k1_KO_s006
                                 + 1.13024 w_k1_MAT_s006
                                 + 0.906255 w_k1_MSFT_s006
                                 + 0.946068 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.24887 w_k1_ADM_s006
                                 + 1.28799 w_k1_AXP_s006
                                 + 1.47068 w_k1_DIS_s006
                                 + 1.08668 w_k1_FDX_s006
                                 + 1.48123 w_k1_IBM_s006
                                 + 1.20852 w_k1_INTC_s006
                                 + 1.39727 w_k1_KO_s006 + 1.47335 w_k1_MAT_s006
                                 + 1.33321 w_k1_MSFT_s006
                                 + 1.19616 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.819052 w_k1_ADM_s006
                                 + 1.05451 w_k1_AXP_s006
                                 + 0.893523 w_k1_DIS_s006
                                 + 0.921999 w_k1_FDX_s006
                                 + 0.928209 w_k1_IBM_s006
                                 + 1.35444 w_k1_INTC_s006
                                 + 0.893138 w_k1_KO_s006
                                 + 0.873858 w_k1_MAT_s006
                                 + 1.20002 w_k1_MSFT_s006
                                 + 0.991772 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.38536 w_k1_ADM_s006
                                 + 1.82418 w_k1_AXP_s006
                                 + 1.81267 w_k1_DIS_s006
                                 + 1.48583 w_k1_FDX_s006
                                 + 1.32363 w_k1_IBM_s006
                                 + 1.09892 w_k1_INTC_s006
                                 + 1.22169 w_k1_KO_s006 + 1.33802 w_k1_MAT_s006
                                 + 1.41177 w_k1_MSFT_s006
                                 + 1.12881 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.740183 w_k1_ADM_s006
                                 + 0.826871 w_k1_AXP_s006
                                 + 0.680607 w_k1_DIS_s006
                                 + 0.802824 w_k1_FDX_s006
                                 + 0.852407 w_k1_IBM_s006
                                 + 1.04678 w_k1_INTC_s006
                                 + 0.789632 w_k1_KO_s006
                                 + 0.86778 w_k1_MAT_s006
                                 + 0.990736 w_k1_MSFT_s006
                                 + 0.836951 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.900076 w_k1_ADM_s006
                                 + 1.15877 w_k1_AXP_s006
                                 + 0.839575 w_k1_DIS_s006
                                 + 0.83643 w_k1_FDX_s006
                                 + 1.48658 w_k1_IBM_s006
                                 + 0.910243 w_k1_INTC_s006
                                 + 1.09381 w_k1_KO_s006 + 1.55213 w_k1_MAT_s006
                                 + 0.907061 w_k1_MSFT_s006
                                 + 0.83353 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.97993 w_k1_ADM_s006
                                 + 0.800432 w_k1_AXP_s006
                                 + 0.583578 w_k1_DIS_s006
                                 + 0.747123 w_k1_FDX_s006
                                 + 0.65853 w_k1_IBM_s006
                                 + 0.825237 w_k1_INTC_s006
                                 + 0.668667 w_k1_KO_s006
                                 + 0.495655 w_k1_MAT_s006
                                 + 0.795616 w_k1_MSFT_s006
                                 + 0.731632 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.05238 w_k1_ADM_s007
                                 + 0.977756 w_k1_AXP_s007
                                 + 1.27651 w_k1_DIS_s007
                                 + 1.00064 w_k1_FDX_s007
                                 + 1.31963 w_k1_IBM_s007
                                 + 1.11389 w_k1_INTC_s007
                                 + 1.36666 w_k1_KO_s007 + 1.66208 w_k1_MAT_s007
                                 + 1.24953 w_k1_MSFT_s007
                                 + 1.16097 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.39417 w_k1_ADM_s007
                                 + 1.39098 w_k1_AXP_s007
                                 + 1.32724 w_k1_DIS_s007
                                 + 1.23172 w_k1_FDX_s007
                                 + 0.877371 w_k1_IBM_s007
                                 + 1.00292 w_k1_INTC_s007
                                 + 1.05612 w_k1_KO_s007 + 1.30685 w_k1_MAT_s007
                                 + 1.21332 w_k1_MSFT_s007
                                 + 1.24168 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.22918 w_k1_ADM_s007
                                 + 1.08029 w_k1_AXP_s007
                                 + 1.54769 w_k1_DIS_s007
                                 + 1.35244 w_k1_FDX_s007
                                 + 0.980935 w_k1_IBM_s007
                                 + 1.20849 w_k1_INTC_s007 + 1.1627 w_k1_KO_s007
                                 + 1.09858 w_k1_MAT_s007
                                 + 1.31419 w_k1_MSFT_s007
                                 + 1.14732 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.03577 w_k1_ADM_s007
                                 + 1.28811 w_k1_AXP_s007
                                 + 1.05917 w_k1_DIS_s007
                                 + 1.12636 w_k1_FDX_s007
                                 + 0.909307 w_k1_IBM_s007
                                 + 1.1492 w_k1_INTC_s007 + 0.87879 w_k1_KO_s007
                                 + 1.17751 w_k1_MAT_s007
                                 + 1.08595 w_k1_MSFT_s007
                                 + 0.994682 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.00301 w_k1_ADM_s007
                                 + 1.02278 w_k1_AXP_s007
                                 + 0.925953 w_k1_DIS_s007
                                 + 0.92933 w_k1_FDX_s007
                                 + 1.19349 w_k1_IBM_s007
                                 + 0.87558 w_k1_INTC_s007
                                 + 1.04153 w_k1_KO_s007 + 1.19272 w_k1_MAT_s007
                                 + 1.01593 w_k1_MSFT_s007
                                 + 0.992916 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.819738 w_k1_ADM_s007
                                 + 0.653665 w_k1_AXP_s007
                                 + 0.868787 w_k1_DIS_s007
                                 + 0.692748 w_k1_FDX_s007
                                 + 1.13419 w_k1_IBM_s007
                                 + 0.714566 w_k1_INTC_s007
                                 + 0.984077 w_k1_KO_s007
                                 + 1.05539 w_k1_MAT_s007
                                 + 0.872125 w_k1_MSFT_s007
                                 + 0.910499 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.1281 w_k1_ADM_s007
                                 + 0.891212 w_k1_AXP_s007
                                 + 0.800073 w_k1_DIS_s007
                                 + 0.928481 w_k1_FDX_s007
                                 + 0.688648 w_k1_IBM_s007
                                 + 1.07046 w_k1_INTC_s007
                                 + 0.841322 w_k1_KO_s007
                                 + 0.756108 w_k1_MAT_s007
                                 + 0.963107 w_k1_MSFT_s007
                                 + 0.959636 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.770623 w_k1_ADM_s007
                                 + 0.695518 w_k1_AXP_s007
                                 + 0.727347 w_k1_DIS_s007
                                 + 0.776368 w_k1_FDX_s007
                                 + 0.766509 w_k1_IBM_s007
                                 + 0.835499 w_k1_INTC_s007
                                 + 0.768725 w_k1_KO_s007
                                 + 0.653692 w_k1_MAT_s007
                                 + 0.832914 w_k1_MSFT_s007
                                 + 0.853117 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.830959 w_k1_ADM_s010
                                 + 0.699278 w_k1_AXP_s010
                                 + 0.763311 w_k1_DIS_s010
                                 + 0.905587 w_k1_FDX_s010
                                 + 0.676082 w_k1_IBM_s010
                                 + 0.686844 w_k1_INTC_s010
                                 + 0.843615 w_k1_KO_s010
                                 + 0.819994 w_k1_MAT_s010
                                 + 0.852237 w_k1_MSFT_s010
                                 + 0.968009 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.843669 w_k1_ADM_s010
                                 + 0.940964 w_k1_AXP_s010
                                 + 1.29312 w_k1_DIS_s010
                                 + 1.15615 w_k1_FDX_s010
                                 + 1.19695 w_k1_IBM_s010
                                 + 0.924083 w_k1_INTC_s010
                                 + 1.4537 w_k1_KO_s010 + 1.24642 w_k1_MAT_s010
                                 + 1.16172 w_k1_MSFT_s010
                                 + 1.15449 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.93115 w_k1_ADM_s010
                                 + 1.24763 w_k1_AXP_s010
                                 + 1.43418 w_k1_DIS_s010
                                 + 1.06049 w_k1_FDX_s010
                                 + 1.76522 w_k1_IBM_s010
                                 + 0.893682 w_k1_INTC_s010
                                 + 1.51868 w_k1_KO_s010 + 1.54542 w_k1_MAT_s010
                                 + 1.19414 w_k1_MSFT_s010
                                 + 1.39633 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.932527 w_k1_ADM_s010
                                 + 0.833404 w_k1_AXP_s010
                                 + 0.720792 w_k1_DIS_s010
                                 + 0.793894 w_k1_FDX_s010
                                 + 0.992502 w_k1_IBM_s010
                                 + 0.851294 w_k1_INTC_s010
                                 + 0.996471 w_k1_KO_s010
                                 + 0.780356 w_k1_MAT_s010
                                 + 0.875489 w_k1_MSFT_s010
                                 + 1.07405 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.03775 w_k1_ADM_s010
                                 + 1.19993 w_k1_AXP_s010 + 1.5324 w_k1_DIS_s010
                                 + 1.20705 w_k1_FDX_s010
                                 + 0.851174 w_k1_IBM_s010
                                 + 1.28456 w_k1_INTC_s010
                                 + 1.12302 w_k1_KO_s010
                                 + 0.945176 w_k1_MAT_s010
                                 + 1.35198 w_k1_MSFT_s010
                                 + 1.13194 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.14723 w_k1_ADM_s010
                                 + 0.956134 w_k1_AXP_s010
                                 + 1.03286 w_k1_DIS_s010
                                 + 1.12437 w_k1_FDX_s010
                                 + 0.858879 w_k1_IBM_s010
                                 + 0.853221 w_k1_INTC_s010
                                 + 1.04301 w_k1_KO_s010
                                 + 0.805417 w_k1_MAT_s010
                                 + 1.08276 w_k1_MSFT_s010 + 1.2436 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.6045 w_k1_ADM_s010
                                 + 0.60222 w_k1_AXP_s010
                                 + 0.751591 w_k1_DIS_s010
                                 + 0.694589 w_k1_FDX_s010
                                 + 0.886189 w_k1_IBM_s010
                                 + 0.719751 w_k1_INTC_s010
                                 + 0.938967 w_k1_KO_s010
                                 + 1.01311 w_k1_MAT_s010
                                 + 0.809064 w_k1_MSFT_s010
                                 + 0.889613 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.04093 w_k1_ADM_s010
                                 + 1.60578 w_k1_AXP_s010
                                 + 1.46382 w_k1_DIS_s010 + 1.4154 w_k1_FDX_s010
                                 + 1.07042 w_k1_IBM_s010
                                 + 1.16895 w_k1_INTC_s010 + 1.3835 w_k1_KO_s010
                                 + 1.73624 w_k1_MAT_s010
                                 + 1.3836 w_k1_MSFT_s010 + 1.26869 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.935372 w_k1_ADM_s011
                                 + 1.25127 w_k1_AXP_s011
                                 + 0.833164 w_k1_DIS_s011
                                 + 0.981492 w_k1_FDX_s011
                                 + 0.977699 w_k1_IBM_s011
                                 + 0.910091 w_k1_INTC_s011
                                 + 1.02688 w_k1_KO_s011 + 1.39217 w_k1_MAT_s011
                                 + 0.977238 w_k1_MSFT_s011
                                 + 1.16318 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.976381 w_k1_ADM_s011
                                 + 1.00831 w_k1_AXP_s011
                                 + 1.13711 w_k1_DIS_s011
                                 + 0.98887 w_k1_FDX_s011
                                 + 1.30019 w_k1_IBM_s011
                                 + 1.01326 w_k1_INTC_s011
                                 + 1.09924 w_k1_KO_s011 + 0.89957 w_k1_MAT_s011
                                 + 0.993021 w_k1_MSFT_s011
                                 + 1.00657 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.979193 w_k1_ADM_s011
                                 + 0.833343 w_k1_AXP_s011
                                 + 0.804499 w_k1_DIS_s011
                                 + 0.92611 w_k1_FDX_s011
                                 + 0.683113 w_k1_IBM_s011
                                 + 0.993089 w_k1_INTC_s011
                                 + 0.817682 w_k1_KO_s011
                                 + 0.851569 w_k1_MAT_s011
                                 + 0.97803 w_k1_MSFT_s011
                                 + 0.948378 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.51872 w_k1_ADM_s011
                                 + 1.85641 w_k1_AXP_s011
                                 + 1.72054 w_k1_DIS_s011
                                 + 1.33503 w_k1_FDX_s011
                                 + 1.46952 w_k1_IBM_s011
                                 + 1.0504 w_k1_INTC_s011 + 1.39727 w_k1_KO_s011
                                 + 2.07266 w_k1_MAT_s011
                                 + 1.10845 w_k1_MSFT_s011 + 1.4407 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.26662 w_k1_ADM_s011
                                 + 1.34392 w_k1_AXP_s011
                                 + 1.81478 w_k1_DIS_s011
                                 + 1.46437 w_k1_FDX_s011
                                 + 1.02322 w_k1_IBM_s011
                                 + 1.50259 w_k1_INTC_s011
                                 + 1.38374 w_k1_KO_s011 + 1.01301 w_k1_MAT_s011
                                 + 1.55385 w_k1_MSFT_s011
                                 + 1.31952 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.7646 w_k1_ADM_s011
                                 + 0.677147 w_k1_AXP_s011
                                 + 0.97424 w_k1_DIS_s011
                                 + 0.827899 w_k1_FDX_s011
                                 + 0.951709 w_k1_IBM_s011
                                 + 1.02023 w_k1_INTC_s011
                                 + 1.01762 w_k1_KO_s011
                                 + 0.914115 w_k1_MAT_s011
                                 + 1.02264 w_k1_MSFT_s011
                                 + 0.951209 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.958256 w_k1_ADM_s011
                                 + 1.02964 w_k1_AXP_s011
                                 + 1.28439 w_k1_DIS_s011
                                 + 1.15441 w_k1_FDX_s011
                                 + 0.863643 w_k1_IBM_s011
                                 + 1.09191 w_k1_INTC_s011
                                 + 0.983756 w_k1_KO_s011
                                 + 0.602597 w_k1_MAT_s011
                                 + 1.19036 w_k1_MSFT_s011
                                 + 1.00839 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.11694 w_k1_ADM_s011
                                 + 1.32292 w_k1_AXP_s011
                                 + 1.40908 w_k1_DIS_s011
                                 + 1.17806 w_k1_FDX_s011
                                 + 1.37853 w_k1_IBM_s011
                                 + 1.32224 w_k1_INTC_s011
                                 + 1.29068 w_k1_KO_s011 + 1.1459 w_k1_MAT_s011
                                 + 1.14874 w_k1_MSFT_s011
                                 + 1.13855 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.11517 w_k1_ADM_s012
                                 + 1.25188 w_k1_AXP_s012
                                 + 1.01739 w_k1_DIS_s012
                                 + 1.02766 w_k1_FDX_s012 + 1.2749 w_k1_IBM_s012
                                 + 1.09558 w_k1_INTC_s012
                                 + 0.957162 w_k1_KO_s012
                                 + 0.990617 w_k1_MAT_s012
                                 + 1.02988 w_k1_MSFT_s012
                                 + 0.838551 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.56425 w_k1_ADM_s012
                                 + 1.2848 w_k1_AXP_s012 + 1.51901 w_k1_DIS_s012
                                 + 1.34772 w_k1_FDX_s012
                                 + 1.25005 w_k1_IBM_s012
                                 + 1.18204 w_k1_INTC_s012
                                 + 1.14009 w_k1_KO_s012 + 1.03934 w_k1_MAT_s012
                                 + 1.17875 w_k1_MSFT_s012
                                 + 1.07097 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.08089 w_k1_ADM_s012
                                 + 0.700801 w_k1_AXP_s012
                                 + 0.754548 w_k1_DIS_s012
                                 + 0.884163 w_k1_FDX_s012
                                 + 0.671969 w_k1_IBM_s012
                                 + 0.910932 w_k1_INTC_s012
                                 + 0.607486 w_k1_KO_s012
                                 + 0.671059 w_k1_MAT_s012
                                 + 0.89398 w_k1_MSFT_s012
                                 + 0.78468 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.951039 w_k1_ADM_s012
                                 + 0.749668 w_k1_AXP_s012
                                 + 0.810148 w_k1_DIS_s012
                                 + 0.850989 w_k1_FDX_s012
                                 + 0.936302 w_k1_IBM_s012
                                 + 0.816385 w_k1_INTC_s012
                                 + 0.837381 w_k1_KO_s012
                                 + 0.974901 w_k1_MAT_s012
                                 + 0.900165 w_k1_MSFT_s012
                                 + 0.813642 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07431 w_k1_ADM_s012
                                 + 1.24753 w_k1_AXP_s012
                                 + 0.843497 w_k1_DIS_s012
                                 + 0.95587 w_k1_FDX_s012
                                 + 0.866338 w_k1_IBM_s012
                                 + 0.902867 w_k1_INTC_s012
                                 + 0.747284 w_k1_KO_s012
                                 + 1.05838 w_k1_MAT_s012
                                 + 0.880948 w_k1_MSFT_s012
                                 + 0.907439 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.37586 w_k1_ADM_s012
                                 + 1.00959 w_k1_AXP_s012
                                 + 1.31889 w_k1_DIS_s012
                                 + 1.24593 w_k1_FDX_s012
                                 + 0.801541 w_k1_IBM_s012
                                 + 0.960452 w_k1_INTC_s012
                                 + 0.835243 w_k1_KO_s012
                                 + 0.659463 w_k1_MAT_s012
                                 + 1.23156 w_k1_MSFT_s012
                                 + 0.962862 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.968177 w_k1_ADM_s012
                                 + 1.03633 w_k1_AXP_s012
                                 + 1.23128 w_k1_DIS_s012
                                 + 0.93185 w_k1_FDX_s012 + 1.3547 w_k1_IBM_s012
                                 + 0.974469 w_k1_INTC_s012
                                 + 1.19089 w_k1_KO_s012 + 1.27722 w_k1_MAT_s012
                                 + 1.05965 w_k1_MSFT_s012
                                 + 0.97368 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.24409 w_k1_ADM_s012
                                 + 1.14317 w_k1_AXP_s012
                                 + 1.23132 w_k1_DIS_s012
                                 + 1.15621 w_k1_FDX_s012
                                 + 1.08902 w_k1_IBM_s012
                                 + 0.980709 w_k1_INTC_s012
                                 + 1.15556 w_k1_KO_s012 + 1.43283 w_k1_MAT_s012
                                 + 1.32575 w_k1_MSFT_s012
                                 + 1.13478 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.33875 w_k1_ADM_s013
                                 + 1.18967 w_k1_AXP_s013
                                 + 1.85535 w_k1_DIS_s013
                                 + 1.48308 w_k1_FDX_s013
                                 + 1.07352 w_k1_IBM_s013
                                 + 1.39545 w_k1_INTC_s013
                                 + 1.37318 w_k1_KO_s013 + 0.9991 w_k1_MAT_s013
                                 + 1.45572 w_k1_MSFT_s013
                                 + 1.23959 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.989705 w_k1_ADM_s013
                                 + 1.23241 w_k1_AXP_s013
                                 + 1.40764 w_k1_DIS_s013
                                 + 1.09634 w_k1_FDX_s013
                                 + 1.17213 w_k1_IBM_s013
                                 + 1.15984 w_k1_INTC_s013
                                 + 1.33109 w_k1_KO_s013 + 1.15366 w_k1_MAT_s013
                                 + 1.19914 w_k1_MSFT_s013
                                 + 1.15371 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.06557 w_k1_ADM_s013
                                 + 0.955218 w_k1_AXP_s013
                                 + 0.93211 w_k1_DIS_s013
                                 + 0.879095 w_k1_FDX_s013
                                 + 1.2278 w_k1_IBM_s013
                                 + 0.913766 w_k1_INTC_s013
                                 + 1.08242 w_k1_KO_s013
                                 + 0.910675 w_k1_MAT_s013
                                 + 0.940265 w_k1_MSFT_s013
                                 + 0.977135 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.31976 w_k1_ADM_s013
                                 + 1.07677 w_k1_AXP_s013
                                 + 1.12209 w_k1_DIS_s013
                                 + 1.25242 w_k1_FDX_s013
                                 + 0.890011 w_k1_IBM_s013
                                 + 1.03875 w_k1_INTC_s013
                                 + 0.994353 w_k1_KO_s013 + 0.7349 w_k1_MAT_s013
                                 + 1.02944 w_k1_MSFT_s013
                                 + 1.07656 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.42769 w_k1_ADM_s013
                                 + 1.45857 w_k1_AXP_s013
                                 + 1.58206 w_k1_DIS_s013
                                 + 1.20117 w_k1_FDX_s013
                                 + 1.83748 w_k1_IBM_s013
                                 + 1.1651 w_k1_INTC_s013 + 1.65499 w_k1_KO_s013
                                 + 1.53879 w_k1_MAT_s013
                                 + 1.24956 w_k1_MSFT_s013
                                 + 1.30507 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.06189 w_k1_ADM_s013
                                 + 0.667588 w_k1_AXP_s013
                                 + 0.739954 w_k1_DIS_s013
                                 + 0.875502 w_k1_FDX_s013
                                 + 0.740564 w_k1_IBM_s013
                                 + 0.685676 w_k1_INTC_s013
                                 + 0.767384 w_k1_KO_s013
                                 + 0.788045 w_k1_MAT_s013
                                 + 0.776524 w_k1_MSFT_s013
                                 + 0.90423 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.946138 w_k1_ADM_s013
                                 + 0.805676 w_k1_AXP_s013
                                 + 0.862827 w_k1_DIS_s013
                                 + 0.914609 w_k1_FDX_s013
                                 + 0.983456 w_k1_IBM_s013
                                 + 0.904203 w_k1_INTC_s013
                                 + 0.855097 w_k1_KO_s013
                                 + 0.58999 w_k1_MAT_s013
                                 + 0.873166 w_k1_MSFT_s013
                                 + 0.863276 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.3138 w_k1_ADM_s013
                                 + 1.49805 w_k1_AXP_s013
                                 + 1.19926 w_k1_DIS_s013
                                 + 1.20945 w_k1_FDX_s013
                                 + 1.32429 w_k1_IBM_s013
                                 + 1.17627 w_k1_INTC_s013
                                 + 1.35762 w_k1_KO_s013 + 1.67336 w_k1_MAT_s013
                                 + 1.18353 w_k1_MSFT_s013
                                 + 1.20329 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.907262 w_k1_ADM_s014
                                 + 1.16821 w_k1_AXP_s014
                                 + 1.21643 w_k1_DIS_s014
                                 + 0.947442 w_k1_FDX_s014
                                 + 1.34882 w_k1_IBM_s014
                                 + 0.9897 w_k1_INTC_s014 + 1.31999 w_k1_KO_s014
                                 + 1.48189 w_k1_MAT_s014
                                 + 1.00719 w_k1_MSFT_s014
                                 + 1.15592 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.730679 w_k1_ADM_s014
                                 + 0.858361 w_k1_AXP_s014
                                 + 0.957109 w_k1_DIS_s014
                                 + 0.845938 w_k1_FDX_s014
                                 + 1.44119 w_k1_IBM_s014
                                 + 0.819443 w_k1_INTC_s014
                                 + 1.1519 w_k1_KO_s014 + 1.43933 w_k1_MAT_s014
                                 + 0.856361 w_k1_MSFT_s014
                                 + 1.02532 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.759949 w_k1_ADM_s014
                                 + 0.911079 w_k1_AXP_s014
                                 + 0.958562 w_k1_DIS_s014
                                 + 0.884069 w_k1_FDX_s014
                                 + 0.930923 w_k1_IBM_s014
                                 + 0.862848 w_k1_INTC_s014
                                 + 0.93738 w_k1_KO_s014
                                 + 0.985918 w_k1_MAT_s014
                                 + 0.953003 w_k1_MSFT_s014
                                 + 0.96583 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20609 w_k1_ADM_s014
                                 + 1.50916 w_k1_AXP_s014
                                 + 1.46077 w_k1_DIS_s014
                                 + 1.18507 w_k1_FDX_s014 + 1.2753 w_k1_IBM_s014
                                 + 1.02968 w_k1_INTC_s014
                                 + 1.41428 w_k1_KO_s014 + 1.30373 w_k1_MAT_s014
                                 + 1.21921 w_k1_MSFT_s014
                                 + 1.22982 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.904296 w_k1_ADM_s014
                                 + 0.811931 w_k1_AXP_s014
                                 + 0.635069 w_k1_DIS_s014
                                 + 0.740856 w_k1_FDX_s014
                                 + 1.15676 w_k1_IBM_s014
                                 + 0.757857 w_k1_INTC_s014
                                 + 0.904881 w_k1_KO_s014
                                 + 0.987547 w_k1_MAT_s014
                                 + 0.771527 w_k1_MSFT_s014
                                 + 0.992584 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.893594 w_k1_ADM_s014
                                 + 0.760213 w_k1_AXP_s014
                                 + 0.655399 w_k1_DIS_s014
                                 + 0.764894 w_k1_FDX_s014
                                 + 0.616704 w_k1_IBM_s014
                                 + 0.907929 w_k1_INTC_s014
                                 + 0.863575 w_k1_KO_s014
                                 + 0.507337 w_k1_MAT_s014
                                 + 0.704625 w_k1_MSFT_s014
                                 + 0.849589 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.18128 w_k1_ADM_s014
                                 + 1.19444 w_k1_AXP_s014
                                 + 1.07663 w_k1_DIS_s014
                                 + 1.12819 w_k1_FDX_s014
                                 + 0.713225 w_k1_IBM_s014
                                 + 1.13365 w_k1_INTC_s014
                                 + 0.914462 w_k1_KO_s014
                                 + 0.738382 w_k1_MAT_s014
                                 + 0.942542 w_k1_MSFT_s014
                                 + 1.05309 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.1633 w_k1_ADM_s014
                                 + 1.26785 w_k1_AXP_s014
                                 + 1.58029 w_k1_DIS_s014
                                 + 1.22501 w_k1_FDX_s014
                                 + 0.826876 w_k1_IBM_s014
                                 + 1.3284 w_k1_INTC_s014 + 1.22276 w_k1_KO_s014
                                 + 1.00004 w_k1_MAT_s014
                                 + 1.35795 w_k1_MSFT_s014
                                 + 1.25088 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.30719 w_k1_ADM_s015
                                 + 1.56498 w_k1_AXP_s015
                                 + 1.54393 w_k1_DIS_s015
                                 + 1.27123 w_k1_FDX_s015
                                 + 1.51273 w_k1_IBM_s015
                                 + 1.11665 w_k1_INTC_s015
                                 + 1.34953 w_k1_KO_s015 + 1.43883 w_k1_MAT_s015
                                 + 1.11687 w_k1_MSFT_s015
                                 + 1.18335 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.0232 w_k1_ADM_s015
                                 + 1.14515 w_k1_AXP_s015
                                 + 1.33188 w_k1_DIS_s015
                                 + 1.23843 w_k1_FDX_s015
                                 + 0.980118 w_k1_IBM_s015
                                 + 1.3315 w_k1_INTC_s015 + 1.13098 w_k1_KO_s015
                                 + 0.799596 w_k1_MAT_s015
                                 + 1.17136 w_k1_MSFT_s015
                                 + 1.07327 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.07243 w_k1_ADM_s015
                                 + 1.62673 w_k1_AXP_s015
                                 + 1.31502 w_k1_DIS_s015
                                 + 1.33302 w_k1_FDX_s015
                                 + 1.10491 w_k1_IBM_s015
                                 + 1.47476 w_k1_INTC_s015
                                 + 1.15298 w_k1_KO_s015 + 1.40618 w_k1_MAT_s015
                                 + 1.25848 w_k1_MSFT_s015
                                 + 1.28994 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.914259 w_k1_ADM_s015
                                 + 0.783595 w_k1_AXP_s015
                                 + 0.936041 w_k1_DIS_s015
                                 + 0.948632 w_k1_FDX_s015
                                 + 0.708509 w_k1_IBM_s015
                                 + 1.10826 w_k1_INTC_s015
                                 + 0.769065 w_k1_KO_s015
                                 + 0.524264 w_k1_MAT_s015
                                 + 0.993043 w_k1_MSFT_s015
                                 + 0.986137 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.17352 w_k1_ADM_s015
                                 + 1.22817 w_k1_AXP_s015
                                 + 1.05384 w_k1_DIS_s015
                                 + 1.09987 w_k1_FDX_s015
                                 + 1.03281 w_k1_IBM_s015
                                 + 1.20355 w_k1_INTC_s015
                                 + 0.926029 w_k1_KO_s015
                                 + 1.11596 w_k1_MAT_s015
                                 + 1.03925 w_k1_MSFT_s015
                                 + 1.04384 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.778645 w_k1_ADM_s015
                                 + 0.948858 w_k1_AXP_s015
                                 + 0.654224 w_k1_DIS_s015
                                 + 0.839531 w_k1_FDX_s015
                                 + 0.949798 w_k1_IBM_s015
                                 + 1.13524 w_k1_INTC_s015
                                 + 0.768456 w_k1_KO_s015
                                 + 0.807898 w_k1_MAT_s015
                                 + 0.88511 w_k1_MSFT_s015
                                 + 0.842117 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.13591 w_k1_ADM_s015
                                 + 1.90201 w_k1_AXP_s015
                                 + 1.80413 w_k1_DIS_s015
                                 + 1.25251 w_k1_FDX_s015
                                 + 1.78334 w_k1_IBM_s015
                                 + 1.14776 w_k1_INTC_s015
                                 + 1.40921 w_k1_KO_s015 + 2.01931 w_k1_MAT_s015
                                 + 1.02523 w_k1_MSFT_s015
                                 + 1.30407 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.903095 w_k1_ADM_s015
                                 + 0.905659 w_k1_AXP_s015 + 1.11 w_k1_DIS_s015
                                 + 1.00146 w_k1_FDX_s015
                                 + 1.24488 w_k1_IBM_s015
                                 + 0.980456 w_k1_INTC_s015
                                 + 1.20868 w_k1_KO_s015 + 1.2787 w_k1_MAT_s015
                                 + 0.955485 w_k1_MSFT_s015
                                 + 1.05337 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.929765 w_k1_ADM_s016
                                 + 0.814684 w_k1_AXP_s016
                                 + 0.898539 w_k1_DIS_s016
                                 + 0.785219 w_k1_FDX_s016
                                 + 0.896993 w_k1_IBM_s016
                                 + 0.796566 w_k1_INTC_s016
                                 + 0.910525 w_k1_KO_s016
                                 + 0.719312 w_k1_MAT_s016
                                 + 0.706985 w_k1_MSFT_s016
                                 + 0.937317 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.983899 w_k1_ADM_s016
                                 + 1.15122 w_k1_AXP_s016
                                 + 1.04224 w_k1_DIS_s016
                                 + 0.964606 w_k1_FDX_s016
                                 + 1.33342 w_k1_IBM_s016
                                 + 1.14624 w_k1_INTC_s016
                                 + 0.989552 w_k1_KO_s016
                                 + 1.26595 w_k1_MAT_s016
                                 + 1.02575 w_k1_MSFT_s016
                                 + 1.04699 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.840951 w_k1_ADM_s016
                                 + 0.774654 w_k1_AXP_s016
                                 + 0.753849 w_k1_DIS_s016
                                 + 0.867424 w_k1_FDX_s016
                                 + 0.629339 w_k1_IBM_s016
                                 + 0.902124 w_k1_INTC_s016
                                 + 0.668322 w_k1_KO_s016
                                 + 0.504968 w_k1_MAT_s016
                                 + 0.814609 w_k1_MSFT_s016
                                 + 0.849169 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.03177 w_k1_ADM_s016
                                 + 0.945082 w_k1_AXP_s016
                                 + 0.896697 w_k1_DIS_s016
                                 + 0.851255 w_k1_FDX_s016
                                 + 1.04934 w_k1_IBM_s016
                                 + 0.742869 w_k1_INTC_s016
                                 + 0.94161 w_k1_KO_s016
                                 + 0.771787 w_k1_MAT_s016
                                 + 0.715589 w_k1_MSFT_s016
                                 + 0.91275 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.12147 w_k1_ADM_s016
                                 + 1.39656 w_k1_AXP_s016
                                 + 1.04356 w_k1_DIS_s016
                                 + 1.12123 w_k1_FDX_s016
                                 + 0.858146 w_k1_IBM_s016
                                 + 1.10006 w_k1_INTC_s016
                                 + 0.873606 w_k1_KO_s016
                                 + 0.820006 w_k1_MAT_s016
                                 + 0.970257 w_k1_MSFT_s016
                                 + 0.951732 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.96699 w_k1_ADM_s016
                                 + 0.833686 w_k1_AXP_s016
                                 + 0.674908 w_k1_DIS_s016
                                 + 0.80256 w_k1_FDX_s016
                                 + 0.816438 w_k1_IBM_s016
                                 + 0.955767 w_k1_INTC_s016
                                 + 0.785393 w_k1_KO_s016
                                 + 0.768879 w_k1_MAT_s016
                                 + 0.830331 w_k1_MSFT_s016
                                 + 0.893214 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.43743 w_k1_ADM_s016
                                 + 1.65042 w_k1_AXP_s016
                                 + 1.76297 w_k1_DIS_s016
                                 + 1.32433 w_k1_FDX_s016
                                 + 1.47035 w_k1_IBM_s016
                                 + 1.07647 w_k1_INTC_s016
                                 + 1.41118 w_k1_KO_s016 + 1.61967 w_k1_MAT_s016
                                 + 1.14828 w_k1_MSFT_s016
                                 + 1.22896 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.19057 w_k1_ADM_s016
                                 + 1.54731 w_k1_AXP_s016
                                 + 1.09356 w_k1_DIS_s016
                                 + 1.12559 w_k1_FDX_s016
                                 + 1.12411 w_k1_IBM_s016
                                 + 1.02011 w_k1_INTC_s016
                                 + 1.09922 w_k1_KO_s016 + 1.17191 w_k1_MAT_s016
                                 + 1.00981 w_k1_MSFT_s016
                                 + 1.06181 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.602377 w_k1_ADM_s017
                                 + 0.650086 w_k1_AXP_s017
                                 + 0.649851 w_k1_DIS_s017
                                 + 0.764974 w_k1_FDX_s017
                                 + 0.948324 w_k1_IBM_s017
                                 + 0.739807 w_k1_INTC_s017
                                 + 0.918375 w_k1_KO_s017
                                 + 0.959928 w_k1_MAT_s017
                                 + 0.778167 w_k1_MSFT_s017
                                 + 0.968366 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.658188 w_k1_ADM_s017
                                 + 0.746373 w_k1_AXP_s017
                                 + 0.862388 w_k1_DIS_s017
                                 + 0.914151 w_k1_FDX_s017
                                 + 0.794047 w_k1_IBM_s017
                                 + 0.969378 w_k1_INTC_s017
                                 + 0.796191 w_k1_KO_s017
                                 + 0.738417 w_k1_MAT_s017
                                 + 0.843228 w_k1_MSFT_s017
                                 + 0.969238 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.07994 w_k1_ADM_s017
                                 + 1.16863 w_k1_AXP_s017
                                 + 1.20918 w_k1_DIS_s017
                                 + 1.18433 w_k1_FDX_s017
                                 + 0.987216 w_k1_IBM_s017
                                 + 1.24774 w_k1_INTC_s017
                                 + 0.983961 w_k1_KO_s017
                                 + 0.968568 w_k1_MAT_s017
                                 + 1.16056 w_k1_MSFT_s017
                                 + 1.13695 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.623478 w_k1_ADM_s017
                                 + 0.579384 w_k1_AXP_s017
                                 + 0.535555 w_k1_DIS_s017
                                 + 0.740279 w_k1_FDX_s017
                                 + 0.707561 w_k1_IBM_s017
                                 + 0.797369 w_k1_INTC_s017
                                 + 0.699598 w_k1_KO_s017
                                 + 0.581249 w_k1_MAT_s017
                                 + 0.684396 w_k1_MSFT_s017
                                 + 0.943087 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.994338 w_k1_ADM_s017
                                 + 1.10421 w_k1_AXP_s017
                                 + 0.818238 w_k1_DIS_s017 + 1.077 w_k1_FDX_s017
                                 + 0.924268 w_k1_IBM_s017
                                 + 0.870557 w_k1_INTC_s017
                                 + 0.871121 w_k1_KO_s017
                                 + 1.03439 w_k1_MAT_s017
                                 + 0.845709 w_k1_MSFT_s017
                                 + 1.03347 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.767991 w_k1_ADM_s017
                                 + 1.00905 w_k1_AXP_s017
                                 + 0.927748 w_k1_DIS_s017
                                 + 0.941024 w_k1_FDX_s017
                                 + 1.30899 w_k1_IBM_s017
                                 + 0.994983 w_k1_INTC_s017
                                 + 0.996187 w_k1_KO_s017
                                 + 1.14723 w_k1_MAT_s017
                                 + 0.941129 w_k1_MSFT_s017
                                 + 1.00845 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.694397 w_k1_ADM_s017
                                 + 0.810136 w_k1_AXP_s017
                                 + 0.804623 w_k1_DIS_s017
                                 + 0.847759 w_k1_FDX_s017
                                 + 1.0431 w_k1_IBM_s017
                                 + 0.800978 w_k1_INTC_s017
                                 + 1.02452 w_k1_KO_s017 + 1.0038 w_k1_MAT_s017
                                 + 0.839842 w_k1_MSFT_s017
                                 + 1.00516 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.880525 w_k1_ADM_s017
                                 + 1.55439 w_k1_AXP_s017
                                 + 1.47318 w_k1_DIS_s017
                                 + 1.36687 w_k1_FDX_s017
                                 + 1.54859 w_k1_IBM_s017
                                 + 1.22397 w_k1_INTC_s017
                                 + 1.33784 w_k1_KO_s017 + 1.5726 w_k1_MAT_s017
                                 + 1.15825 w_k1_MSFT_s017 + 1.3698 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.885703 w_k1_ADM_s100
                                 + 0.8075 w_k1_AXP_s100
                                 + 0.866446 w_k1_DIS_s100
                                 + 0.883522 w_k1_FDX_s100
                                 + 0.887365 w_k1_IBM_s100
                                 + 0.989355 w_k1_INTC_s100
                                 + 0.925379 w_k1_KO_s100
                                 + 0.903205 w_k1_MAT_s100
                                 + 1.02072 w_k1_MSFT_s100
                                 + 0.918357 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.84819 w_k1_ADM_s100
                                 + 0.823112 w_k1_AXP_s100
                                 + 0.656214 w_k1_DIS_s100
                                 + 0.889553 w_k1_FDX_s100
                                 + 0.602206 w_k1_IBM_s100
                                 + 1.02278 w_k1_INTC_s100
                                 + 0.708207 w_k1_KO_s100
                                 + 0.695069 w_k1_MAT_s100
                                 + 0.836107 w_k1_MSFT_s100
                                 + 0.835306 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.915076 w_k1_ADM_s100
                                 + 0.996179 w_k1_AXP_s100
                                 + 0.992456 w_k1_DIS_s100
                                 + 0.882264 w_k1_FDX_s100
                                 + 1.40057 w_k1_IBM_s100
                                 + 0.778201 w_k1_INTC_s100
                                 + 1.12052 w_k1_KO_s100 + 1.64193 w_k1_MAT_s100
                                 + 0.862387 w_k1_MSFT_s100
                                 + 1.0452 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.31001 w_k1_ADM_s100
                                 + 1.12661 w_k1_AXP_s100
                                 + 0.982402 w_k1_DIS_s100
                                 + 1.24793 w_k1_FDX_s100
                                 + 0.758084 w_k1_IBM_s100
                                 + 1.30218 w_k1_INTC_s100
                                 + 0.942683 w_k1_KO_s100
                                 + 0.734627 w_k1_MAT_s100
                                 + 1.15724 w_k1_MSFT_s100
                                 + 1.11242 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.612043 w_k1_ADM_s100
                                 + 0.65371 w_k1_AXP_s100
                                 + 0.539004 w_k1_DIS_s100
                                 + 0.666103 w_k1_FDX_s100
                                 + 0.778718 w_k1_IBM_s100
                                 + 0.897516 w_k1_INTC_s100
                                 + 0.679201 w_k1_KO_s100
                                 + 0.619156 w_k1_MAT_s100
                                 + 0.644185 w_k1_MSFT_s100
                                 + 0.698435 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.12998 w_k1_ADM_s100
                                 + 1.05452 w_k1_AXP_s100
                                 + 0.960196 w_k1_DIS_s100
                                 + 1.04503 w_k1_FDX_s100
                                 + 0.869718 w_k1_IBM_s100
                                 + 1.19211 w_k1_INTC_s100
                                 + 0.969317 w_k1_KO_s100
                                 + 0.74048 w_k1_MAT_s100
                                 + 1.03104 w_k1_MSFT_s100
                                 + 1.02281 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.30087 w_k1_ADM_s100
                                 + 1.36655 w_k1_AXP_s100
                                 + 1.36489 w_k1_DIS_s100
                                 + 1.29893 w_k1_FDX_s100 + 1.4979 w_k1_IBM_s100
                                 + 1.48041 w_k1_INTC_s100
                                 + 1.37895 w_k1_KO_s100 + 1.994 w_k1_MAT_s100
                                 + 1.36009 w_k1_MSFT_s100 + 1.2818 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.13137 w_k1_ADM_s100
                                 + 1.27015 w_k1_AXP_s100 + 1.5158 w_k1_DIS_s100
                                 + 1.35188 w_k1_FDX_s100
                                 + 1.02923 w_k1_IBM_s100
                                 + 1.16123 w_k1_INTC_s100 + 1.0595 w_k1_KO_s100
                                 + 1.47199 w_k1_MAT_s100
                                 + 1.31856 w_k1_MSFT_s100
                                 + 1.09895 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.999219 w_k1_ADM_s101
                                 + 0.783892 w_k1_AXP_s101
                                 + 0.932851 w_k1_DIS_s101
                                 + 0.780554 w_k1_FDX_s101
                                 + 1.5331 w_k1_IBM_s101
                                 + 1.19715 w_k1_INTC_s101
                                 + 1.27948 w_k1_KO_s101 + 1.18376 w_k1_MAT_s101
                                 + 0.915731 w_k1_MSFT_s101
                                 + 0.887243 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.09502 w_k1_ADM_s101
                                 + 1.03206 w_k1_AXP_s101
                                 + 1.26092 w_k1_DIS_s101
                                 + 1.15729 w_k1_FDX_s101
                                 + 0.827497 w_k1_IBM_s101
                                 + 1.23387 w_k1_INTC_s101
                                 + 1.09186 w_k1_KO_s101
                                 + 0.841967 w_k1_MAT_s101
                                 + 1.21927 w_k1_MSFT_s101
                                 + 0.904358 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.47563 w_k1_ADM_s101
                                 + 1.55428 w_k1_AXP_s101
                                 + 1.59064 w_k1_DIS_s101
                                 + 1.53508 w_k1_FDX_s101
                                 + 0.91576 w_k1_IBM_s101
                                 + 1.31796 w_k1_INTC_s101
                                 + 1.36342 w_k1_KO_s101 + 1.18896 w_k1_MAT_s101
                                 + 1.23534 w_k1_MSFT_s101
                                 + 1.14428 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.1314 w_k1_ADM_s101
                                 + 0.785464 w_k1_AXP_s101
                                 + 0.852109 w_k1_DIS_s101
                                 + 0.816995 w_k1_FDX_s101
                                 + 1.17611 w_k1_IBM_s101
                                 + 1.1907 w_k1_INTC_s101 + 1.13893 w_k1_KO_s101
                                 + 0.899118 w_k1_MAT_s101
                                 + 0.846587 w_k1_MSFT_s101
                                 + 1.00928 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.12405 w_k1_ADM_s101
                                 + 1.15645 w_k1_AXP_s101
                                 + 1.17963 w_k1_DIS_s101
                                 + 1.11972 w_k1_FDX_s101
                                 + 1.12846 w_k1_IBM_s101
                                 + 1.52678 w_k1_INTC_s101
                                 + 1.22908 w_k1_KO_s101 + 0.98538 w_k1_MAT_s101
                                 + 1.12063 w_k1_MSFT_s101
                                 + 1.05534 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.857823 w_k1_ADM_s101
                                 + 0.887747 w_k1_AXP_s101
                                 + 0.733388 w_k1_DIS_s101
                                 + 0.847656 w_k1_FDX_s101
                                 + 0.785268 w_k1_IBM_s101
                                 + 1.04301 w_k1_INTC_s101
                                 + 0.783594 w_k1_KO_s101
                                 + 0.702244 w_k1_MAT_s101
                                 + 0.872715 w_k1_MSFT_s101
                                 + 0.852281 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.1849 w_k1_ADM_s101
                                 + 1.14497 w_k1_AXP_s101
                                 + 1.08347 w_k1_DIS_s101
                                 + 1.19582 w_k1_FDX_s101
                                 + 0.579859 w_k1_IBM_s101
                                 + 1.16226 w_k1_INTC_s101
                                 + 0.922897 w_k1_KO_s101
                                 + 0.557964 w_k1_MAT_s101
                                 + 1.05816 w_k1_MSFT_s101
                                 + 0.896451 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.42855 w_k1_ADM_s101
                                 + 1.2461 w_k1_AXP_s101 + 1.23763 w_k1_DIS_s101
                                 + 0.962435 w_k1_FDX_s101
                                 + 1.70662 w_k1_IBM_s101
                                 + 1.14341 w_k1_INTC_s101
                                 + 1.60112 w_k1_KO_s101 + 1.51977 w_k1_MAT_s101
                                 + 1.19053 w_k1_MSFT_s101
                                 + 1.12805 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.18444 w_k1_ADM_s102
                                 + 1.11975 w_k1_AXP_s102
                                 + 1.56795 w_k1_DIS_s102
                                 + 1.26837 w_k1_FDX_s102
                                 + 1.13167 w_k1_IBM_s102
                                 + 0.986471 w_k1_INTC_s102
                                 + 1.36986 w_k1_KO_s102
                                 + 0.998335 w_k1_MAT_s102
                                 + 1.22594 w_k1_MSFT_s102
                                 + 1.14542 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.06794 w_k1_ADM_s102
                                 + 0.963005 w_k1_AXP_s102
                                 + 0.891457 w_k1_DIS_s102
                                 + 0.976462 w_k1_FDX_s102
                                 + 0.89105 w_k1_IBM_s102
                                 + 1.00673 w_k1_INTC_s102
                                 + 1.00134 w_k1_KO_s102 + 1.08194 w_k1_MAT_s102
                                 + 1.00246 w_k1_MSFT_s102
                                 + 1.04433 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.28467 w_k1_ADM_s102
                                 + 1.41589 w_k1_AXP_s102
                                 + 1.73041 w_k1_DIS_s102
                                 + 1.60795 w_k1_FDX_s102
                                 + 1.00684 w_k1_IBM_s102
                                 + 1.00982 w_k1_INTC_s102
                                 + 1.31478 w_k1_KO_s102 + 1.20269 w_k1_MAT_s102
                                 + 1.45445 w_k1_MSFT_s102
                                 + 1.29376 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.737636 w_k1_ADM_s102
                                 + 0.680204 w_k1_AXP_s102
                                 + 0.637948 w_k1_DIS_s102
                                 + 0.850396 w_k1_FDX_s102
                                 + 0.682756 w_k1_IBM_s102
                                 + 0.760834 w_k1_INTC_s102
                                 + 0.729261 w_k1_KO_s102
                                 + 0.837668 w_k1_MAT_s102
                                 + 0.777348 w_k1_MSFT_s102
                                 + 0.902455 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.17697 w_k1_ADM_s102
                                 + 0.945957 w_k1_AXP_s102
                                 + 1.24874 w_k1_DIS_s102
                                 + 1.28034 w_k1_FDX_s102
                                 + 0.718631 w_k1_IBM_s102
                                 + 1.08537 w_k1_INTC_s102
                                 + 0.956644 w_k1_KO_s102
                                 + 0.741402 w_k1_MAT_s102
                                 + 1.31498 w_k1_MSFT_s102
                                 + 1.20406 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.694516 w_k1_ADM_s102
                                 + 0.808149 w_k1_AXP_s102
                                 + 0.577085 w_k1_DIS_s102
                                 + 0.740173 w_k1_FDX_s102
                                 + 1.00726 w_k1_IBM_s102
                                 + 0.788651 w_k1_INTC_s102
                                 + 0.806759 w_k1_KO_s102
                                 + 1.20832 w_k1_MAT_s102
                                 + 0.696465 w_k1_MSFT_s102
                                 + 0.883717 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.673498 w_k1_ADM_s102
                                 + 0.911797 w_k1_AXP_s102
                                 + 0.875787 w_k1_DIS_s102
                                 + 0.796611 w_k1_FDX_s102
                                 + 1.52881 w_k1_IBM_s102
                                 + 0.816229 w_k1_INTC_s102
                                 + 1.1319 w_k1_KO_s102 + 1.51679 w_k1_MAT_s102
                                 + 0.894299 w_k1_MSFT_s102
                                 + 0.956527 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.967492 w_k1_ADM_s102
                                 + 1.32824 w_k1_AXP_s102
                                 + 1.08853 w_k1_DIS_s102 + 1.1416 w_k1_FDX_s102
                                 + 1.2812 w_k1_IBM_s102
                                 + 0.989695 w_k1_INTC_s102
                                 + 1.12303 w_k1_KO_s102 + 1.50131 w_k1_MAT_s102
                                 + 1.05454 w_k1_MSFT_s102
                                 + 1.07572 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.968433 w_k1_ADM_s103
                                 + 0.758118 w_k1_AXP_s103
                                 + 0.891114 w_k1_DIS_s103
                                 + 0.96409 w_k1_FDX_s103
                                 + 0.64137 w_k1_IBM_s103
                                 + 0.767231 w_k1_INTC_s103
                                 + 0.807869 w_k1_KO_s103
                                 + 0.716811 w_k1_MAT_s103
                                 + 1.02772 w_k1_MSFT_s103
                                 + 0.883922 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.834211 w_k1_ADM_s103
                                 + 0.85828 w_k1_AXP_s103 + 1.0451 w_k1_DIS_s103
                                 + 0.81021 w_k1_FDX_s103
                                 + 1.43855 w_k1_IBM_s103
                                 + 0.737543 w_k1_INTC_s103
                                 + 1.04419 w_k1_KO_s103 + 1.67357 w_k1_MAT_s103
                                 + 0.915232 w_k1_MSFT_s103
                                 + 0.958192 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.920549 w_k1_ADM_s103
                                 + 1.22243 w_k1_AXP_s103
                                 + 1.42526 w_k1_DIS_s103
                                 + 1.09072 w_k1_FDX_s103
                                 + 1.16142 w_k1_IBM_s103
                                 + 0.931203 w_k1_INTC_s103
                                 + 1.1065 w_k1_KO_s103 + 1.24889 w_k1_MAT_s103
                                 + 1.12991 w_k1_MSFT_s103
                                 + 1.09904 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.16023 w_k1_ADM_s103
                                 + 0.949171 w_k1_AXP_s103
                                 + 1.30071 w_k1_DIS_s103
                                 + 1.15113 w_k1_FDX_s103
                                 + 0.840115 w_k1_IBM_s103
                                 + 1.01648 w_k1_INTC_s103
                                 + 1.06873 w_k1_KO_s103
                                 + 0.964078 w_k1_MAT_s103
                                 + 1.2766 w_k1_MSFT_s103 + 1.11859 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.38808 w_k1_ADM_s103
                                 + 1.73127 w_k1_AXP_s103
                                 + 1.85927 w_k1_DIS_s103
                                 + 1.48258 w_k1_FDX_s103
                                 + 1.14045 w_k1_IBM_s103
                                 + 1.16233 w_k1_INTC_s103
                                 + 1.37334 w_k1_KO_s103 + 1.37459 w_k1_MAT_s103
                                 + 1.44709 w_k1_MSFT_s103 + 1.3946 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.925354 w_k1_ADM_s103
                                 + 0.947333 w_k1_AXP_s103
                                 + 0.86717 w_k1_DIS_s103
                                 + 0.929347 w_k1_FDX_s103
                                 + 0.940523 w_k1_IBM_s103
                                 + 0.916633 w_k1_INTC_s103
                                 + 0.859984 w_k1_KO_s103
                                 + 0.856763 w_k1_MAT_s103
                                 + 0.912575 w_k1_MSFT_s103
                                 + 0.988914 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.657669 w_k1_ADM_s103
                                 + 0.607469 w_k1_AXP_s103
                                 + 0.646851 w_k1_DIS_s103
                                 + 0.684275 w_k1_FDX_s103
                                 + 0.820804 w_k1_IBM_s103
                                 + 0.847368 w_k1_INTC_s103
                                 + 0.831283 w_k1_KO_s103
                                 + 0.759795 w_k1_MAT_s103
                                 + 0.82125 w_k1_MSFT_s103
                                 + 0.783528 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.910729 w_k1_ADM_s103
                                 + 1.59316 w_k1_AXP_s103
                                 + 1.09585 w_k1_DIS_s103
                                 + 1.17043 w_k1_FDX_s103
                                 + 1.18621 w_k1_IBM_s103
                                 + 1.10644 w_k1_INTC_s103
                                 + 1.11615 w_k1_KO_s103 + 1.42811 w_k1_MAT_s103
                                 + 1.07025 w_k1_MSFT_s103 + 1.0821 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.824655 w_k1_ADM_s104
                                 + 0.751995 w_k1_AXP_s104
                                 + 0.780469 w_k1_DIS_s104
                                 + 0.79678 w_k1_FDX_s104
                                 + 1.03243 w_k1_IBM_s104
                                 + 0.770504 w_k1_INTC_s104
                                 + 1.07895 w_k1_KO_s104
                                 + 0.981748 w_k1_MAT_s104
                                 + 0.852111 w_k1_MSFT_s104
                                 + 0.855682 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.8227 w_k1_ADM_s104
                                 + 0.941774 w_k1_AXP_s104
                                 + 0.664068 w_k1_DIS_s104
                                 + 0.857726 w_k1_FDX_s104
                                 + 0.819356 w_k1_IBM_s104
                                 + 0.944781 w_k1_INTC_s104
                                 + 0.823544 w_k1_KO_s104
                                 + 0.840598 w_k1_MAT_s104
                                 + 0.873769 w_k1_MSFT_s104
                                 + 0.941691 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.21317 w_k1_ADM_s104
                                 + 1.28814 w_k1_AXP_s104
                                 + 1.26055 w_k1_DIS_s104
                                 + 1.25959 w_k1_FDX_s104
                                 + 0.765335 w_k1_IBM_s104
                                 + 1.16057 w_k1_INTC_s104
                                 + 0.886286 w_k1_KO_s104
                                 + 0.620314 w_k1_MAT_s104
                                 + 1.22311 w_k1_MSFT_s104
                                 + 1.01394 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.848605 w_k1_ADM_s104
                                 + 0.900835 w_k1_AXP_s104
                                 + 1.12758 w_k1_DIS_s104
                                 + 0.951279 w_k1_FDX_s104
                                 + 0.936366 w_k1_IBM_s104
                                 + 0.932512 w_k1_INTC_s104
                                 + 1.1071 w_k1_KO_s104 + 1.0354 w_k1_MAT_s104
                                 + 1.10363 w_k1_MSFT_s104
                                 + 1.03721 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.19128 w_k1_ADM_s104
                                 + 1.1708 w_k1_AXP_s104 + 1.23994 w_k1_DIS_s104
                                 + 1.14762 w_k1_FDX_s104
                                 + 1.21706 w_k1_IBM_s104
                                 + 1.13251 w_k1_INTC_s104
                                 + 1.12623 w_k1_KO_s104
                                 + 0.971213 w_k1_MAT_s104
                                 + 1.1215 w_k1_MSFT_s104 + 1.14581 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.04271 w_k1_ADM_s104
                                 + 1.10469 w_k1_AXP_s104
                                 + 0.763653 w_k1_DIS_s104
                                 + 0.878955 w_k1_FDX_s104
                                 + 1.18108 w_k1_IBM_s104
                                 + 0.817952 w_k1_INTC_s104
                                 + 0.939581 w_k1_KO_s104
                                 + 0.836396 w_k1_MAT_s104
                                 + 0.813919 w_k1_MSFT_s104
                                 + 1.00457 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.937992 w_k1_ADM_s104
                                 + 1.03069 w_k1_AXP_s104
                                 + 1.15659 w_k1_DIS_s104
                                 + 0.944833 w_k1_FDX_s104
                                 + 1.56383 w_k1_IBM_s104
                                 + 0.843681 w_k1_INTC_s104
                                 + 1.25505 w_k1_KO_s104 + 1.6307 w_k1_MAT_s104
                                 + 0.916863 w_k1_MSFT_s104
                                 + 1.08005 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.918234 w_k1_ADM_s104
                                 + 0.758867 w_k1_AXP_s104
                                 + 0.748534 w_k1_DIS_s104
                                 + 0.89599 w_k1_FDX_s104
                                 + 0.602412 w_k1_IBM_s104
                                 + 0.808991 w_k1_INTC_s104
                                 + 0.812587 w_k1_KO_s104
                                 + 0.671899 w_k1_MAT_s104
                                 + 0.838984 w_k1_MSFT_s104
                                 + 0.935099 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.06253 w_k1_ADM_s105
                                 + 1.58288 w_k1_AXP_s105
                                 + 1.45467 w_k1_DIS_s105 + 1.3636 w_k1_FDX_s105
                                 + 0.843755 w_k1_IBM_s105
                                 + 1.14311 w_k1_INTC_s105
                                 + 1.01796 w_k1_KO_s105 + 1.46275 w_k1_MAT_s105
                                 + 1.26982 w_k1_MSFT_s105
                                 + 1.09935 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.75784 w_k1_ADM_s105
                                 + 1.30451 w_k1_AXP_s105
                                 + 1.10239 w_k1_DIS_s105
                                 + 0.853098 w_k1_FDX_s105
                                 + 1.63942 w_k1_IBM_s105
                                 + 0.966733 w_k1_INTC_s105
                                 + 1.46844 w_k1_KO_s105 + 2.12922 w_k1_MAT_s105
                                 + 1.01267 w_k1_MSFT_s105 + 1.0193 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.757111 w_k1_ADM_s105
                                 + 0.735549 w_k1_AXP_s105
                                 + 0.814159 w_k1_DIS_s105
                                 + 0.823781 w_k1_FDX_s105
                                 + 0.810172 w_k1_IBM_s105
                                 + 0.644855 w_k1_INTC_s105
                                 + 0.748047 w_k1_KO_s105 + 1.0235 w_k1_MAT_s105
                                 + 0.861269 w_k1_MSFT_s105
                                 + 0.838858 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.748906 w_k1_ADM_s105
                                 + 1.12127 w_k1_AXP_s105
                                 + 1.10918 w_k1_DIS_s105
                                 + 0.967276 w_k1_FDX_s105
                                 + 1.00893 w_k1_IBM_s105
                                 + 0.914922 w_k1_INTC_s105
                                 + 1.0512 w_k1_KO_s105 + 1.38299 w_k1_MAT_s105
                                 + 1.01088 w_k1_MSFT_s105
                                 + 1.01617 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.08525 w_k1_ADM_s105
                                 + 1.57977 w_k1_AXP_s105
                                 + 1.43794 w_k1_DIS_s105
                                 + 1.16637 w_k1_FDX_s105
                                 + 1.39406 w_k1_IBM_s105
                                 + 1.08767 w_k1_INTC_s105
                                 + 1.29301 w_k1_KO_s105 + 1.67073 w_k1_MAT_s105
                                 + 1.1428 w_k1_MSFT_s105 + 1.02686 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.820384 w_k1_ADM_s105
                                 + 0.917013 w_k1_AXP_s105
                                 + 1.11523 w_k1_DIS_s105
                                 + 0.868947 w_k1_FDX_s105
                                 + 1.12868 w_k1_IBM_s105
                                 + 0.809224 w_k1_INTC_s105
                                 + 1.14244 w_k1_KO_s105 + 1.38257 w_k1_MAT_s105
                                 + 0.957395 w_k1_MSFT_s105
                                 + 0.919923 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.14088 w_k1_ADM_s105
                                 + 1.67362 w_k1_AXP_s105 + 2.0111 w_k1_DIS_s105
                                 + 1.44324 w_k1_FDX_s105
                                 + 1.16237 w_k1_IBM_s105
                                 + 1.18132 w_k1_INTC_s105
                                 + 1.35766 w_k1_KO_s105 + 1.71193 w_k1_MAT_s105
                                 + 1.35824 w_k1_MSFT_s105
                                 + 1.29669 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.895205 w_k1_ADM_s105
                                 + 1.05557 w_k1_AXP_s105
                                 + 1.06377 w_k1_DIS_s105
                                 + 1.02061 w_k1_FDX_s105
                                 + 0.746859 w_k1_IBM_s105
                                 + 0.886005 w_k1_INTC_s105
                                 + 1.03621 w_k1_KO_s105 + 1.06703 w_k1_MAT_s105
                                 + 1.06095 w_k1_MSFT_s105
                                 + 0.843643 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.756154 w_k1_ADM_s106
                                 + 1.00395 w_k1_AXP_s106
                                 + 0.831308 w_k1_DIS_s106
                                 + 0.991295 w_k1_FDX_s106
                                 + 0.79669 w_k1_IBM_s106
                                 + 0.929834 w_k1_INTC_s106
                                 + 0.751312 w_k1_KO_s106
                                 + 0.698026 w_k1_MAT_s106
                                 + 0.898199 w_k1_MSFT_s106
                                 + 0.834186 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.694398 w_k1_ADM_s106
                                 + 0.736001 w_k1_AXP_s106
                                 + 0.738564 w_k1_DIS_s106
                                 + 0.730305 w_k1_FDX_s106
                                 + 0.731055 w_k1_IBM_s106
                                 + 0.896894 w_k1_INTC_s106
                                 + 0.758122 w_k1_KO_s106
                                 + 0.517011 w_k1_MAT_s106
                                 + 0.891407 w_k1_MSFT_s106
                                 + 0.793903 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.932483 w_k1_ADM_s106
                                 + 1.06723 w_k1_AXP_s106
                                 + 0.981812 w_k1_DIS_s106
                                 + 1.03389 w_k1_FDX_s106
                                 + 0.993849 w_k1_IBM_s106
                                 + 1.07916 w_k1_INTC_s106
                                 + 1.02975 w_k1_KO_s106
                                 + 0.889041 w_k1_MAT_s106
                                 + 1.11487 w_k1_MSFT_s106
                                 + 1.01232 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.820103 w_k1_ADM_s106
                                 + 0.916828 w_k1_AXP_s106
                                 + 0.82552 w_k1_DIS_s106
                                 + 0.864104 w_k1_FDX_s106
                                 + 1.06391 w_k1_IBM_s106
                                 + 1.02533 w_k1_INTC_s106
                                 + 1.03507 w_k1_KO_s106
                                 + 0.998984 w_k1_MAT_s106
                                 + 1.10662 w_k1_MSFT_s106
                                 + 0.902127 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.10163 w_k1_ADM_s106
                                 + 0.995213 w_k1_AXP_s106
                                 + 1.44397 w_k1_DIS_s106
                                 + 1.21546 w_k1_FDX_s106
                                 + 1.10003 w_k1_IBM_s106
                                 + 1.22852 w_k1_INTC_s106
                                 + 1.21615 w_k1_KO_s106
                                 + 0.637263 w_k1_MAT_s106
                                 + 1.35779 w_k1_MSFT_s106
                                 + 1.06537 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.05387 w_k1_ADM_s106
                                 + 1.59111 w_k1_AXP_s106
                                 + 1.11537 w_k1_DIS_s106
                                 + 1.31611 w_k1_FDX_s106
                                 + 0.961035 w_k1_IBM_s106
                                 + 1.27119 w_k1_INTC_s106
                                 + 0.921593 w_k1_KO_s106
                                 + 1.05147 w_k1_MAT_s106
                                 + 1.23104 w_k1_MSFT_s106
                                 + 1.08782 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.13039 w_k1_ADM_s106
                                 + 1.0703 w_k1_AXP_s106
                                 + 0.979066 w_k1_DIS_s106
                                 + 1.16408 w_k1_FDX_s106
                                 + 0.688152 w_k1_IBM_s106
                                 + 1.24733 w_k1_INTC_s106
                                 + 0.909665 w_k1_KO_s106
                                 + 0.807503 w_k1_MAT_s106
                                 + 1.3099 w_k1_MSFT_s106 + 1.05819 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.874454 w_k1_ADM_s106
                                 + 1.28345 w_k1_AXP_s106
                                 + 1.16278 w_k1_DIS_s106
                                 + 1.12032 w_k1_FDX_s106
                                 + 1.46347 w_k1_IBM_s106
                                 + 1.08509 w_k1_INTC_s106
                                 + 1.27087 w_k1_KO_s106 + 1.59431 w_k1_MAT_s106
                                 + 1.26261 w_k1_MSFT_s106
                                 + 1.05121 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.12303 w_k1_ADM_s107
                                 + 0.926738 w_k1_AXP_s107
                                 + 1.25928 w_k1_DIS_s107
                                 + 1.06972 w_k1_FDX_s107
                                 + 1.12972 w_k1_IBM_s107
                                 + 0.909281 w_k1_INTC_s107
                                 + 1.05211 w_k1_KO_s107
                                 + 0.976588 w_k1_MAT_s107
                                 + 0.9977 w_k1_MSFT_s107 + 1.06668 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.910303 w_k1_ADM_s107
                                 + 0.666664 w_k1_AXP_s107
                                 + 0.81364 w_k1_DIS_s107
                                 + 0.888826 w_k1_FDX_s107
                                 + 0.782643 w_k1_IBM_s107
                                 + 0.858972 w_k1_INTC_s107
                                 + 0.785507 w_k1_KO_s107
                                 + 0.733664 w_k1_MAT_s107
                                 + 0.87845 w_k1_MSFT_s107
                                 + 0.834083 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.40496 w_k1_ADM_s107
                                 + 1.48314 w_k1_AXP_s107
                                 + 1.76821 w_k1_DIS_s107
                                 + 1.39427 w_k1_FDX_s107
                                 + 1.25655 w_k1_IBM_s107
                                 + 1.25751 w_k1_INTC_s107
                                 + 1.25866 w_k1_KO_s107 + 1.41538 w_k1_MAT_s107
                                 + 1.35578 w_k1_MSFT_s107
                                 + 1.27017 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.815632 w_k1_ADM_s107
                                 + 0.699211 w_k1_AXP_s107
                                 + 0.884918 w_k1_DIS_s107
                                 + 0.848111 w_k1_FDX_s107
                                 + 1.11706 w_k1_IBM_s107
                                 + 0.845446 w_k1_INTC_s107
                                 + 0.884248 w_k1_KO_s107
                                 + 1.05411 w_k1_MAT_s107
                                 + 0.922341 w_k1_MSFT_s107
                                 + 0.880943 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.6731 w_k1_ADM_s107
                                 + 0.505444 w_k1_AXP_s107
                                 + 0.56833 w_k1_DIS_s107 + 0.6747 w_k1_FDX_s107
                                 + 0.575887 w_k1_IBM_s107
                                 + 0.605562 w_k1_INTC_s107
                                 + 0.622634 w_k1_KO_s107
                                 + 0.610544 w_k1_MAT_s107
                                 + 0.658322 w_k1_MSFT_s107
                                 + 0.772999 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.16107 w_k1_ADM_s107
                                 + 1.01581 w_k1_AXP_s107
                                 + 1.24215 w_k1_DIS_s107
                                 + 1.16757 w_k1_FDX_s107
                                 + 0.877672 w_k1_IBM_s107
                                 + 1.03477 w_k1_INTC_s107
                                 + 0.906324 w_k1_KO_s107
                                 + 0.989494 w_k1_MAT_s107
                                 + 1.07753 w_k1_MSFT_s107
                                 + 1.04055 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.13888 w_k1_ADM_s107
                                 + 1.38929 w_k1_AXP_s107
                                 + 1.15971 w_k1_DIS_s107
                                 + 0.980508 w_k1_FDX_s107
                                 + 1.60767 w_k1_IBM_s107
                                 + 1.01074 w_k1_INTC_s107
                                 + 1.21093 w_k1_KO_s107 + 1.84911 w_k1_MAT_s107
                                 + 0.989718 w_k1_MSFT_s107
                                 + 1.11832 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.852726 w_k1_ADM_s107
                                 + 0.817962 w_k1_AXP_s107
                                 + 0.59311 w_k1_DIS_s107
                                 + 0.733062 w_k1_FDX_s107 + 1.062 w_k1_IBM_s107
                                 + 0.746554 w_k1_INTC_s107
                                 + 0.795317 w_k1_KO_s107
                                 + 1.03818 w_k1_MAT_s107
                                 + 0.645809 w_k1_MSFT_s107
                                 + 0.822902 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.819134 w_k1_ADM_s110
                                 + 1.00936 w_k1_AXP_s110
                                 + 0.912586 w_k1_DIS_s110
                                 + 0.982531 w_k1_FDX_s110
                                 + 1.09699 w_k1_IBM_s110
                                 + 1.2135 w_k1_INTC_s110 + 1.0573 w_k1_KO_s110
                                 + 1.21616 w_k1_MAT_s110
                                 + 0.978041 w_k1_MSFT_s110
                                 + 1.06374 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.18762 w_k1_ADM_s110
                                 + 0.965456 w_k1_AXP_s110
                                 + 1.13111 w_k1_DIS_s110
                                 + 1.16707 w_k1_FDX_s110
                                 + 0.815376 w_k1_IBM_s110
                                 + 1.0347 w_k1_INTC_s110
                                 + 0.934526 w_k1_KO_s110
                                 + 1.13895 w_k1_MAT_s110
                                 + 1.04982 w_k1_MSFT_s110
                                 + 1.23785 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.822047 w_k1_ADM_s110
                                 + 0.889213 w_k1_AXP_s110
                                 + 0.752801 w_k1_DIS_s110
                                 + 0.805321 w_k1_FDX_s110
                                 + 0.916945 w_k1_IBM_s110
                                 + 0.687408 w_k1_INTC_s110
                                 + 0.817362 w_k1_KO_s110
                                 + 1.21991 w_k1_MAT_s110
                                 + 0.785668 w_k1_MSFT_s110
                                 + 0.978793 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.955895 w_k1_ADM_s110
                                 + 1.2472 w_k1_AXP_s110 + 1.25111 w_k1_DIS_s110
                                 + 1.02492 w_k1_FDX_s110
                                 + 1.80767 w_k1_IBM_s110
                                 + 1.15212 w_k1_INTC_s110
                                 + 1.41437 w_k1_KO_s110 + 2.10765 w_k1_MAT_s110
                                 + 1.08875 w_k1_MSFT_s110
                                 + 1.15462 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.04141 w_k1_ADM_s110
                                 + 1.18492 w_k1_AXP_s110
                                 + 1.14076 w_k1_DIS_s110
                                 + 1.26289 w_k1_FDX_s110
                                 + 1.04223 w_k1_IBM_s110
                                 + 1.27976 w_k1_INTC_s110
                                 + 0.922836 w_k1_KO_s110
                                 + 1.09323 w_k1_MAT_s110
                                 + 1.12392 w_k1_MSFT_s110
                                 + 1.17978 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.20983 w_k1_ADM_s110
                                 + 1.50684 w_k1_AXP_s110
                                 + 1.52081 w_k1_DIS_s110
                                 + 1.34936 w_k1_FDX_s110
                                 + 1.40063 w_k1_IBM_s110
                                 + 1.20962 w_k1_INTC_s110
                                 + 1.28557 w_k1_KO_s110 + 1.85497 w_k1_MAT_s110
                                 + 1.11306 w_k1_MSFT_s110
                                 + 1.38979 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.518178 w_k1_ADM_s110
                                 + 0.61968 w_k1_AXP_s110
                                 + 0.531314 w_k1_DIS_s110
                                 + 0.691145 w_k1_FDX_s110
                                 + 0.694623 w_k1_IBM_s110
                                 + 0.841525 w_k1_INTC_s110
                                 + 0.599304 w_k1_KO_s110
                                 + 0.838353 w_k1_MAT_s110
                                 + 0.721316 w_k1_MSFT_s110
                                 + 0.835482 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.964181 w_k1_ADM_s110
                                 + 0.792315 w_k1_AXP_s110 + w_k1_DIS_s110
                                 + 0.980952 w_k1_FDX_s110
                                 + 0.698714 w_k1_IBM_s110
                                 + 1.04912 w_k1_INTC_s110
                                 + 0.89375 w_k1_KO_s110
                                 + 0.719338 w_k1_MAT_s110
                                 + 0.923923 w_k1_MSFT_s110
                                 + 1.04768 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.787316 w_k1_ADM_s111
                                 + 0.936797 w_k1_AXP_s111
                                 + 0.797679 w_k1_DIS_s111
                                 + 0.854422 w_k1_FDX_s111
                                 + 1.23452 w_k1_IBM_s111
                                 + 0.911667 w_k1_INTC_s111
                                 + 0.929062 w_k1_KO_s111
                                 + 1.34454 w_k1_MAT_s111
                                 + 0.769824 w_k1_MSFT_s111
                                 + 0.898996 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.824854 w_k1_ADM_s111
                                 + 0.76946 w_k1_AXP_s111
                                 + 0.667326 w_k1_DIS_s111
                                 + 0.857337 w_k1_FDX_s111
                                 + 0.666974 w_k1_IBM_s111
                                 + 0.880065 w_k1_INTC_s111
                                 + 0.692616 w_k1_KO_s111
                                 + 0.900078 w_k1_MAT_s111
                                 + 0.90964 w_k1_MSFT_s111
                                 + 0.783867 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.924229 w_k1_ADM_s111
                                 + 1.29022 w_k1_AXP_s111
                                 + 0.871506 w_k1_DIS_s111
                                 + 0.989794 w_k1_FDX_s111
                                 + 1.05973 w_k1_IBM_s111
                                 + 0.866809 w_k1_INTC_s111
                                 + 0.881855 w_k1_KO_s111
                                 + 1.56314 w_k1_MAT_s111
                                 + 1.00473 w_k1_MSFT_s111
                                 + 1.00768 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.31205 w_k1_ADM_s111
                                 + 1.43011 w_k1_AXP_s111
                                 + 1.18963 w_k1_DIS_s111
                                 + 1.24146 w_k1_FDX_s111
                                 + 0.849765 w_k1_IBM_s111
                                 + 1.09965 w_k1_INTC_s111
                                 + 0.999094 w_k1_KO_s111
                                 + 1.77471 w_k1_MAT_s111
                                 + 1.26047 w_k1_MSFT_s111
                                 + 0.972418 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.91458 w_k1_ADM_s111
                                 + 0.815311 w_k1_AXP_s111
                                 + 1.02022 w_k1_DIS_s111
                                 + 1.00011 w_k1_FDX_s111
                                 + 0.593647 w_k1_IBM_s111
                                 + 0.783781 w_k1_INTC_s111
                                 + 0.749233 w_k1_KO_s111
                                 + 0.867352 w_k1_MAT_s111
                                 + 1.06551 w_k1_MSFT_s111 + 0.922 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.964713 w_k1_ADM_s111
                                 + 1.14557 w_k1_AXP_s111
                                 + 1.22618 w_k1_DIS_s111
                                 + 1.03585 w_k1_FDX_s111
                                 + 1.33393 w_k1_IBM_s111
                                 + 1.02671 w_k1_INTC_s111
                                 + 0.978009 w_k1_KO_s111
                                 + 1.50904 w_k1_MAT_s111
                                 + 1.20407 w_k1_MSFT_s111
                                 + 1.02019 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.05267 w_k1_ADM_s111
                                 + 1.61193 w_k1_AXP_s111
                                 + 1.67969 w_k1_DIS_s111
                                 + 1.37062 w_k1_FDX_s111
                                 + 1.39211 w_k1_IBM_s111
                                 + 1.33083 w_k1_INTC_s111
                                 + 1.36276 w_k1_KO_s111 + 1.68878 w_k1_MAT_s111
                                 + 1.43875 w_k1_MSFT_s111
                                 + 1.06269 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.1218 w_k1_ADM_s111
                                 + 0.978858 w_k1_AXP_s111
                                 + 1.01158 w_k1_DIS_s111
                                 + 1.12753 w_k1_FDX_s111
                                 + 0.86257 w_k1_IBM_s111
                                 + 1.11047 w_k1_INTC_s111
                                 + 0.847868 w_k1_KO_s111
                                 + 0.956429 w_k1_MAT_s111
                                 + 1.09257 w_k1_MSFT_s111
                                 + 0.982084 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.966931 w_k1_ADM_s112
                                 + 1.04543 w_k1_AXP_s112
                                 + 1.17995 w_k1_DIS_s112
                                 + 1.10058 w_k1_FDX_s112
                                 + 0.935928 w_k1_IBM_s112
                                 + 1.16647 w_k1_INTC_s112
                                 + 0.906269 w_k1_KO_s112
                                 + 0.944932 w_k1_MAT_s112
                                 + 1.03268 w_k1_MSFT_s112
                                 + 0.908796 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.859768 w_k1_ADM_s112
                                 + 0.717128 w_k1_AXP_s112
                                 + 0.810201 w_k1_DIS_s112
                                 + 0.891226 w_k1_FDX_s112
                                 + 0.685489 w_k1_IBM_s112
                                 + 0.957294 w_k1_INTC_s112
                                 + 0.682107 w_k1_KO_s112
                                 + 0.444855 w_k1_MAT_s112
                                 + 0.828551 w_k1_MSFT_s112
                                 + 0.866121 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.897351 w_k1_ADM_s112
                                 + 0.864867 w_k1_AXP_s112
                                 + 0.867292 w_k1_DIS_s112
                                 + 0.924551 w_k1_FDX_s112
                                 + 1.01602 w_k1_IBM_s112
                                 + 1.08159 w_k1_INTC_s112
                                 + 0.940018 w_k1_KO_s112
                                 + 0.795379 w_k1_MAT_s112
                                 + 0.910767 w_k1_MSFT_s112
                                 + 1.01092 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.01143 w_k1_ADM_s112
                                 + 1.23977 w_k1_AXP_s112
                                 + 1.03844 w_k1_DIS_s112
                                 + 0.988625 w_k1_FDX_s112
                                 + 1.26131 w_k1_IBM_s112
                                 + 1.14941 w_k1_INTC_s112
                                 + 1.07138 w_k1_KO_s112 + 1.46105 w_k1_MAT_s112
                                 + 0.94531 w_k1_MSFT_s112
                                 + 1.11767 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.653398 w_k1_ADM_s112
                                 + 0.688639 w_k1_AXP_s112
                                 + 0.77324 w_k1_DIS_s112
                                 + 0.724363 w_k1_FDX_s112
                                 + 0.891757 w_k1_IBM_s112
                                 + 0.811482 w_k1_INTC_s112
                                 + 0.796356 w_k1_KO_s112
                                 + 0.687288 w_k1_MAT_s112
                                 + 0.751102 w_k1_MSFT_s112
                                 + 0.86574 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.2166 w_k1_ADM_s112
                                 + 1.6024 w_k1_AXP_s112 + 1.73034 w_k1_DIS_s112
                                 + 1.42751 w_k1_FDX_s112
                                 + 1.59591 w_k1_IBM_s112
                                 + 1.64081 w_k1_INTC_s112
                                 + 1.38835 w_k1_KO_s112 + 1.23658 w_k1_MAT_s112
                                 + 1.39395 w_k1_MSFT_s112
                                 + 1.52652 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.07723 w_k1_ADM_s112
                                 + 1.30458 w_k1_AXP_s112
                                 + 1.38797 w_k1_DIS_s112
                                 + 1.30409 w_k1_FDX_s112
                                 + 1.03787 w_k1_IBM_s112
                                 + 1.26964 w_k1_INTC_s112
                                 + 0.969059 w_k1_KO_s112
                                 + 1.21608 w_k1_MAT_s112
                                 + 1.27869 w_k1_MSFT_s112 + 1.1023 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.02124 w_k1_ADM_s112
                                 + 1.24262 w_k1_AXP_s112
                                 + 1.21675 w_k1_DIS_s112
                                 + 0.989399 w_k1_FDX_s112
                                 + 1.65087 w_k1_IBM_s112
                                 + 1.25663 w_k1_INTC_s112
                                 + 1.24027 w_k1_KO_s112 + 1.0321 w_k1_MAT_s112
                                 + 0.951746 w_k1_MSFT_s112
                                 + 1.10793 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.830494 w_k1_ADM_s113
                                 + 0.917852 w_k1_AXP_s113
                                 + 1.02789 w_k1_DIS_s113
                                 + 0.84532 w_k1_FDX_s113
                                 + 1.60851 w_k1_IBM_s113
                                 + 0.788368 w_k1_INTC_s113
                                 + 1.18098 w_k1_KO_s113 + 1.47809 w_k1_MAT_s113
                                 + 0.902572 w_k1_MSFT_s113
                                 + 1.11998 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.37542 w_k1_ADM_s113
                                 + 1.49388 w_k1_AXP_s113
                                 + 1.05895 w_k1_DIS_s113
                                 + 1.17258 w_k1_FDX_s113
                                 + 0.973955 w_k1_IBM_s113
                                 + 1.16782 w_k1_INTC_s113
                                 + 1.10145 w_k1_KO_s113 + 1.04224 w_k1_MAT_s113
                                 + 1.12112 w_k1_MSFT_s113
                                 + 0.91867 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.828011 w_k1_ADM_s113
                                 + 0.805828 w_k1_AXP_s113
                                 + 0.772786 w_k1_DIS_s113
                                 + 0.762013 w_k1_FDX_s113
                                 + 1.05543 w_k1_IBM_s113
                                 + 0.890894 w_k1_INTC_s113
                                 + 0.923354 w_k1_KO_s113
                                 + 0.727648 w_k1_MAT_s113
                                 + 0.786646 w_k1_MSFT_s113
                                 + 0.87309 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.13334 w_k1_ADM_s113
                                 + 0.908017 w_k1_AXP_s113
                                 + 1.41369 w_k1_DIS_s113 + 1.0536 w_k1_FDX_s113
                                 + 0.894441 w_k1_IBM_s113
                                 + 0.910729 w_k1_INTC_s113
                                 + 1.07434 w_k1_KO_s113
                                 + 0.698134 w_k1_MAT_s113
                                 + 1.09467 w_k1_MSFT_s113
                                 + 0.981991 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.13566 w_k1_ADM_s113
                                 + 1.43078 w_k1_AXP_s113 + 1.6773 w_k1_DIS_s113
                                 + 1.14773 w_k1_FDX_s113
                                 + 1.54417 w_k1_IBM_s113
                                 + 1.10335 w_k1_INTC_s113
                                 + 1.47411 w_k1_KO_s113 + 1.39144 w_k1_MAT_s113
                                 + 1.19522 w_k1_MSFT_s113
                                 + 1.13271 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.959846 w_k1_ADM_s113
                                 + 0.723566 w_k1_AXP_s113
                                 + 0.652609 w_k1_DIS_s113
                                 + 0.729996 w_k1_FDX_s113
                                 + 0.649546 w_k1_IBM_s113
                                 + 0.879265 w_k1_INTC_s113
                                 + 0.672174 w_k1_KO_s113
                                 + 0.44751 w_k1_MAT_s113
                                 + 0.875709 w_k1_MSFT_s113
                                 + 0.83272 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.00569 w_k1_ADM_s113
                                 + 0.927446 w_k1_AXP_s113
                                 + 1.05278 w_k1_DIS_s113
                                 + 1.10617 w_k1_FDX_s113
                                 + 0.702466 w_k1_IBM_s113
                                 + 1.17185 w_k1_INTC_s113
                                 + 1.00531 w_k1_KO_s113
                                 + 0.719925 w_k1_MAT_s113
                                 + 1.09286 w_k1_MSFT_s113
                                 + 0.984609 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.37441 w_k1_ADM_s113
                                 + 1.18979 w_k1_AXP_s113
                                 + 1.32474 w_k1_DIS_s113
                                 + 1.08923 w_k1_FDX_s113
                                 + 1.03117 w_k1_IBM_s113
                                 + 1.03054 w_k1_INTC_s113
                                 + 1.22007 w_k1_KO_s113 + 1.09484 w_k1_MAT_s113
                                 + 1.22597 w_k1_MSFT_s113
                                 + 1.14981 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.968169 w_k1_ADM_s114
                                 + 1.21141 w_k1_AXP_s114
                                 + 1.40414 w_k1_DIS_s114 + 1.265 w_k1_FDX_s114
                                 + 1.12112 w_k1_IBM_s114
                                 + 1.39944 w_k1_INTC_s114 + 1.3256 w_k1_KO_s114
                                 + 1.04676 w_k1_MAT_s114
                                 + 1.40023 w_k1_MSFT_s114
                                 + 1.28488 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.804291 w_k1_ADM_s114
                                 + 0.446986 w_k1_AXP_s114
                                 + 0.597221 w_k1_DIS_s114
                                 + 0.780791 w_k1_FDX_s114
                                 + 0.726116 w_k1_IBM_s114
                                 + 0.803917 w_k1_INTC_s114
                                 + 0.766131 w_k1_KO_s114
                                 + 0.621282 w_k1_MAT_s114
                                 + 0.904633 w_k1_MSFT_s114
                                 + 0.822423 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.789368 w_k1_ADM_s114
                                 + 0.71855 w_k1_AXP_s114
                                 + 0.783833 w_k1_DIS_s114
                                 + 0.880104 w_k1_FDX_s114
                                 + 0.814883 w_k1_IBM_s114
                                 + 0.99716 w_k1_INTC_s114
                                 + 0.793404 w_k1_KO_s114
                                 + 0.846974 w_k1_MAT_s114
                                 + 0.953408 w_k1_MSFT_s114
                                 + 0.919007 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.14994 w_k1_ADM_s114
                                 + 0.834288 w_k1_AXP_s114
                                 + 1.05208 w_k1_DIS_s114
                                 + 1.10293 w_k1_FDX_s114
                                 + 0.942018 w_k1_IBM_s114
                                 + 1.15958 w_k1_INTC_s114
                                 + 0.856784 w_k1_KO_s114
                                 + 0.671486 w_k1_MAT_s114
                                 + 0.927459 w_k1_MSFT_s114
                                 + 1.06423 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.34438 w_k1_ADM_s114
                                 + 1.24008 w_k1_AXP_s114
                                 + 0.960169 w_k1_DIS_s114
                                 + 1.03333 w_k1_FDX_s114
                                 + 1.53217 w_k1_IBM_s114
                                 + 0.935358 w_k1_INTC_s114
                                 + 1.21981 w_k1_KO_s114 + 1.47167 w_k1_MAT_s114
                                 + 1.12678 w_k1_MSFT_s114
                                 + 1.22437 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.21809 w_k1_ADM_s114
                                 + 0.85437 w_k1_AXP_s114
                                 + 0.936324 w_k1_DIS_s114
                                 + 1.11863 w_k1_FDX_s114
                                 + 0.597561 w_k1_IBM_s114
                                 + 1.09735 w_k1_INTC_s114
                                 + 0.729388 w_k1_KO_s114
                                 + 0.481488 w_k1_MAT_s114
                                 + 1.25004 w_k1_MSFT_s114
                                 + 0.983905 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.758427 w_k1_ADM_s114
                                 + 0.664649 w_k1_AXP_s114
                                 + 0.751374 w_k1_DIS_s114
                                 + 0.770819 w_k1_FDX_s114
                                 + 1.3232 w_k1_IBM_s114
                                 + 0.912369 w_k1_INTC_s114
                                 + 0.975887 w_k1_KO_s114
                                 + 0.983215 w_k1_MAT_s114
                                 + 0.861947 w_k1_MSFT_s114
                                 + 0.999842 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.806131 w_k1_ADM_s114
                                 + 0.944356 w_k1_AXP_s114
                                 + 1.00807 w_k1_DIS_s114
                                 + 1.00755 w_k1_FDX_s114
                                 + 0.855048 w_k1_IBM_s114
                                 + 1.28578 w_k1_INTC_s114
                                 + 1.00871 w_k1_KO_s114 + 1.00158 w_k1_MAT_s114
                                 + 1.16776 w_k1_MSFT_s114
                                 + 1.05783 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.11654 w_k1_ADM_s115
                                 + 1.45044 w_k1_AXP_s115
                                 + 1.21132 w_k1_DIS_s115
                                 + 1.35952 w_k1_FDX_s115
                                 + 0.953692 w_k1_IBM_s115
                                 + 1.22679 w_k1_INTC_s115
                                 + 0.955197 w_k1_KO_s115
                                 + 1.03938 w_k1_MAT_s115
                                 + 1.13128 w_k1_MSFT_s115
                                 + 1.10495 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.44091 w_k1_ADM_s115
                                 + 1.34883 w_k1_AXP_s115
                                 + 1.67339 w_k1_DIS_s115
                                 + 1.38993 w_k1_FDX_s115
                                 + 1.12586 w_k1_IBM_s115
                                 + 1.56787 w_k1_INTC_s115
                                 + 1.43922 w_k1_KO_s115
                                 + 0.992096 w_k1_MAT_s115
                                 + 1.41747 w_k1_MSFT_s115 + 1.2378 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.06362 w_k1_ADM_s115
                                 + 1.17088 w_k1_AXP_s115
                                 + 0.986712 w_k1_DIS_s115
                                 + 1.06597 w_k1_FDX_s115
                                 + 1.10781 w_k1_IBM_s115
                                 + 1.11258 w_k1_INTC_s115
                                 + 1.05633 w_k1_KO_s115 + 1.09482 w_k1_MAT_s115
                                 + 0.986748 w_k1_MSFT_s115
                                 + 1.00474 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.35178 w_k1_ADM_s115
                                 + 1.40687 w_k1_AXP_s115
                                 + 1.36639 w_k1_DIS_s115
                                 + 1.26569 w_k1_FDX_s115
                                 + 1.26856 w_k1_IBM_s115
                                 + 1.48836 w_k1_INTC_s115
                                 + 1.29248 w_k1_KO_s115 + 1.36026 w_k1_MAT_s115
                                 + 1.18093 w_k1_MSFT_s115
                                 + 1.15098 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.12156 w_k1_ADM_s115
                                 + 0.831047 w_k1_AXP_s115
                                 + 0.842776 w_k1_DIS_s115
                                 + 1.05125 w_k1_FDX_s115
                                 + 0.721787 w_k1_IBM_s115
                                 + 1.01952 w_k1_INTC_s115
                                 + 0.808282 w_k1_KO_s115
                                 + 0.580043 w_k1_MAT_s115
                                 + 0.969762 w_k1_MSFT_s115
                                 + 0.954619 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.807826 w_k1_ADM_s115
                                 + 0.742723 w_k1_AXP_s115
                                 + 0.843431 w_k1_DIS_s115
                                 + 0.823216 w_k1_FDX_s115
                                 + 0.894887 w_k1_IBM_s115
                                 + 0.894879 w_k1_INTC_s115
                                 + 0.872797 w_k1_KO_s115
                                 + 0.92342 w_k1_MAT_s115
                                 + 0.995724 w_k1_MSFT_s115
                                 + 0.991796 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.976996 w_k1_ADM_s115
                                 + 0.820222 w_k1_AXP_s115
                                 + 0.977064 w_k1_DIS_s115
                                 + 0.897778 w_k1_FDX_s115
                                 + 1.38849 w_k1_IBM_s115
                                 + 1.08966 w_k1_INTC_s115
                                 + 1.01511 w_k1_KO_s115 + 1.13336 w_k1_MAT_s115
                                 + 0.941127 w_k1_MSFT_s115
                                 + 1.01646 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.22785 w_k1_ADM_s115
                                 + 1.31498 w_k1_AXP_s115 + 1.4944 w_k1_DIS_s115
                                 + 1.23252 w_k1_FDX_s115
                                 + 1.68746 w_k1_IBM_s115
                                 + 1.18721 w_k1_INTC_s115
                                 + 1.28512 w_k1_KO_s115 + 1.76961 w_k1_MAT_s115
                                 + 1.19977 w_k1_MSFT_s115
                                 + 1.21032 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.583342 w_k1_ADM_s116
                                 + 0.674185 w_k1_AXP_s116
                                 + 0.597645 w_k1_DIS_s116
                                 + 0.720895 w_k1_FDX_s116
                                 + 0.691922 w_k1_IBM_s116
                                 + 0.762412 w_k1_INTC_s116
                                 + 0.723099 w_k1_KO_s116
                                 + 0.654256 w_k1_MAT_s116
                                 + 0.785628 w_k1_MSFT_s116
                                 + 0.790102 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.859196 w_k1_ADM_s116
                                 + 1.00737 w_k1_AXP_s116
                                 + 0.943118 w_k1_DIS_s116
                                 + 1.0186 w_k1_FDX_s116
                                 + 0.862032 w_k1_IBM_s116
                                 + 0.76951 w_k1_INTC_s116
                                 + 0.995784 w_k1_KO_s116
                                 + 0.874694 w_k1_MAT_s116
                                 + 0.932322 w_k1_MSFT_s116
                                 + 0.903574 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.767728 w_k1_ADM_s116
                                 + 1.04287 w_k1_AXP_s116
                                 + 1.13735 w_k1_DIS_s116
                                 + 1.11113 w_k1_FDX_s116
                                 + 0.794251 w_k1_IBM_s116
                                 + 1.09549 w_k1_INTC_s116
                                 + 0.975987 w_k1_KO_s116 + 1.1329 w_k1_MAT_s116
                                 + 1.14173 w_k1_MSFT_s116
                                 + 0.952891 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.714698 w_k1_ADM_s116
                                 + 0.863446 w_k1_AXP_s116
                                 + 0.773938 w_k1_DIS_s116
                                 + 0.898538 w_k1_FDX_s116
                                 + 0.739734 w_k1_IBM_s116
                                 + 0.757425 w_k1_INTC_s116
                                 + 0.823154 w_k1_KO_s116
                                 + 0.930142 w_k1_MAT_s116
                                 + 0.877109 w_k1_MSFT_s116
                                 + 0.870636 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.860209 w_k1_ADM_s116
                                 + 1.52707 w_k1_AXP_s116
                                 + 1.14722 w_k1_DIS_s116
                                 + 1.12187 w_k1_FDX_s116
                                 + 1.43339 w_k1_IBM_s116
                                 + 0.889001 w_k1_INTC_s116
                                 + 1.23542 w_k1_KO_s116 + 1.88657 w_k1_MAT_s116
                                 + 0.999637 w_k1_MSFT_s116
                                 + 1.07416 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.75958 w_k1_ADM_s116
                                 + 0.958812 w_k1_AXP_s116
                                 + 0.911514 w_k1_DIS_s116
                                 + 0.8983 w_k1_FDX_s116 + 1.1368 w_k1_IBM_s116
                                 + 0.890905 w_k1_INTC_s116
                                 + 0.942058 w_k1_KO_s116
                                 + 1.78075 w_k1_MAT_s116
                                 + 1.05017 w_k1_MSFT_s116
                                 + 0.929685 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.866897 w_k1_ADM_s116
                                 + 1.1227 w_k1_AXP_s116 + 1.45114 w_k1_DIS_s116
                                 + 1.14844 w_k1_FDX_s116
                                 + 1.24195 w_k1_IBM_s116
                                 + 1.26809 w_k1_INTC_s116 + 1.4986 w_k1_KO_s116
                                 + 1.2228 w_k1_MAT_s116
                                 + 1.21225 w_k1_MSFT_s116
                                 + 1.08925 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.0863 w_k1_ADM_s116
                                 + 1.39404 w_k1_AXP_s116
                                 + 1.39263 w_k1_DIS_s116
                                 + 1.40125 w_k1_FDX_s116
                                 + 0.89941 w_k1_IBM_s116
                                 + 0.839599 w_k1_INTC_s116
                                 + 1.01957 w_k1_KO_s116 + 1.26113 w_k1_MAT_s116
                                 + 1.19747 w_k1_MSFT_s116
                                 + 1.06741 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.827494 w_k1_ADM_s117
                                 + 0.897397 w_k1_AXP_s117
                                 + 0.795567 w_k1_DIS_s117
                                 + 0.866867 w_k1_FDX_s117
                                 + 0.994924 w_k1_IBM_s117
                                 + 0.776638 w_k1_INTC_s117
                                 + 0.949529 w_k1_KO_s117
                                 + 0.903717 w_k1_MAT_s117
                                 + 0.845635 w_k1_MSFT_s117
                                 + 0.928117 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.81313 w_k1_ADM_s117
                                 + 1.06433 w_k1_AXP_s117
                                 + 0.755436 w_k1_DIS_s117
                                 + 0.976408 w_k1_FDX_s117
                                 + 0.791495 w_k1_IBM_s117
                                 + 0.840231 w_k1_INTC_s117
                                 + 0.790968 w_k1_KO_s117
                                 + 0.985302 w_k1_MAT_s117
                                 + 0.75848 w_k1_MSFT_s117
                                 + 0.948165 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.25011 w_k1_ADM_s117
                                 + 1.59315 w_k1_AXP_s117
                                 + 1.48705 w_k1_DIS_s117
                                 + 1.33481 w_k1_FDX_s117
                                 + 1.25373 w_k1_IBM_s117
                                 + 0.875802 w_k1_INTC_s117
                                 + 1.11379 w_k1_KO_s117 + 1.33722 w_k1_MAT_s117
                                 + 1.07189 w_k1_MSFT_s117 + 1.0538 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.909378 w_k1_ADM_s117
                                 + 0.961442 w_k1_AXP_s117
                                 + 1.02615 w_k1_DIS_s117 + 1.1049 w_k1_FDX_s117
                                 + 0.85062 w_k1_IBM_s117
                                 + 0.890071 w_k1_INTC_s117
                                 + 0.953442 w_k1_KO_s117
                                 + 0.911896 w_k1_MAT_s117
                                 + 1.03073 w_k1_MSFT_s117
                                 + 0.973479 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.918626 w_k1_ADM_s117
                                 + 0.75305 w_k1_AXP_s117
                                 + 0.779647 w_k1_DIS_s117
                                 + 0.866351 w_k1_FDX_s117
                                 + 0.69914 w_k1_IBM_s117
                                 + 0.688718 w_k1_INTC_s117
                                 + 0.849937 w_k1_KO_s117
                                 + 0.618199 w_k1_MAT_s117
                                 + 0.869616 w_k1_MSFT_s117
                                 + 0.847876 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.01828 w_k1_ADM_s117
                                 + 1.1448 w_k1_AXP_s117 + 1.20502 w_k1_DIS_s117
                                 + 1.1752 w_k1_FDX_s117 + 1.24827 w_k1_IBM_s117
                                 + 1.15489 w_k1_INTC_s117
                                 + 1.20951 w_k1_KO_s117 + 1.43441 w_k1_MAT_s117
                                 + 1.14134 w_k1_MSFT_s117
                                 + 1.19672 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.985927 w_k1_ADM_s117
                                 + 0.860233 w_k1_AXP_s117
                                 + 1.32265 w_k1_DIS_s117
                                 + 1.15109 w_k1_FDX_s117
                                 + 0.944163 w_k1_IBM_s117
                                 + 0.878785 w_k1_INTC_s117
                                 + 1.17229 w_k1_KO_s117
                                 + 0.656133 w_k1_MAT_s117
                                 + 1.20361 w_k1_MSFT_s117
                                 + 1.09643 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.926482 w_k1_ADM_s117
                                 + 0.948833 w_k1_AXP_s117
                                 + 1.08144 w_k1_DIS_s117
                                 + 0.972563 w_k1_FDX_s117
                                 + 1.45548 w_k1_IBM_s117
                                 + 0.921897 w_k1_INTC_s117
                                 + 1.37328 w_k1_KO_s117 + 1.49466 w_k1_MAT_s117
                                 + 1.10447 w_k1_MSFT_s117 + 1.0304 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.023528762048934 w_k1_ADM_s000
                                 + 0.0417316409943562 w_k1_ADM_s001
                                 + 0.0228067384924412 w_k1_ADM_s002
                                 + 0.0550148281480121 w_k1_ADM_s003
                                 + 0.0847633795513015 w_k1_ADM_s004
                                 + 0.0581656953673818 w_k1_ADM_s005
                                 + 0.0208761674945803 w_k1_ADM_s006
                                 + 0.0322768719455887 w_k1_ADM_s007
                                 + 0.0221219657186667 w_k1_ADM_s010
                                 + 0.0208201711784502 w_k1_ADM_s011
                                 + 0.0180438558886033 w_k1_ADM_s012
                                 + 0.0269165791104502 w_k1_ADM_s013
                                 + 0.0494015519155769 w_k1_ADM_s014
                                 + 0.0157650275322875 w_k1_ADM_s015
                                 + 0.0226119810750021 w_k1_ADM_s016
                                 + 0.0246493256022333 w_k1_ADM_s017
                                 + 0.0292136851056486 w_k1_ADM_s100
                                 + 0.0363623572962028 w_k1_ADM_s101
                                 + 0.0573989573864277 w_k1_ADM_s102
                                 + 0.0735808694771697 w_k1_ADM_s103
                                 + 0.0588407492549411 w_k1_ADM_s104
                                 + 0.0205799296264144 w_k1_ADM_s105
                                 + 0.0160091168132537 w_k1_ADM_s106
                                 + 0.0362257892260022 w_k1_ADM_s107
                                 + 0.0146872759741864 w_k1_ADM_s110
                                 + 0.0136762087497744 w_k1_ADM_s111
                                 + 0.0120293775748888 w_k1_ADM_s112
                                 + 0.0230663586457434 w_k1_ADM_s113
                                 + 0.0172998556467772 w_k1_ADM_s114
                                 + 0.0466797262018103 w_k1_ADM_s115
                                 + 0.00820933619703304 w_k1_ADM_s116
                                 + 0.0146758699377802 w_k1_ADM_s117
                                 + 0.0214314111081469 w_k1_AXP_s000
                                 + 0.0435311288712987 w_k1_AXP_s001
                                 + 0.0228354411917657 w_k1_AXP_s002
                                 + 0.0548024836147147 w_k1_AXP_s003
                                 + 0.0766420556335824 w_k1_AXP_s004
                                 + 0.0570723250516037 w_k1_AXP_s005
                                 + 0.0219794035853996 w_k1_AXP_s006
                                 + 0.0312110835782342 w_k1_AXP_s007
                                 + 0.0231534651612736 w_k1_AXP_s010
                                 + 0.0214593023472489 w_k1_AXP_s011
                                 + 0.0158888056692419 w_k1_AXP_s012
                                 + 0.0234886596078869 w_k1_AXP_s013
                                 + 0.0553344580109283 w_k1_AXP_s014
                                 + 0.0178333450462054 w_k1_AXP_s015
                                 + 0.0246773469580345 w_k1_AXP_s016
                                 + 0.0300762315156483 w_k1_AXP_s017
                                 + 0.0296062712647693 w_k1_AXP_s100
                                 + 0.0337840214521897 w_k1_AXP_s101
                                 + 0.0602880360066631 w_k1_AXP_s102
                                 + 0.077710685201345 w_k1_AXP_s103
                                 + 0.0598522472555913 w_k1_AXP_s104
                                 + 0.0269458875435928 w_k1_AXP_s105
                                 + 0.0193358867549496 w_k1_AXP_s106
                                 + 0.0321857461692681 w_k1_AXP_s107
                                 + 0.0162500288539382 w_k1_AXP_s110
                                 + 0.0155852006207843 w_k1_AXP_s111
                                 + 0.0136337556794744 w_k1_AXP_s112
                                 + 0.0224274008810122 w_k1_AXP_s113
                                 + 0.0163863834243882 w_k1_AXP_s114
                                 + 0.0432467523572664 w_k1_AXP_s115
                                 + 0.0108195263126475 w_k1_AXP_s116
                                 + 0.015827663925989 w_k1_AXP_s117
                                 + 0.0236006128136748 w_k1_DIS_s000
                                 + 0.0405283065636989 w_k1_DIS_s001
                                 + 0.0218849584907531 w_k1_DIS_s002
                                 + 0.0576931577472721 w_k1_DIS_s003
                                 + 0.0773163695109551 w_k1_DIS_s004
                                 + 0.0557575079002938 w_k1_DIS_s005
                                 + 0.0199347960970076 w_k1_DIS_s006
                                 + 0.0324702181894541 w_k1_DIS_s007
                                 + 0.0257062929105424 w_k1_DIS_s010
                                 + 0.0239230248652259 w_k1_DIS_s011
                                 + 0.0169394565743307 w_k1_DIS_s012
                                 + 0.0248161652981154 w_k1_DIS_s013
                                 + 0.0551969722643376 w_k1_DIS_s014
                                 + 0.0179522724410009 w_k1_DIS_s015
                                 + 0.021772347242987 w_k1_DIS_s016
                                 + 0.0289224645426295 w_k1_DIS_s017
                                 + 0.0297519949095337 w_k1_DIS_s100
                                 + 0.0341988676503078 w_k1_DIS_s101
                                 + 0.0601308413030285 w_k1_DIS_s102
                                 + 0.0829619420416736 w_k1_DIS_s103
                                 + 0.0585937619402697 w_k1_DIS_s104
                                 + 0.0272201792191474 w_k1_DIS_s105
                                 + 0.0182026606744328 w_k1_DIS_s106
                                 + 0.0383715186065572 w_k1_DIS_s107
                                 + 0.0156675799570473 w_k1_DIS_s110
                                 + 0.0146782786048231 w_k1_DIS_s111
                                 + 0.0138956317724841 w_k1_DIS_s112
                                 + 0.0235733881500227 w_k1_DIS_s113
                                 + 0.0175012959929162 w_k1_DIS_s114
                                 + 0.0456179302153953 w_k1_DIS_s115
                                 + 0.0105764239045779 w_k1_DIS_s116
                                 + 0.0161339761651151 w_k1_DIS_s117
                                 + 0.0220589154191185 w_k1_FDX_s000
                                 + 0.0396903766308734 w_k1_FDX_s001
                                 + 0.021730532764163 w_k1_FDX_s002
                                 + 0.0575250452482638 w_k1_FDX_s003
                                 + 0.0725651856401641 w_k1_FDX_s004
                                 + 0.0551048527603615 w_k1_FDX_s005
                                 + 0.0193555787216221 w_k1_FDX_s006
                                 + 0.0310836191074827 w_k1_FDX_s007
                                 + 0.0247103687384788 w_k1_FDX_s010
                                 + 0.0218402515671929 w_k1_FDX_s011
                                 + 0.0162272284050585 w_k1_FDX_s012
                                 + 0.0243632334016018 w_k1_FDX_s013
                                 + 0.0497041272717795 w_k1_FDX_s014
                                 + 0.0170462944822651 w_k1_FDX_s015
                                 + 0.0210437549384966 w_k1_FDX_s016
                                 + 0.0301174549516739 w_k1_FDX_s017
                                 + 0.0298299265520285 w_k1_FDX_s100
                                 + 0.0333422790519738 w_k1_FDX_s101
                                 + 0.0621065103818784 w_k1_FDX_s102
                                 + 0.076779132458251 w_k1_FDX_s103
                                 + 0.0586913959649169 w_k1_FDX_s104
                                 + 0.0238403815864076 w_k1_FDX_s105
                                 + 0.0186487897680006 w_k1_FDX_s106
                                 + 0.0352802900633042 w_k1_FDX_s107
                                 + 0.0162071459696441 w_k1_FDX_s110
                                 + 0.0148125081118542 w_k1_FDX_s111
                                 + 0.0130743662850592 w_k1_FDX_s112
                                 + 0.0209231906909853 w_k1_FDX_s113
                                 + 0.0180390130393413 w_k1_FDX_s114
                                 + 0.0458692737412785 w_k1_FDX_s115
                                 + 0.0104025843155272 w_k1_FDX_s116
                                 + 0.0160060427133712 w_k1_FDX_s117
                                 + 0.0223543022515583 w_k1_IBM_s000
                                 + 0.0394532219182479 w_k1_IBM_s001
                                 + 0.0207950891117727 w_k1_IBM_s002
                                 + 0.056714840717683 w_k1_IBM_s003
                                 + 0.0757603115555175 w_k1_IBM_s004
                                 + 0.0550722296160065 w_k1_IBM_s005
                                 + 0.021079652099408 w_k1_IBM_s006
                                 + 0.0302460903194395 w_k1_IBM_s007
                                 + 0.0249250689080215 w_k1_IBM_s010
                                 + 0.0207645300222704 w_k1_IBM_s011
                                 + 0.0165414126456516 w_k1_IBM_s012
                                 + 0.0254386197514205 w_k1_IBM_s013
                                 + 0.0513631361110735 w_k1_IBM_s014
                                 + 0.0168246508307761 w_k1_IBM_s015
                                 + 0.0217749045887425 w_k1_IBM_s016
                                 + 0.0318102226361074 w_k1_IBM_s017
                                 + 0.030445205567496 w_k1_IBM_s100
                                 + 0.0331597317516612 w_k1_IBM_s101
                                 + 0.0629605484292767 w_k1_IBM_s102
                                 + 0.0755539514751692 w_k1_IBM_s103
                                 + 0.0558840060305841 w_k1_IBM_s104
                                 + 0.023624514958341 w_k1_IBM_s105
                                 + 0.0184487291004407 w_k1_IBM_s106
                                 + 0.0362512172230616 w_k1_IBM_s107
                                 + 0.0166493560293541 w_k1_IBM_s110
                                 + 0.014079833918095 w_k1_IBM_s111
                                 + 0.0139302897590229 w_k1_IBM_s112
                                 + 0.0217982782334943 w_k1_IBM_s113
                                 + 0.018492159637872 w_k1_IBM_s114
                                 + 0.0454650761109371 w_k1_IBM_s115
                                 + 0.0105111696640995 w_k1_IBM_s116
                                 + 0.0161586621319207 w_k1_IBM_s117
                                 + 0.0234811284218201 w_k1_INTC_s000
                                 + 0.0386517225115971 w_k1_INTC_s001
                                 + 0.0208993088630278 w_k1_INTC_s002
                                 + 0.0540552396195604 w_k1_INTC_s003
                                 + 0.0709598069035622 w_k1_INTC_s004
                                 + 0.0463404687400356 w_k1_INTC_s005
                                 + 0.0207548836921519 w_k1_INTC_s006
                                 + 0.031079530544243 w_k1_INTC_s007
                                 + 0.0212021692002121 w_k1_INTC_s010
                                 + 0.0226264509894226 w_k1_INTC_s011
                                 + 0.0153292046890848 w_k1_INTC_s012
                                 + 0.0227303541820831 w_k1_INTC_s013
                                 + 0.0497910012258705 w_k1_INTC_s014
                                 + 0.0185229962125779 w_k1_INTC_s015
                                 + 0.0207338095266679 w_k1_INTC_s016
                                 + 0.0299192949098811 w_k1_INTC_s017
                                 + 0.030999381052466 w_k1_INTC_s100
                                 + 0.0392203193408349 w_k1_INTC_s101
                                 + 0.0563896424153954 w_k1_INTC_s102
                                 + 0.0699857381704319 w_k1_INTC_s103
                                 + 0.0559975607316722 w_k1_INTC_s104
                                 + 0.0212289089553262 w_k1_INTC_s105
                                 + 0.019198206329837 w_k1_INTC_s106
                                 + 0.0324292792971162 w_k1_INTC_s107
                                 + 0.016850982987731 w_k1_INTC_s110
                                 + 0.0142150041240592 w_k1_INTC_s111
                                 + 0.0144072226163108 w_k1_INTC_s112
                                 + 0.0212225423683712 w_k1_INTC_s113
                                 + 0.0195129457618754 w_k1_INTC_s114
                                 + 0.0483821033661616 w_k1_INTC_s115
                                 + 0.00953289545351915 w_k1_INTC_s116
                                 + 0.01336840700551 w_k1_INTC_s117
                                 + 0.0221043797520855 w_k1_KO_s000
                                 + 0.0410430913652646 w_k1_KO_s001
                                 + 0.023432072811133 w_k1_KO_s002
                                 + 0.0535165124680459 w_k1_KO_s003
                                 + 0.080290883824338 w_k1_KO_s004
                                 + 0.0576126505177488 w_k1_KO_s005
                                 + 0.0201251722969861 w_k1_KO_s006
                                 + 0.0309510481199626 w_k1_KO_s007
                                 + 0.0278804641286897 w_k1_KO_s010
                                 + 0.021964357470655 w_k1_KO_s011
                                 + 0.0149580825651813 w_k1_KO_s012
                                 + 0.0250626356553184 w_k1_KO_s013
                                 + 0.0556628668319936 w_k1_KO_s014
                                 + 0.0161956460961661 w_k1_KO_s015
                                 + 0.0203628291087217 w_k1_KO_s016
                                 + 0.0287539934293158 w_k1_KO_s017
                                 + 0.0293327472422089 w_k1_KO_s100
                                 + 0.036145401380021 w_k1_KO_s101
                                 + 0.0622472343951521 w_k1_KO_s102
                                 + 0.0754825340347535 w_k1_KO_s103
                                 + 0.0564829009423659 w_k1_KO_s104
                                 + 0.0251796649327404 w_k1_KO_s105
                                 + 0.0181205804330031 w_k1_KO_s106
                                 + 0.0332601806059584 w_k1_KO_s107
                                 + 0.0156756627253029 w_k1_KO_s110
                                 + 0.0130841461157666 w_k1_KO_s111
                                 + 0.0123587677704035 w_k1_KO_s112
                                 + 0.0228650806103667 w_k1_KO_s113
                                 + 0.0180930666732177 w_k1_KO_s114
                                 + 0.043650532402004 w_k1_KO_s115
                                 + 0.0108577611054338 w_k1_KO_s116
                                 + 0.0162171554124387 w_k1_KO_s117
                                 + 0.0238683409121084 w_k1_MAT_s000
                                 + 0.034680947939993 w_k1_MAT_s001
                                 + 0.0227892695940281 w_k1_MAT_s002
                                 + 0.0462650966301675 w_k1_MAT_s003
                                 + 0.072140610100194 w_k1_MAT_s004
                                 + 0.0555782384491501 w_k1_MAT_s005
                                 + 0.0204676916245829 w_k1_MAT_s006
                                 + 0.0342582162477108 w_k1_MAT_s007
                                 + 0.0258432915161166 w_k1_MAT_s010
                                 + 0.0201350240480517 w_k1_MAT_s011
                                 + 0.0162887792146215 w_k1_MAT_s012
                                 + 0.0222835797780155 w_k1_MAT_s013
                                 + 0.0527367552723817 w_k1_MAT_s014
                                 + 0.0160086469101347 w_k1_MAT_s015
                                 + 0.0204611640767669 w_k1_MAT_s016
                                 + 0.0304954454197241 w_k1_MAT_s017
                                 + 0.0344490733627631 w_k1_MAT_s100
                                 + 0.0302532986916643 w_k1_MAT_s101
                                 + 0.0700704964737259 w_k1_MAT_s102
                                 + 0.0814328397979105 w_k1_MAT_s103
                                 + 0.0507509957569484 w_k1_MAT_s104
                                 + 0.0316480888330367 w_k1_MAT_s105
                                 + 0.0171117512163915 w_k1_MAT_s106
                                 + 0.036048236354113 w_k1_MAT_s107
                                 + 0.0199281099644456 w_k1_MAT_s110
                                 + 0.0182756834221872 w_k1_MAT_s111
                                 + 0.0128997150528348 w_k1_MAT_s112
                                 + 0.0196702903535309 w_k1_MAT_s113
                                 + 0.017008982204965 w_k1_MAT_s114
                                 + 0.0429782832944487 w_k1_MAT_s115
                                 + 0.013121828544168 w_k1_MAT_s116
                                 + 0.0162053220959165 w_k1_MAT_s117
                                 + 0.023842141073362 w_k1_MSFT_s000
                                 + 0.0419123372607628 w_k1_MSFT_s001
                                 + 0.0219002479696757 w_k1_MSFT_s002
                                 + 0.056507452994232 w_k1_MSFT_s003
                                 + 0.0664572800581089 w_k1_MSFT_s004
                                 + 0.0517798086504784 w_k1_MSFT_s005
                                 + 0.0210844604867416 w_k1_MSFT_s006
                                 + 0.0328860315386499 w_k1_MSFT_s007
                                 + 0.0253720675528145 w_k1_MSFT_s010
                                 + 0.0227983181422438 w_k1_MSFT_s011
                                 + 0.016761271174088 w_k1_MSFT_s012
                                 + 0.023506861682133 w_k1_MSFT_s013
                                 + 0.0500938613516859 w_k1_MSFT_s014
                                 + 0.0164769077294319 w_k1_MSFT_s015
                                 + 0.0193657794659772 w_k1_MSFT_s016
                                 + 0.0282404516528701 w_k1_MSFT_s017
                                 + 0.0301044503134895 w_k1_MSFT_s100
                                 + 0.0331704345836158 w_k1_MSFT_s101
                                 + 0.0612041418180521 w_k1_MSFT_s102
                                 + 0.0800206413268822 w_k1_MSFT_s103
                                 + 0.0580913149214279 w_k1_MSFT_s104
                                 + 0.0245071536828464 w_k1_MSFT_s105
                                 + 0.0203570804572824 w_k1_MSFT_s106
                                 + 0.0341677284929121 w_k1_MSFT_s107
                                 + 0.0154463688539519 w_k1_MSFT_s110
                                 + 0.0152960615391717 w_k1_MSFT_s111
                                 + 0.0127144168135491 w_k1_MSFT_s112
                                 + 0.0219034887001175 w_k1_MSFT_s113
                                 + 0.0196303230548259 w_k1_MSFT_s114
                                 + 0.0459361310960826 w_k1_MSFT_s115
                                 + 0.0105268741496328 w_k1_MSFT_s116
                                 + 0.0152808956069244 w_k1_MSFT_s117
                                 + 0.0220248152541797 w_k1_PG_s000
                                 + 0.0399143437758409 w_k1_PG_s001
                                 + 0.0209662391114412 w_k1_PG_s002
                                 + 0.0546583995474831 w_k1_PG_s003
                                 + 0.0714214312083402 w_k1_PG_s004
                                 + 0.0543952848392083 w_k1_PG_s005
                                 + 0.0192738784911544 w_k1_PG_s006
                                 + 0.0315845385223387 w_k1_PG_s007
                                 + 0.0273835846769531 w_k1_PG_s010
                                 + 0.0219942424463257 w_k1_PG_s011
                                 + 0.0147781968390766 w_k1_PG_s012
                                 + 0.0242378478008286 w_k1_PG_s013
                                 + 0.0538930496345757 w_k1_PG_s014
                                 + 0.0168332977965314 w_k1_PG_s015
                                 + 0.0209218025222329 w_k1_PG_s016
                                 + 0.0315265935793095 w_k1_PG_s017
                                 + 0.0295470504032206 w_k1_PG_s100
                                 + 0.0315612101346883 w_k1_PG_s101
                                 + 0.0630907477011893 w_k1_PG_s102
                                 + 0.0778407423453971 w_k1_PG_s103
                                 + 0.05763541185791 w_k1_PG_s104
                                 + 0.0224615017992571 w_k1_PG_s105
                                 + 0.0172742017179081 w_k1_PG_s106
                                 + 0.0346061651998261 w_k1_PG_s107
                                 + 0.017706192251337 w_k1_PG_s110
                                 + 0.0133457056389239 w_k1_PG_s111
                                 + 0.0131235002546312 w_k1_PG_s112
                                 + 0.0209483522422576 w_k1_PG_s113
                                 + 0.0191395603789527 w_k1_PG_s114
                                 + 0.0455089539035593 w_k1_PG_s115
                                 + 0.00994369827120238 w_k1_PG_s116
                                 + 0.0154101791164695 w_k1_PG_s117 >= 1.15
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
