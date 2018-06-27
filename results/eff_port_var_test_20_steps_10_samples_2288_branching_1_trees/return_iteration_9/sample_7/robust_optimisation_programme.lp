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
 w_k1_ADM_s0_return_constraint:  1.07905 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.910644 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.0064 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.07474 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.957307 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.16737 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.63601 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.07102 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.798782 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.846802 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.674963 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.07037 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.29422 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.08997 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.12152 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.813813 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.82435 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.860157 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.10441 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.02465 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.965523 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.34847 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.24267 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.08246 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.792181 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.02516 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.957985 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.14451 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.958009 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.53238 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.2093 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.843679 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.977997 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.34714 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.07773 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.813525 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.923984 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06685 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23142 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.860123 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.834545 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.07242 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.9878 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.28541 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.654724 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.864972 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.994694 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.773614 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.501702 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.14597 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.12647 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.44267 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.31042 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.776238 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.79942 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.905712 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.07708 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.25782 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.10242 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.42927 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.25071 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.05016 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.635762 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.734775 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.862305 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.993848 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.993504 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.62173 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.29354 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.09402 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.877454 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.5799 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.07145 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.733981 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.730771 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.25673 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31589 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.879763 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.798106 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.11317 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.863495 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.28046 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.561716 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.11912 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.867635 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.728556 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.561184 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.09896 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.56868 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.14314 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.36895 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.72626 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.729002 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.23103 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.18117 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.03673 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.892257 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.4671 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.41657 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.1186 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.674687 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.807827 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.907939 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.15439 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.923336 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.84803 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.882731 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.13155 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.788125 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.78401 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.2492 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.720203 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.787542 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.35917 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17169 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.908626 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.895048 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.06512 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.928835 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.18668 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.665776 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.10463 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.788936 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.850603 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.6944 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.18902 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.23119 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.16887 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.03337 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.770374 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.710229 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.00103 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.23495 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.09974 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.85584 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.2291 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.2365 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.07473 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.758296 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.958822 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.01668 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.27221 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.860073 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.65175 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.02195 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.969714 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.957579 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.43325 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.20119 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.76031 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.842752 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.963196 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.19873 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.925913 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.730162 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.13602 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.88201 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.39437 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.16686 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.692043 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.52452 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.893933 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.636128 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.854223 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.31823 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.41838 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.98275 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.87256 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.22403 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.08548 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.630483 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.841097 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.36034 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.2132 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.31006 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.952489 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.799503 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.514366 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.810895 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.733962 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.43889 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.03348 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.829216 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.17638 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.582105 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.950199 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.720189 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.03631 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.763946 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.80078 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.012 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.964927 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.916332 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.27386 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.988955 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.21112 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.762168 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.0331 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.06901 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.894487 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.739611 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.1735 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.21428 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.15821 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.943145 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.739782 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.685649 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.08429 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.12748 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.03171 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.874491 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.11384 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.05312 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.832744 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.725285 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.695375 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.857374 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.02935 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.834573 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.24312 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.880389 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.870144 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.726061 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.07644 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.09289 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.699082 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.783047 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.13608 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1696 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.903072 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.775782 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.12788 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.901791 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.27868 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.952493 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.0922 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.1693 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.922678 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.643253 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.01329 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.39351 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.27386 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.50586 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.833841 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.864237 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.18772 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.939979 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.997799 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.21754 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.27401 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.24363 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.0441 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.789394 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.684516 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.941672 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.97197 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.07028 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.45188 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.887092 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.15088 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.764117 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.25876 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.972918 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.921757 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.813666 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.61506 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.33308 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.914418 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.691477 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.08918 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.736429 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.42827 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.06761 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.839915 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.45132 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.870859 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.503554 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.855222 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.02153 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.67098 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.98897 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.868566 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.27393 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.01778 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.505047 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.04334 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.28016 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.24147 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.60801 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.23754 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.885569 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.558059 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.14851 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.06301 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.64569 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.38582 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.18911 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.17092 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.597093 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.10549 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.802631 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.888083 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.675974 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.68017 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1625 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.937975 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.851303 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.07828 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.945997 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.14206 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.797378 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.13678 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.8114 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.919758 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.750862 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.09174 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.28604 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.0684 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.01401 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.785614 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.713198 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.0436 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.17425 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.02049 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.825457 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.20584 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03375 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.938843 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.828079 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.894353 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.974816 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.19537 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.89356 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.44079 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.987831 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.04046 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.902617 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.29305 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.16006 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.758784 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.895286 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.11286 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13876 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.924762 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.859044 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.07707 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.967052 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.1419 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.895695 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04768 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.02017 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.927519 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.770738 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.03838 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.17201 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08098 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.23981 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.877505 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.894268 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.999095 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.987771 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.950843 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.997966 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.12364 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.11651 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05081 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.804911 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.864655 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.926576 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.01979 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.950842 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.32309 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07019 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.08853 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.93275 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.24596 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.13319 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.86089 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.890323 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.1386 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0949996852670748 z_k1_s0000
                                 - 0.0659999289749354 z_k1_s0001
                                 - 0.227000104216622 z_k1_s0002
                                 - 0.234999959607602 z_k1_s0003
                                 - 0.0499999376931377 z_k1_s0004
                                 - 0.058999962007882 z_k1_s0005
                                 - 0.123000114134965 z_k1_s0006
                                 - 0.079999956408988 z_k1_s0007
                                 - 0.247999464544133 z_k1_s0010
                                 - 0.522001023590556 z_k1_s0011
                                 - 0.244001019391414 z_k1_s0012
                                 - 0.136000122434207 z_k1_s0013
                                 - 0.113000173499067 z_k1_s0014
                                 - 0.201999915274127 z_k1_s0015
                                 - 0.125000040760796 z_k1_s0016
                                 - 0.153000137738483 z_k1_s0017
                                 - 0.14099974111533 z_k1_s0020
                                 - 0.123999738002161 z_k1_s0021
                                 - 0.0579999992331764 z_k1_s0022
                                 - 0.0549999759073401 z_k1_s0023
                                 - 0.0829999354820073 z_k1_s0024
                                 - 0.270000163326729 z_k1_s0025
                                 - 0.109999564614974 z_k1_s0026
                                 - 0.126999567728144 z_k1_s0027
                                 - 0.341000440067058 z_k1_s0030
                                 - 0.298000694823675 z_k1_s0031
                                 - 0.617000499790265 z_k1_s0032
                                 - 0.275000885763204 z_k1_s0033
                                 - 0.338000567189499 z_k1_s0034
                                 - 0.124999953792833 z_k1_s0035
                                 - 0.201999825646921 z_k1_s0036
                                 - 0.154999858067201 z_k1_s0037
                                 - 0.0619999762513817 z_k1_s0040
                                 - 0.265999691949569 z_k1_s0041
                                 - 0.0779999288904664 z_k1_s0042
                                 - 0.0599999593465063 z_k1_s0043
                                 - 0.112000398873266 z_k1_s0044
                                 - 0.0629999847038194 z_k1_s0045
                                 - 0.151000089418622 z_k1_s0046
                                 - 0.120999835845491 z_k1_s0047
                                 - 0.561000904154518 z_k1_s0050
                                 - 0.265999870367042 z_k1_s0051
                                 - 0.229999890143681 z_k1_s0052
                                 - 0.334000294855158 z_k1_s0053
                                 - 0.182999775760542 z_k1_s0054
                                 - 0.483999965036074 z_k1_s0055
                                 - 0.642998665348781 z_k1_s0056
                                 - 0.267999786796199 z_k1_s0057
                                 - 0.32600010768748 z_k1_s0060
                                 - 0.12400001810199 z_k1_s0061
                                 - 0.117999917064813 z_k1_s0062
                                 - 0.0829999486811809 z_k1_s0063
                                 - 0.116999969225272 z_k1_s0064
                                 - 0.191999369190284 z_k1_s0065
                                 - 0.205999466943719 z_k1_s0066
                                 - 0.213999325659999 z_k1_s0067
                                 - 0.208999669539932 z_k1_s0070
                                 - 0.305999417422169 z_k1_s0071
                                 - 0.128000430751297 z_k1_s0072
                                 - 0.065999914812597 z_k1_s0073
                                 - 0.147999853119127 z_k1_s0074
                                 - 0.0899999500534963 z_k1_s0075
                                 - 0.14900018843727 z_k1_s0076
                                 - 0.117999869767961 z_k1_s0077
                                 - 0.336001013422683 z_k1_s0100
                                 - 0.297999407380595 z_k1_s0101
                                 - 0.344000664694593 z_k1_s0102
                                 - 0.20500013359412 z_k1_s0103
                                 - 0.494999628702764 z_k1_s0104
                                 - 0.220999958138021 z_k1_s0105
                                 - 0.172000071347255 z_k1_s0106
                                 - 0.538999015698473 z_k1_s0107
                                 - 0.774003110048217 z_k1_s0110
                                 - 0.608000577038252 z_k1_s0111
                                 - 0.427000529664261 z_k1_s0112
                                 - 0.531000449729927 z_k1_s0113
                                 - 1.25600142171729 z_k1_s0114
                                 - 0.569000445388548 z_k1_s0115
                                 - 0.395000056951504 z_k1_s0116
                                 - 1.90500369538805 z_k1_s0117
                                 - 0.149999940753423 z_k1_s0120
                                 - 0.281000161791409 z_k1_s0121
                                 - 0.36000021360821 z_k1_s0122
                                 - 0.239999549405482 z_k1_s0123
                                 - 0.241999145375534 z_k1_s0124
                                 - 0.0989999253172598 z_k1_s0125
                                 - 0.1489999648684 z_k1_s0126
                                 - 0.257999471135894 z_k1_s0127
                                 - 0.0550000661536415 z_k1_s0130
                                 - 0.104000075272299 z_k1_s0131
                                 - 0.141000518321446 z_k1_s0132
                                 - 0.255999784823783 z_k1_s0133
                                 - 0.26500069237695 z_k1_s0134
                                 - 0.155000560069667 z_k1_s0135
                                 - 0.122000109377137 z_k1_s0136
                                 - 0.272000028250485 z_k1_s0137
                                 - 0.436998182432431 z_k1_s0140
                                 - 0.294000098785942 z_k1_s0141
                                 - 0.339000101104196 z_k1_s0142
                                 - 0.934999211767985 z_k1_s0143
                                 - 0.506001197027723 z_k1_s0144
                                 - 0.397000271372203 z_k1_s0145
                                 - 0.699000537850315 z_k1_s0146
                                 - 0.575000029622131 z_k1_s0147
                                 - 0.614000822891412 z_k1_s0150
                                 - 0.591997312255537 z_k1_s0151
                                 - 0.606000454294346 z_k1_s0152
                                 - 0.309999301172633 z_k1_s0153
                                 - 0.316999373995672 z_k1_s0154
                                 - 0.697999699172714 z_k1_s0155
                                 - 1.39599440435754 z_k1_s0156
                                 - 0.460999017159812 z_k1_s0157
                                 - 0.36599975356686 z_k1_s0160
                                 - 0.236999360620951 z_k1_s0161
                                 - 0.17399996281051 z_k1_s0162
                                 - 0.0850000005223836 z_k1_s0163
                                 - 0.145000077408806 z_k1_s0164
                                 - 0.246000900594589 z_k1_s0165
                                 - 0.102999991068296 z_k1_s0166
                                 - 0.270000671189043 z_k1_s0167
                                 - 1.20099954986901 z_k1_s0170
                                 - 0.900998596336644 z_k1_s0171
                                 - 0.660000912371507 z_k1_s0172
                                 - 0.340999779908793 z_k1_s0173
                                 - 0.44600011364512 z_k1_s0174
                                 - 1.14800032973137 z_k1_s0175
                                 - 0.896997872383065 z_k1_s0176
                                 - 0.33300011010117 z_k1_s0177
                                 - 1.32699881210248 z_k1_s1000
                                 - 1.24700327360983 z_k1_s1001
                                 - 1.54000004266643 z_k1_s1002
                                 - 0.808999278459764 z_k1_s1003
                                 - 0.430000034804245 z_k1_s1004
                                 - 1.3010004824172 z_k1_s1005
                                 - 0.679999091758628 z_k1_s1006
                                 - 1.4789965043337 z_k1_s1007
                                 - 0.199000061028707 z_k1_s1010
                                 - 0.647001182546613 z_k1_s1011
                                 - 0.392999153191353 z_k1_s1012
                                 - 0.418999913534813 z_k1_s1013
                                 - 0.314999879961016 z_k1_s1014
                                 - 0.837000569056425 z_k1_s1015
                                 - 0.272000036393006 z_k1_s1016
                                 - 0.259999916834489 z_k1_s1017
                                 - 1.25900479527028 z_k1_s1020
                                 - 2.13899641058512 z_k1_s1021
                                 - 0.906000510603141 z_k1_s1022
                                 - 1.34999732011245 z_k1_s1023
                                 - 0.560000124102741 z_k1_s1024
                                 - 1.18199718755158 z_k1_s1025
                                 - 1.43300118745668 z_k1_s1026
                                 - 0.70399996535479 z_k1_s1027
                                 - 0.302999261101722 z_k1_s1030
                                 - 0.567999886169815 z_k1_s1031
                                 - 0.172999846863793 z_k1_s1032
                                 - 0.223999838360025 z_k1_s1033
                                 - 0.172999846863793 z_k1_s1034
                                 - 0.254000378322056 z_k1_s1035
                                 - 0.376000233689617 z_k1_s1036
                                 - 0.366999145261787 z_k1_s1037
                                 - 0.215999925619992 z_k1_s1040
                                 - 0.68899900450234 z_k1_s1041
                                 - 0.782998351398574 z_k1_s1042
                                 - 0.270999769176119 z_k1_s1043
                                 - 0.857997559475572 z_k1_s1044
                                 - 0.89500073291071 z_k1_s1045
                                 - 0.266999896271944 z_k1_s1046
                                 - 0.263999673369068 z_k1_s1047
                                 - 0.345998691050737 z_k1_s1050
                                 - 0.286999752769951 z_k1_s1051
                                 - 0.353999611681448 z_k1_s1052
                                 - 0.380999353936938 z_k1_s1053
                                 - 0.472999472695838 z_k1_s1054
                                 - 0.317000961886345 z_k1_s1055
                                 - 0.461998580703406 z_k1_s1056
                                 - 0.371000446397323 z_k1_s1057
                                 - 1.30000022497635 z_k1_s1060
                                 - 0.939995158086801 z_k1_s1061
                                 - 1.92199549485009 z_k1_s1062
                                 - 0.551998594151897 z_k1_s1063
                                 - 1.10499643778859 z_k1_s1064
                                 - 0.938994240405868 z_k1_s1065
                                 - 0.415998904255137 z_k1_s1066
                                 - 1.16699494714171 z_k1_s1067
                                 - 0.377000237914622 z_k1_s1070
                                 - 0.179999869229801 z_k1_s1071
                                 - 0.39199873743382 z_k1_s1072
                                 - 0.269999941344697 z_k1_s1073
                                 - 0.137000045608233 z_k1_s1074
                                 - 0.655000479002856 z_k1_s1075
                                 - 0.32399948961365 z_k1_s1076
                                 - 0.414999736696484 z_k1_s1077
                                 - 0.165000521286981 z_k1_s1100
                                 - 0.114000164943412 z_k1_s1101
                                 - 0.0820001542242672 z_k1_s1102
                                 - 0.173000792392207 z_k1_s1103
                                 - 0.147000507801199 z_k1_s1104
                                 - 0.274000337839334 z_k1_s1105
                                 - 0.0750000959574876 z_k1_s1106
                                 - 0.162999857009697 z_k1_s1107
                                 - 0.377000231978187 z_k1_s1110
                                 - 0.884000280900547 z_k1_s1111
                                 - 0.945997989836752 z_k1_s1112
                                 - 0.394000123280108 z_k1_s1113
                                 - 0.301000169475421 z_k1_s1114
                                 - 0.386000049785072 z_k1_s1115
                                 - 0.379999888713784 z_k1_s1116
                                 - 0.56999834977051 z_k1_s1117
                                 - 0.135999823531127 z_k1_s1120
                                 - 0.125000218702124 z_k1_s1121
                                 - 0.0969998676802767 z_k1_s1122
                                 - 0.136999867424824 z_k1_s1123
                                 - 0.11699986755255 z_k1_s1124
                                 - 0.0930005701071521 z_k1_s1125
                                 - 0.0989999554676703 z_k1_s1126
                                 - 0.0740001751277441 z_k1_s1127
                                 - 0.120999706506075 z_k1_s1130
                                 - 0.111999684351189 z_k1_s1131
                                 - 0.153999514808014 z_k1_s1132
                                 - 0.25500028655597 z_k1_s1133
                                 - 0.398998184640356 z_k1_s1134
                                 - 0.178999462627429 z_k1_s1135
                                 - 0.551998765972903 z_k1_s1136
                                 - 0.274999426013571 z_k1_s1137
                                 - 0.233001014185864 z_k1_s1140
                                 - 0.344001459720596 z_k1_s1141
                                 - 0.28500048064033 z_k1_s1142
                                 - 0.242000955553131 z_k1_s1143
                                 - 0.106000517064199 z_k1_s1144
                                 - 0.271000961404063 z_k1_s1145
                                 - 0.0750003294641325 z_k1_s1146
                                 - 0.197001248716798 z_k1_s1147
                                 - 0.576000563508704 z_k1_s1150
                                 - 0.42700118845696 z_k1_s1151
                                 - 0.180000363146888 z_k1_s1152
                                 - 0.261000177402207 z_k1_s1153
                                 - 0.137000215430551 z_k1_s1154
                                 - 0.139000407901493 z_k1_s1155
                                 - 0.352001452813352 z_k1_s1156
                                 - 0.422000707279604 z_k1_s1157
                                 - 0.311000445095863 z_k1_s1160
                                 - 0.227001453152137 z_k1_s1161
                                 - 0.111000558705439 z_k1_s1162
                                 - 0.386001377196588 z_k1_s1163
                                 - 0.395001489048675 z_k1_s1164
                                 - 0.197000875110939 z_k1_s1165
                                 - 0.168000651499264 z_k1_s1166
                                 - 0.257001825992427 z_k1_s1167
                                 - 0.181000321990039 z_k1_s1170
                                 - 0.105000210310286 z_k1_s1171
                                 - 0.129000291935554 z_k1_s1172
                                 - 0.291000696105853 z_k1_s1173
                                 - 0.17100028797951 z_k1_s1174
                                 - 0.217999860627937 z_k1_s1175
                                 - 0.254001157467955 z_k1_s1176
                                 - 0.119000257925026 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.865104 w_k1_ADM_s000
                                 + 0.754895 w_k1_AXP_s000
                                 + 0.739296 w_k1_DIS_s000
                                 + 0.813147 w_k1_FDX_s000
                                 + 0.682165 w_k1_IBM_s000
                                 + 0.98753 w_k1_INTC_s000
                                 + 0.710346 w_k1_KO_s000
                                 + 0.741668 w_k1_MAT_s000
                                 + 0.803219 w_k1_MSFT_s000
                                 + 0.868444 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.32076 w_k1_ADM_s000
                                 + 1.1562 w_k1_AXP_s000 + 1.29032 w_k1_DIS_s000
                                 + 1.24656 w_k1_FDX_s000
                                 + 0.801643 w_k1_IBM_s000
                                 + 1.20161 w_k1_INTC_s000 + 1.0384 w_k1_KO_s000
                                 + 0.87349 w_k1_MAT_s000
                                 + 1.20066 w_k1_MSFT_s000
                                 + 1.06373 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.01585 w_k1_ADM_s000
                                 + 0.822251 w_k1_AXP_s000
                                 + 0.868631 w_k1_DIS_s000
                                 + 0.853672 w_k1_FDX_s000
                                 + 0.953264 w_k1_IBM_s000
                                 + 0.988022 w_k1_INTC_s000
                                 + 0.938874 w_k1_KO_s000
                                 + 0.824654 w_k1_MAT_s000
                                 + 0.80831 w_k1_MSFT_s000
                                 + 0.800858 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.04007 w_k1_ADM_s000
                                 + 1.10871 w_k1_AXP_s000
                                 + 0.987967 w_k1_DIS_s000
                                 + 1.07196 w_k1_FDX_s000
                                 + 1.03881 w_k1_IBM_s000
                                 + 1.01201 w_k1_INTC_s000
                                 + 0.880224 w_k1_KO_s000
                                 + 0.956335 w_k1_MAT_s000
                                 + 0.949429 w_k1_MSFT_s000
                                 + 0.957029 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.02786 w_k1_ADM_s000
                                 + 1.03139 w_k1_AXP_s000
                                 + 0.850048 w_k1_DIS_s000
                                 + 1.00546 w_k1_FDX_s000
                                 + 0.700811 w_k1_IBM_s000
                                 + 1.17387 w_k1_INTC_s000
                                 + 0.756163 w_k1_KO_s000
                                 + 0.696178 w_k1_MAT_s000
                                 + 0.899844 w_k1_MSFT_s000
                                 + 0.98331 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.10904 w_k1_ADM_s000
                                 + 1.21295 w_k1_AXP_s000
                                 + 1.24873 w_k1_DIS_s000
                                 + 0.93499 w_k1_FDX_s000
                                 + 1.56103 w_k1_IBM_s000
                                 + 1.23101 w_k1_INTC_s000
                                 + 1.25022 w_k1_KO_s000 + 1.51901 w_k1_MAT_s000
                                 + 1.07849 w_k1_MSFT_s000
                                 + 1.12169 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.49082 w_k1_ADM_s000
                                 + 1.48372 w_k1_AXP_s000
                                 + 1.65642 w_k1_DIS_s000
                                 + 1.23264 w_k1_FDX_s000 + 1.3879 w_k1_IBM_s000
                                 + 1.51731 w_k1_INTC_s000
                                 + 1.44057 w_k1_KO_s000 + 1.53713 w_k1_MAT_s000
                                 + 1.34787 w_k1_MSFT_s000
                                 + 1.21007 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.07383 w_k1_ADM_s000
                                 + 1.10379 w_k1_AXP_s000
                                 + 1.52812 w_k1_DIS_s000
                                 + 1.20146 w_k1_FDX_s000 + 1.0278 w_k1_IBM_s000
                                 + 1.48871 w_k1_INTC_s000
                                 + 1.34696 w_k1_KO_s000 + 1.24196 w_k1_MAT_s000
                                 + 1.37147 w_k1_MSFT_s000 + 1.1091 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.92531 w_k1_ADM_s001
                                 + 0.969481 w_k1_AXP_s001
                                 + 0.744748 w_k1_DIS_s001
                                 + 0.859538 w_k1_FDX_s001
                                 + 0.88442 w_k1_IBM_s001
                                 + 0.943201 w_k1_INTC_s001
                                 + 0.780403 w_k1_KO_s001
                                 + 0.845599 w_k1_MAT_s001
                                 + 0.915755 w_k1_MSFT_s001
                                 + 0.869999 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.03045 w_k1_ADM_s001
                                 + 1.00363 w_k1_AXP_s001 + 1.0977 w_k1_DIS_s001
                                 + 1.07092 w_k1_FDX_s001
                                 + 1.02269 w_k1_IBM_s001
                                 + 1.28642 w_k1_INTC_s001
                                 + 0.924278 w_k1_KO_s001
                                 + 0.774799 w_k1_MAT_s001
                                 + 1.25139 w_k1_MSFT_s001
                                 + 1.08408 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.772726 w_k1_ADM_s001
                                 + 0.6337 w_k1_AXP_s001
                                 + 0.631576 w_k1_DIS_s001
                                 + 0.69639 w_k1_FDX_s001
                                 + 0.906422 w_k1_IBM_s001
                                 + 0.994549 w_k1_INTC_s001
                                 + 0.748193 w_k1_KO_s001
                                 + 0.50527 w_k1_MAT_s001
                                 + 0.779345 w_k1_MSFT_s001
                                 + 0.766698 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.814196 w_k1_ADM_s001
                                 + 0.882695 w_k1_AXP_s001
                                 + 0.770303 w_k1_DIS_s001
                                 + 1.00115 w_k1_FDX_s001
                                 + 0.688131 w_k1_IBM_s001
                                 + 1.18342 w_k1_INTC_s001
                                 + 0.643498 w_k1_KO_s001
                                 + 0.495458 w_k1_MAT_s001
                                 + 1.12244 w_k1_MSFT_s001
                                 + 0.821245 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.85845 w_k1_ADM_s001
                                 + 0.93507 w_k1_AXP_s001
                                 + 1.00013 w_k1_DIS_s001
                                 + 0.888641 w_k1_FDX_s001 + 1.707 w_k1_IBM_s001
                                 + 1.05443 w_k1_INTC_s001
                                 + 1.09472 w_k1_KO_s001 + 1.19653 w_k1_MAT_s001
                                 + 1.06989 w_k1_MSFT_s001 + 1.0318 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.0647 w_k1_ADM_s001
                                 + 1.36978 w_k1_AXP_s001 + 1.7961 w_k1_DIS_s001
                                 + 1.41026 w_k1_FDX_s001
                                 + 1.34488 w_k1_IBM_s001
                                 + 1.44778 w_k1_INTC_s001
                                 + 1.34669 w_k1_KO_s001
                                 + 0.993873 w_k1_MAT_s001
                                 + 1.31374 w_k1_MSFT_s001
                                 + 1.26499 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.15099 w_k1_ADM_s001
                                 + 1.06642 w_k1_AXP_s001
                                 + 1.31711 w_k1_DIS_s001
                                 + 1.11041 w_k1_FDX_s001
                                 + 1.60083 w_k1_IBM_s001
                                 + 1.39647 w_k1_INTC_s001
                                 + 1.11538 w_k1_KO_s001
                                 + 0.746741 w_k1_MAT_s001
                                 + 1.22465 w_k1_MSFT_s001
                                 + 1.04317 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.737507 w_k1_ADM_s001
                                 + 0.734546 w_k1_AXP_s001
                                 + 0.932694 w_k1_DIS_s001
                                 + 0.775049 w_k1_FDX_s001
                                 + 1.32067 w_k1_IBM_s001
                                 + 1.04046 w_k1_INTC_s001
                                 + 0.985917 w_k1_KO_s001
                                 + 0.884076 w_k1_MAT_s001
                                 + 0.864219 w_k1_MSFT_s001
                                 + 0.9703 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.796526 w_k1_ADM_s002
                                 + 0.668324 w_k1_AXP_s002
                                 + 0.718904 w_k1_DIS_s002
                                 + 0.769265 w_k1_FDX_s002
                                 + 0.766708 w_k1_IBM_s002
                                 + 0.728419 w_k1_INTC_s002
                                 + 0.887704 w_k1_KO_s002
                                 + 0.873303 w_k1_MAT_s002
                                 + 0.888859 w_k1_MSFT_s002
                                 + 0.861157 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.70688 w_k1_ADM_s002
                                 + 0.83146 w_k1_AXP_s002
                                 + 0.608613 w_k1_DIS_s002
                                 + 0.783554 w_k1_FDX_s002
                                 + 1.06805 w_k1_IBM_s002
                                 + 0.824056 w_k1_INTC_s002
                                 + 0.779926 w_k1_KO_s002
                                 + 1.26913 w_k1_MAT_s002
                                 + 0.949094 w_k1_MSFT_s002
                                 + 0.800577 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.28606 w_k1_ADM_s002
                                 + 1.4387 w_k1_AXP_s002 + 1.60589 w_k1_DIS_s002
                                 + 1.38403 w_k1_FDX_s002 + 0.9858 w_k1_IBM_s002
                                 + 0.967147 w_k1_INTC_s002
                                 + 1.05789 w_k1_KO_s002 + 1.40909 w_k1_MAT_s002
                                 + 1.32381 w_k1_MSFT_s002
                                 + 1.13694 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.02251 w_k1_ADM_s002
                                 + 1.0131 w_k1_AXP_s002 + 1.4389 w_k1_DIS_s002
                                 + 1.07252 w_k1_FDX_s002
                                 + 1.68603 w_k1_IBM_s002
                                 + 1.00574 w_k1_INTC_s002
                                 + 1.50167 w_k1_KO_s002 + 1.76267 w_k1_MAT_s002
                                 + 1.26722 w_k1_MSFT_s002
                                 + 1.07481 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.23563 w_k1_ADM_s002
                                 + 1.02492 w_k1_AXP_s002 + 1.4864 w_k1_DIS_s002
                                 + 1.14657 w_k1_FDX_s002
                                 + 1.19441 w_k1_IBM_s002
                                 + 1.09689 w_k1_INTC_s002
                                 + 1.33644 w_k1_KO_s002 + 1.34347 w_k1_MAT_s002
                                 + 1.40815 w_k1_MSFT_s002
                                 + 1.20241 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.1944 w_k1_ADM_s002
                                 + 0.93177 w_k1_AXP_s002
                                 + 0.89596 w_k1_DIS_s002
                                 + 1.04404 w_k1_FDX_s002
                                 + 0.835121 w_k1_IBM_s002
                                 + 0.926693 w_k1_INTC_s002
                                 + 0.884886 w_k1_KO_s002
                                 + 0.86756 w_k1_MAT_s002
                                 + 1.14591 w_k1_MSFT_s002
                                 + 1.09932 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.876232 w_k1_ADM_s002
                                 + 0.90929 w_k1_AXP_s002
                                 + 1.00761 w_k1_DIS_s002
                                 + 0.941372 w_k1_FDX_s002
                                 + 1.35016 w_k1_IBM_s002
                                 + 0.927913 w_k1_INTC_s002
                                 + 1.09523 w_k1_KO_s002 + 1.50524 w_k1_MAT_s002
                                 + 1.06183 w_k1_MSFT_s002
                                 + 0.922316 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.806229 w_k1_ADM_s002
                                 + 1.33219 w_k1_AXP_s002
                                 + 1.18146 w_k1_DIS_s002 + 1.2828 w_k1_FDX_s002
                                 + 1.23483 w_k1_IBM_s002
                                 + 1.2348 w_k1_INTC_s002 + 1.14837 w_k1_KO_s002
                                 + 1.69796 w_k1_MAT_s002
                                 + 1.30517 w_k1_MSFT_s002
                                 + 1.20086 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.28569 w_k1_ADM_s003
                                 + 1.51558 w_k1_AXP_s003
                                 + 1.28551 w_k1_DIS_s003
                                 + 1.19804 w_k1_FDX_s003
                                 + 1.19696 w_k1_IBM_s003
                                 + 1.17151 w_k1_INTC_s003
                                 + 1.23574 w_k1_KO_s003 + 1.62147 w_k1_MAT_s003
                                 + 1.2586 w_k1_MSFT_s003 + 1.03397 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.17348 w_k1_ADM_s003
                                 + 0.977441 w_k1_AXP_s003
                                 + 0.954286 w_k1_DIS_s003
                                 + 0.947525 w_k1_FDX_s003
                                 + 0.706606 w_k1_IBM_s003
                                 + 1.03542 w_k1_INTC_s003
                                 + 0.81642 w_k1_KO_s003
                                 + 0.558771 w_k1_MAT_s003
                                 + 0.966773 w_k1_MSFT_s003
                                 + 0.880482 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.07057 w_k1_ADM_s003
                                 + 1.06768 w_k1_AXP_s003
                                 + 1.32973 w_k1_DIS_s003
                                 + 1.00579 w_k1_FDX_s003
                                 + 1.45321 w_k1_IBM_s003
                                 + 1.16113 w_k1_INTC_s003
                                 + 1.23734 w_k1_KO_s003 + 1.00074 w_k1_MAT_s003
                                 + 1.13797 w_k1_MSFT_s003
                                 + 1.03504 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.927426 w_k1_ADM_s003
                                 + 0.751105 w_k1_AXP_s003
                                 + 0.939967 w_k1_DIS_s003
                                 + 0.81172 w_k1_FDX_s003
                                 + 0.999209 w_k1_IBM_s003
                                 + 0.975001 w_k1_INTC_s003
                                 + 0.978248 w_k1_KO_s003
                                 + 0.825454 w_k1_MAT_s003
                                 + 0.942046 w_k1_MSFT_s003
                                 + 0.880297 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.27663 w_k1_ADM_s003
                                 + 1.15851 w_k1_AXP_s003
                                 + 1.04267 w_k1_DIS_s003
                                 + 1.11781 w_k1_FDX_s003
                                 + 0.826996 w_k1_IBM_s003
                                 + 1.27669 w_k1_INTC_s003
                                 + 0.949606 w_k1_KO_s003
                                 + 0.833012 w_k1_MAT_s003
                                 + 1.17065 w_k1_MSFT_s003 + 1.081 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.813753 w_k1_ADM_s003
                                 + 0.693582 w_k1_AXP_s003
                                 + 0.579034 w_k1_DIS_s003
                                 + 0.708979 w_k1_FDX_s003
                                 + 0.86352 w_k1_IBM_s003
                                 + 0.72546 w_k1_INTC_s003 + 0.6972 w_k1_KO_s003
                                 + 0.924805 w_k1_MAT_s003
                                 + 0.695426 w_k1_MSFT_s003
                                 + 0.905885 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.685338 w_k1_ADM_s003
                                 + 0.763688 w_k1_AXP_s003
                                 + 0.659495 w_k1_DIS_s003
                                 + 0.815028 w_k1_FDX_s003
                                 + 0.757208 w_k1_IBM_s003
                                 + 1.00191 w_k1_INTC_s003
                                 + 0.672715 w_k1_KO_s003
                                 + 0.604292 w_k1_MAT_s003
                                 + 0.775115 w_k1_MSFT_s003
                                 + 0.755233 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.06344 w_k1_ADM_s003
                                 + 1.24331 w_k1_AXP_s003
                                 + 0.750718 w_k1_DIS_s003
                                 + 0.922647 w_k1_FDX_s003
                                 + 1.0008 w_k1_IBM_s003
                                 + 0.869396 w_k1_INTC_s003
                                 + 0.766732 w_k1_KO_s003
                                 + 1.18514 w_k1_MAT_s003
                                 + 0.705023 w_k1_MSFT_s003
                                 + 0.949447 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.700873 w_k1_ADM_s004
                                 + 0.709612 w_k1_AXP_s004
                                 + 0.523788 w_k1_DIS_s004
                                 + 0.601057 w_k1_FDX_s004
                                 + 0.964806 w_k1_IBM_s004
                                 + 0.690309 w_k1_INTC_s004
                                 + 0.800081 w_k1_KO_s004
                                 + 1.20148 w_k1_MAT_s004
                                 + 0.70214 w_k1_MSFT_s004
                                 + 0.822204 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.06285 w_k1_ADM_s004
                                 + 1.09326 w_k1_AXP_s004
                                 + 0.994255 w_k1_DIS_s004
                                 + 0.924527 w_k1_FDX_s004
                                 + 1.28847 w_k1_IBM_s004
                                 + 1.20192 w_k1_INTC_s004
                                 + 1.09119 w_k1_KO_s004 + 1.05889 w_k1_MAT_s004
                                 + 0.927555 w_k1_MSFT_s004
                                 + 0.990438 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.9442 w_k1_ADM_s004
                                 + 1.03513 w_k1_AXP_s004
                                 + 0.728993 w_k1_DIS_s004
                                 + 0.921529 w_k1_FDX_s004
                                 + 0.742355 w_k1_IBM_s004
                                 + 1.07664 w_k1_INTC_s004
                                 + 0.803233 w_k1_KO_s004
                                 + 0.781326 w_k1_MAT_s004
                                 + 0.896023 w_k1_MSFT_s004
                                 + 0.800394 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.04781 w_k1_ADM_s004
                                 + 1.15252 w_k1_AXP_s004
                                 + 0.789016 w_k1_DIS_s004
                                 + 0.939742 w_k1_FDX_s004
                                 + 0.690841 w_k1_IBM_s004
                                 + 0.944916 w_k1_INTC_s004
                                 + 1.00317 w_k1_KO_s004 + 1.08721 w_k1_MAT_s004
                                 + 0.927573 w_k1_MSFT_s004
                                 + 0.994815 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.696247 w_k1_ADM_s004
                                 + 0.74522 w_k1_AXP_s004
                                 + 0.745435 w_k1_DIS_s004
                                 + 0.744189 w_k1_FDX_s004
                                 + 1.0388 w_k1_IBM_s004
                                 + 1.01074 w_k1_INTC_s004
                                 + 0.978918 w_k1_KO_s004
                                 + 1.03206 w_k1_MAT_s004
                                 + 0.802435 w_k1_MSFT_s004
                                 + 0.959649 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.2104 w_k1_ADM_s004
                                 + 1.57067 w_k1_AXP_s004
                                 + 1.07447 w_k1_DIS_s004
                                 + 1.29896 w_k1_FDX_s004
                                 + 0.691488 w_k1_IBM_s004
                                 + 1.30998 w_k1_INTC_s004
                                 + 0.929288 w_k1_KO_s004
                                 + 0.805756 w_k1_MAT_s004
                                 + 1.11656 w_k1_MSFT_s004 + 1.1674 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.06905 w_k1_ADM_s004
                                 + 1.34018 w_k1_AXP_s004
                                 + 1.25498 w_k1_DIS_s004
                                 + 1.18865 w_k1_FDX_s004
                                 + 0.885145 w_k1_IBM_s004
                                 + 1.11293 w_k1_INTC_s004
                                 + 1.13389 w_k1_KO_s004
                                 + 0.964966 w_k1_MAT_s004
                                 + 1.22377 w_k1_MSFT_s004
                                 + 1.02691 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.46115 w_k1_ADM_s004
                                 + 1.52187 w_k1_AXP_s004
                                 + 1.66343 w_k1_DIS_s004
                                 + 1.33422 w_k1_FDX_s004
                                 + 1.37166 w_k1_IBM_s004
                                 + 1.34563 w_k1_INTC_s004
                                 + 1.55725 w_k1_KO_s004 + 1.48853 w_k1_MAT_s004
                                 + 1.21892 w_k1_MSFT_s004
                                 + 1.32249 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.800428 w_k1_ADM_s005
                                 + 0.854308 w_k1_AXP_s005
                                 + 0.700332 w_k1_DIS_s005
                                 + 0.749539 w_k1_FDX_s005
                                 + 1.13411 w_k1_IBM_s005
                                 + 0.798372 w_k1_INTC_s005
                                 + 0.956455 w_k1_KO_s005
                                 + 0.937015 w_k1_MAT_s005
                                 + 0.75131 w_k1_MSFT_s005
                                 + 1.01719 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.1393 w_k1_ADM_s005
                                 + 1.13991 w_k1_AXP_s005
                                 + 1.10663 w_k1_DIS_s005 + 1.1155 w_k1_FDX_s005
                                 + 0.97103 w_k1_IBM_s005
                                 + 1.25371 w_k1_INTC_s005
                                 + 1.03837 w_k1_KO_s005 + 1.07322 w_k1_MAT_s005
                                 + 1.19033 w_k1_MSFT_s005
                                 + 1.05151 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.0525 w_k1_ADM_s005
                                 + 0.853568 w_k1_AXP_s005
                                 + 0.940858 w_k1_DIS_s005
                                 + 1.04898 w_k1_FDX_s005
                                 + 0.729576 w_k1_IBM_s005
                                 + 1.15083 w_k1_INTC_s005
                                 + 0.797673 w_k1_KO_s005
                                 + 0.569312 w_k1_MAT_s005
                                 + 1.08359 w_k1_MSFT_s005
                                 + 0.99942 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.32164 w_k1_ADM_s005
                                 + 1.56091 w_k1_AXP_s005
                                 + 1.54947 w_k1_DIS_s005
                                 + 1.36595 w_k1_FDX_s005
                                 + 1.26396 w_k1_IBM_s005
                                 + 1.32501 w_k1_INTC_s005 + 1.3696 w_k1_KO_s005
                                 + 1.55852 w_k1_MAT_s005
                                 + 1.32014 w_k1_MSFT_s005
                                 + 1.43503 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.28723 w_k1_ADM_s005
                                 + 1.53449 w_k1_AXP_s005
                                 + 0.884521 w_k1_DIS_s005
                                 + 1.10408 w_k1_FDX_s005
                                 + 0.988729 w_k1_IBM_s005
                                 + 0.966258 w_k1_INTC_s005
                                 + 0.852609 w_k1_KO_s005
                                 + 1.53346 w_k1_MAT_s005
                                 + 0.888973 w_k1_MSFT_s005
                                 + 1.15915 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.01439 w_k1_ADM_s005
                                 + 1.25108 w_k1_AXP_s005
                                 + 1.08009 w_k1_DIS_s005
                                 + 0.990271 w_k1_FDX_s005
                                 + 1.37011 w_k1_IBM_s005
                                 + 1.24812 w_k1_INTC_s005
                                 + 1.21753 w_k1_KO_s005 + 1.44407 w_k1_MAT_s005
                                 + 0.968858 w_k1_MSFT_s005
                                 + 1.1554 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.877338 w_k1_ADM_s005
                                 + 0.779123 w_k1_AXP_s005
                                 + 0.681104 w_k1_DIS_s005
                                 + 0.810896 w_k1_FDX_s005
                                 + 0.7622 w_k1_IBM_s005
                                 + 0.82738 w_k1_INTC_s005
                                 + 0.737981 w_k1_KO_s005
                                 + 0.62361 w_k1_MAT_s005
                                 + 0.919697 w_k1_MSFT_s005
                                 + 0.810027 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.08866 w_k1_ADM_s005
                                 + 1.2272 w_k1_AXP_s005 + 1.33284 w_k1_DIS_s005
                                 + 1.34153 w_k1_FDX_s005
                                 + 0.952725 w_k1_IBM_s005
                                 + 1.41187 w_k1_INTC_s005
                                 + 1.19281 w_k1_KO_s005 + 1.09672 w_k1_MAT_s005
                                 + 1.28733 w_k1_MSFT_s005
                                 + 1.22734 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.759556 w_k1_ADM_s006
                                 + 0.757778 w_k1_AXP_s006
                                 + 0.649275 w_k1_DIS_s006
                                 + 0.754519 w_k1_FDX_s006
                                 + 0.985344 w_k1_IBM_s006
                                 + 0.998214 w_k1_INTC_s006
                                 + 0.902539 w_k1_KO_s006
                                 + 1.01097 w_k1_MAT_s006
                                 + 0.832544 w_k1_MSFT_s006
                                 + 0.819619 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.07367 w_k1_ADM_s006
                                 + 1.36104 w_k1_AXP_s006
                                 + 1.39635 w_k1_DIS_s006 + 1.1554 w_k1_FDX_s006
                                 + 1.10765 w_k1_IBM_s006
                                 + 1.16546 w_k1_INTC_s006
                                 + 1.31753 w_k1_KO_s006 + 1.55934 w_k1_MAT_s006
                                 + 1.21912 w_k1_MSFT_s006
                                 + 1.07335 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.772023 w_k1_ADM_s006
                                 + 0.771079 w_k1_AXP_s006
                                 + 0.948063 w_k1_DIS_s006
                                 + 0.90696 w_k1_FDX_s006
                                 + 0.694458 w_k1_IBM_s006
                                 + 1.01172 w_k1_INTC_s006
                                 + 0.903464 w_k1_KO_s006
                                 + 0.573654 w_k1_MAT_s006
                                 + 0.898467 w_k1_MSFT_s006
                                 + 0.905766 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.900272 w_k1_ADM_s006
                                 + 1.31636 w_k1_AXP_s006 + 1.4961 w_k1_DIS_s006
                                 + 1.1832 w_k1_FDX_s006 + 1.6218 w_k1_IBM_s006
                                 + 1.20449 w_k1_INTC_s006
                                 + 1.50824 w_k1_KO_s006 + 2.14089 w_k1_MAT_s006
                                 + 1.06838 w_k1_MSFT_s006
                                 + 1.28652 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.01644 w_k1_ADM_s006
                                 + 1.04419 w_k1_AXP_s006
                                 + 0.977684 w_k1_DIS_s006
                                 + 0.872734 w_k1_FDX_s006
                                 + 1.37522 w_k1_IBM_s006
                                 + 0.96224 w_k1_INTC_s006
                                 + 1.22277 w_k1_KO_s006 + 1.72972 w_k1_MAT_s006
                                 + 0.889631 w_k1_MSFT_s006
                                 + 1.14284 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.01115 w_k1_ADM_s006
                                 + 1.57443 w_k1_AXP_s006
                                 + 1.45869 w_k1_DIS_s006
                                 + 1.37533 w_k1_FDX_s006
                                 + 0.974305 w_k1_IBM_s006
                                 + 1.4943 w_k1_INTC_s006 + 1.09615 w_k1_KO_s006
                                 + 1.49214 w_k1_MAT_s006
                                 + 1.35232 w_k1_MSFT_s006
                                 + 1.22798 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.73438 w_k1_ADM_s006
                                 + 0.828463 w_k1_AXP_s006
                                 + 0.869053 w_k1_DIS_s006
                                 + 0.922119 w_k1_FDX_s006
                                 + 0.90678 w_k1_IBM_s006
                                 + 0.895941 w_k1_INTC_s006
                                 + 0.88017 w_k1_KO_s006 + 1.0502 w_k1_MAT_s006
                                 + 0.939073 w_k1_MSFT_s006
                                 + 0.960454 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.994407 w_k1_ADM_s006
                                 + 1.28149 w_k1_AXP_s006
                                 + 1.02007 w_k1_DIS_s006
                                 + 1.14255 w_k1_FDX_s006
                                 + 0.732326 w_k1_IBM_s006
                                 + 1.12824 w_k1_INTC_s006
                                 + 0.861715 w_k1_KO_s006
                                 + 1.13645 w_k1_MAT_s006
                                 + 1.11528 w_k1_MSFT_s006
                                 + 0.971491 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.38519 w_k1_ADM_s007
                                 + 1.44569 w_k1_AXP_s007
                                 + 1.69824 w_k1_DIS_s007
                                 + 1.37665 w_k1_FDX_s007
                                 + 1.08585 w_k1_IBM_s007
                                 + 1.24541 w_k1_INTC_s007
                                 + 1.32228 w_k1_KO_s007 + 1.2231 w_k1_MAT_s007
                                 + 1.276 w_k1_MSFT_s007 + 1.31805 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.17951 w_k1_ADM_s007
                                 + 0.964295 w_k1_AXP_s007
                                 + 1.05089 w_k1_DIS_s007
                                 + 1.04269 w_k1_FDX_s007
                                 + 1.10969 w_k1_IBM_s007
                                 + 0.986996 w_k1_INTC_s007
                                 + 1.11713 w_k1_KO_s007 + 1.04555 w_k1_MAT_s007
                                 + 1.12367 w_k1_MSFT_s007
                                 + 1.21174 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.715755 w_k1_ADM_s007
                                 + 0.741636 w_k1_AXP_s007
                                 + 0.786164 w_k1_DIS_s007
                                 + 0.708861 w_k1_FDX_s007
                                 + 1.05776 w_k1_IBM_s007
                                 + 0.889177 w_k1_INTC_s007
                                 + 0.935181 w_k1_KO_s007
                                 + 1.01125 w_k1_MAT_s007
                                 + 0.856425 w_k1_MSFT_s007
                                 + 0.911918 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.59594 w_k1_ADM_s007
                                 + 1.06531 w_k1_AXP_s007
                                 + 1.40764 w_k1_DIS_s007
                                 + 1.12099 w_k1_FDX_s007
                                 + 1.80357 w_k1_IBM_s007
                                 + 1.15787 w_k1_INTC_s007
                                 + 1.47078 w_k1_KO_s007 + 1.85549 w_k1_MAT_s007
                                 + 1.33555 w_k1_MSFT_s007
                                 + 1.27527 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.981057 w_k1_ADM_s007
                                 + 1.11909 w_k1_AXP_s007
                                 + 0.850556 w_k1_DIS_s007
                                 + 0.880535 w_k1_FDX_s007
                                 + 0.946151 w_k1_IBM_s007
                                 + 0.885216 w_k1_INTC_s007
                                 + 0.893757 w_k1_KO_s007
                                 + 1.08839 w_k1_MAT_s007
                                 + 0.897281 w_k1_MSFT_s007
                                 + 1.09951 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.979187 w_k1_ADM_s007
                                 + 0.914702 w_k1_AXP_s007
                                 + 0.848328 w_k1_DIS_s007
                                 + 0.95932 w_k1_FDX_s007
                                 + 0.653752 w_k1_IBM_s007
                                 + 0.976441 w_k1_INTC_s007
                                 + 0.754849 w_k1_KO_s007
                                 + 0.578628 w_k1_MAT_s007
                                 + 1.02541 w_k1_MSFT_s007
                                 + 0.946832 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.07972 w_k1_ADM_s007
                                 + 0.839306 w_k1_AXP_s007
                                 + 0.966292 w_k1_DIS_s007
                                 + 1.06352 w_k1_FDX_s007
                                 + 0.684179 w_k1_IBM_s007
                                 + 1.06997 w_k1_INTC_s007
                                 + 0.920801 w_k1_KO_s007
                                 + 0.682196 w_k1_MAT_s007
                                 + 1.21061 w_k1_MSFT_s007
                                 + 1.05707 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.697525 w_k1_ADM_s007
                                 + 0.654845 w_k1_AXP_s007
                                 + 0.631185 w_k1_DIS_s007
                                 + 0.747827 w_k1_FDX_s007
                                 + 0.657904 w_k1_IBM_s007
                                 + 0.989086 w_k1_INTC_s007
                                 + 0.646337 w_k1_KO_s007
                                 + 0.660847 w_k1_MAT_s007
                                 + 0.851769 w_k1_MSFT_s007
                                 + 0.779515 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.05395 w_k1_ADM_s010
                                 + 1.23065 w_k1_AXP_s010 + 1.0806 w_k1_DIS_s010
                                 + 1.00782 w_k1_FDX_s010
                                 + 1.10335 w_k1_IBM_s010
                                 + 1.12456 w_k1_INTC_s010
                                 + 1.05784 w_k1_KO_s010 + 1.17573 w_k1_MAT_s010
                                 + 1.07915 w_k1_MSFT_s010
                                 + 0.977564 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.15433 w_k1_ADM_s010
                                 + 1.47624 w_k1_AXP_s010
                                 + 1.82832 w_k1_DIS_s010
                                 + 1.43752 w_k1_FDX_s010
                                 + 1.08583 w_k1_IBM_s010
                                 + 1.33835 w_k1_INTC_s010
                                 + 1.35135 w_k1_KO_s010 + 1.55475 w_k1_MAT_s010
                                 + 1.50233 w_k1_MSFT_s010
                                 + 1.15963 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.03345 w_k1_ADM_s010
                                 + 0.760474 w_k1_AXP_s010
                                 + 0.886442 w_k1_DIS_s010
                                 + 0.778212 w_k1_FDX_s010
                                 + 1.27647 w_k1_IBM_s010
                                 + 0.737712 w_k1_INTC_s010
                                 + 0.971336 w_k1_KO_s010
                                 + 1.05599 w_k1_MAT_s010
                                 + 0.76263 w_k1_MSFT_s010
                                 + 0.93279 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.875881 w_k1_ADM_s010
                                 + 0.721469 w_k1_AXP_s010
                                 + 0.675754 w_k1_DIS_s010
                                 + 0.805725 w_k1_FDX_s010
                                 + 0.582568 w_k1_IBM_s010
                                 + 0.969853 w_k1_INTC_s010
                                 + 0.656226 w_k1_KO_s010
                                 + 0.758862 w_k1_MAT_s010
                                 + 0.965377 w_k1_MSFT_s010
                                 + 0.720351 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.06966 w_k1_ADM_s010
                                 + 0.950408 w_k1_AXP_s010
                                 + 1.20803 w_k1_DIS_s010
                                 + 1.14296 w_k1_FDX_s010
                                 + 0.819767 w_k1_IBM_s010
                                 + 1.17199 w_k1_INTC_s010
                                 + 0.994151 w_k1_KO_s010
                                 + 0.735338 w_k1_MAT_s010
                                 + 1.11955 w_k1_MSFT_s010
                                 + 1.00737 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.992056 w_k1_ADM_s010
                                 + 1.63314 w_k1_AXP_s010
                                 + 1.62364 w_k1_DIS_s010
                                 + 1.40587 w_k1_FDX_s010
                                 + 1.57481 w_k1_IBM_s010
                                 + 1.02941 w_k1_INTC_s010
                                 + 1.32456 w_k1_KO_s010 + 2.02805 w_k1_MAT_s010
                                 + 1.20171 w_k1_MSFT_s010
                                 + 1.08398 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.632573 w_k1_ADM_s010
                                 + 0.587038 w_k1_AXP_s010
                                 + 0.437913 w_k1_DIS_s010
                                 + 0.65186 w_k1_FDX_s010
                                 + 0.741911 w_k1_IBM_s010
                                 + 0.83001 w_k1_INTC_s010
                                 + 0.578716 w_k1_KO_s010
                                 + 0.648004 w_k1_MAT_s010
                                 + 0.63744 w_k1_MSFT_s010
                                 + 0.669281 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.842676 w_k1_ADM_s010
                                 + 0.910875 w_k1_AXP_s010
                                 + 0.772854 w_k1_DIS_s010
                                 + 0.892862 w_k1_FDX_s010
                                 + 0.886771 w_k1_IBM_s010
                                 + 0.732388 w_k1_INTC_s010
                                 + 0.822901 w_k1_KO_s010
                                 + 0.974293 w_k1_MAT_s010
                                 + 0.932425 w_k1_MSFT_s010
                                 + 0.851892 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.11545 w_k1_ADM_s011
                                 + 1.03433 w_k1_AXP_s011
                                 + 1.21429 w_k1_DIS_s011
                                 + 1.05626 w_k1_FDX_s011
                                 + 1.25945 w_k1_IBM_s011
                                 + 1.03917 w_k1_INTC_s011
                                 + 1.21721 w_k1_KO_s011 + 1.10235 w_k1_MAT_s011
                                 + 1.15572 w_k1_MSFT_s011
                                 + 1.04863 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.12397 w_k1_ADM_s011
                                 + 1.43406 w_k1_AXP_s011
                                 + 1.26159 w_k1_DIS_s011
                                 + 1.29984 w_k1_FDX_s011
                                 + 1.08901 w_k1_IBM_s011
                                 + 1.14378 w_k1_INTC_s011
                                 + 1.01532 w_k1_KO_s011 + 1.35473 w_k1_MAT_s011
                                 + 1.2909 w_k1_MSFT_s011 + 1.08888 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.32819 w_k1_ADM_s011
                                 + 1.21291 w_k1_AXP_s011
                                 + 1.62959 w_k1_DIS_s011
                                 + 1.39889 w_k1_FDX_s011
                                 + 0.877857 w_k1_IBM_s011
                                 + 1.36388 w_k1_INTC_s011
                                 + 1.20005 w_k1_KO_s011
                                 + 0.838687 w_k1_MAT_s011
                                 + 1.3064 w_k1_MSFT_s011 + 1.18528 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.13049 w_k1_ADM_s011
                                 + 1.47257 w_k1_AXP_s011
                                 + 1.44435 w_k1_DIS_s011 + 1.1283 w_k1_FDX_s011
                                 + 1.54144 w_k1_IBM_s011
                                 + 1.31884 w_k1_INTC_s011 + 1.5107 w_k1_KO_s011
                                 + 1.54159 w_k1_MAT_s011
                                 + 1.17083 w_k1_MSFT_s011
                                 + 1.15905 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.02578 w_k1_ADM_s011
                                 + 0.795118 w_k1_AXP_s011
                                 + 0.746294 w_k1_DIS_s011
                                 + 0.951721 w_k1_FDX_s011
                                 + 0.702193 w_k1_IBM_s011
                                 + 0.840895 w_k1_INTC_s011
                                 + 0.780426 w_k1_KO_s011
                                 + 0.592528 w_k1_MAT_s011
                                 + 0.980571 w_k1_MSFT_s011
                                 + 0.848843 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.68457 w_k1_ADM_s011
                                 + 0.615031 w_k1_AXP_s011
                                 + 0.490994 w_k1_DIS_s011
                                 + 0.639445 w_k1_FDX_s011
                                 + 0.717268 w_k1_IBM_s011
                                 + 0.647591 w_k1_INTC_s011
                                 + 0.575312 w_k1_KO_s011
                                 + 0.734659 w_k1_MAT_s011
                                 + 0.640426 w_k1_MSFT_s011
                                 + 0.694442 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.761789 w_k1_ADM_s011
                                 + 0.715404 w_k1_AXP_s011
                                 + 0.665765 w_k1_DIS_s011
                                 + 0.756773 w_k1_FDX_s011
                                 + 1.2355 w_k1_IBM_s011
                                 + 0.824033 w_k1_INTC_s011
                                 + 0.865929 w_k1_KO_s011
                                 + 1.05756 w_k1_MAT_s011
                                 + 0.805413 w_k1_MSFT_s011
                                 + 0.789027 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.13229 w_k1_ADM_s011
                                 + 0.985066 w_k1_AXP_s011
                                 + 0.967736 w_k1_DIS_s011
                                 + 0.99131 w_k1_FDX_s011
                                 + 0.960546 w_k1_IBM_s011
                                 + 1.01025 w_k1_INTC_s011
                                 + 0.918337 w_k1_KO_s011
                                 + 0.946516 w_k1_MAT_s011
                                 + 1.03265 w_k1_MSFT_s011
                                 + 0.944604 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.565832 w_k1_ADM_s012
                                 + 0.607836 w_k1_AXP_s012
                                 + 0.693662 w_k1_DIS_s012
                                 + 0.727355 w_k1_FDX_s012
                                 + 1.18396 w_k1_IBM_s012
                                 + 0.984513 w_k1_INTC_s012
                                 + 1.00059 w_k1_KO_s012 + 1.02006 w_k1_MAT_s012
                                 + 0.857791 w_k1_MSFT_s012
                                 + 0.782604 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.911722 w_k1_ADM_s012
                                 + 0.795518 w_k1_AXP_s012
                                 + 0.671294 w_k1_DIS_s012
                                 + 0.848816 w_k1_FDX_s012
                                 + 0.83704 w_k1_IBM_s012
                                 + 1.04729 w_k1_INTC_s012
                                 + 0.892897 w_k1_KO_s012
                                 + 0.664314 w_k1_MAT_s012
                                 + 0.997442 w_k1_MSFT_s012
                                 + 0.741215 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.09262 w_k1_ADM_s012
                                 + 1.01729 w_k1_AXP_s012
                                 + 0.889668 w_k1_DIS_s012
                                 + 0.951979 w_k1_FDX_s012
                                 + 1.17818 w_k1_IBM_s012
                                 + 0.867292 w_k1_INTC_s012
                                 + 0.996336 w_k1_KO_s012
                                 + 1.24054 w_k1_MAT_s012
                                 + 1.04879 w_k1_MSFT_s012
                                 + 1.06058 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.32757 w_k1_ADM_s012
                                 + 1.46757 w_k1_AXP_s012
                                 + 1.47586 w_k1_DIS_s012
                                 + 1.27713 w_k1_FDX_s012
                                 + 1.43921 w_k1_IBM_s012
                                 + 1.09687 w_k1_INTC_s012
                                 + 1.53332 w_k1_KO_s012 + 1.31793 w_k1_MAT_s012
                                 + 1.19413 w_k1_MSFT_s012
                                 + 1.17754 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.00905 w_k1_ADM_s012
                                 + 1.08033 w_k1_AXP_s012
                                 + 0.914969 w_k1_DIS_s012
                                 + 1.04575 w_k1_FDX_s012
                                 + 0.691728 w_k1_IBM_s012
                                 + 1.07225 w_k1_INTC_s012
                                 + 0.948222 w_k1_KO_s012
                                 + 0.766823 w_k1_MAT_s012
                                 + 0.923751 w_k1_MSFT_s012
                                 + 0.892904 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.703072 w_k1_ADM_s012
                                 + 0.71533 w_k1_AXP_s012
                                 + 0.464307 w_k1_DIS_s012
                                 + 0.758508 w_k1_FDX_s012
                                 + 0.67184 w_k1_IBM_s012
                                 + 0.901068 w_k1_INTC_s012
                                 + 0.691686 w_k1_KO_s012
                                 + 0.545653 w_k1_MAT_s012
                                 + 0.774483 w_k1_MSFT_s012
                                 + 0.709162 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.21332 w_k1_ADM_s012
                                 + 1.27754 w_k1_AXP_s012
                                 + 1.01053 w_k1_DIS_s012 + 1.2459 w_k1_FDX_s012
                                 + 0.959298 w_k1_IBM_s012
                                 + 1.10645 w_k1_INTC_s012
                                 + 0.916059 w_k1_KO_s012
                                 + 0.988477 w_k1_MAT_s012
                                 + 1.14115 w_k1_MSFT_s012
                                 + 0.866814 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.12147 w_k1_ADM_s012
                                 + 1.27408 w_k1_AXP_s012
                                 + 1.26823 w_k1_DIS_s012
                                 + 1.23504 w_k1_FDX_s012
                                 + 1.01858 w_k1_IBM_s012
                                 + 1.47044 w_k1_INTC_s012
                                 + 1.21701 w_k1_KO_s012
                                 + 0.817984 w_k1_MAT_s012
                                 + 1.29915 w_k1_MSFT_s012
                                 + 1.17076 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.02035 w_k1_ADM_s013
                                 + 0.709355 w_k1_AXP_s013
                                 + 0.702201 w_k1_DIS_s013
                                 + 0.684909 w_k1_FDX_s013
                                 + 0.994404 w_k1_IBM_s013
                                 + 0.737972 w_k1_INTC_s013
                                 + 0.945797 w_k1_KO_s013
                                 + 1.05153 w_k1_MAT_s013
                                 + 0.828527 w_k1_MSFT_s013
                                 + 0.843232 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.852145 w_k1_ADM_s013
                                 + 0.814861 w_k1_AXP_s013
                                 + 0.831849 w_k1_DIS_s013
                                 + 0.78039 w_k1_FDX_s013
                                 + 1.26657 w_k1_IBM_s013
                                 + 0.793716 w_k1_INTC_s013
                                 + 0.984791 w_k1_KO_s013
                                 + 0.749201 w_k1_MAT_s013
                                 + 0.860091 w_k1_MSFT_s013
                                 + 0.877963 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.24761 w_k1_ADM_s013
                                 + 1.12004 w_k1_AXP_s013
                                 + 1.24663 w_k1_DIS_s013 + 1.0438 w_k1_FDX_s013
                                 + 1.56184 w_k1_IBM_s013
                                 + 1.18377 w_k1_INTC_s013
                                 + 1.20414 w_k1_KO_s013 + 1.43615 w_k1_MAT_s013
                                 + 1.20609 w_k1_MSFT_s013
                                 + 1.28365 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.05395 w_k1_ADM_s013
                                 + 0.908728 w_k1_AXP_s013
                                 + 0.966472 w_k1_DIS_s013
                                 + 0.890786 w_k1_FDX_s013
                                 + 0.870504 w_k1_IBM_s013
                                 + 0.657096 w_k1_INTC_s013
                                 + 0.95785 w_k1_KO_s013 + 1.02435 w_k1_MAT_s013
                                 + 0.949183 w_k1_MSFT_s013
                                 + 1.01291 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15793 w_k1_ADM_s013
                                 + 1.11391 w_k1_AXP_s013
                                 + 1.13569 w_k1_DIS_s013
                                 + 1.08088 w_k1_FDX_s013
                                 + 1.10761 w_k1_IBM_s013
                                 + 1.03343 w_k1_INTC_s013
                                 + 1.09524 w_k1_KO_s013
                                 + 0.817243 w_k1_MAT_s013
                                 + 1.14253 w_k1_MSFT_s013
                                 + 1.10284 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.91366 w_k1_ADM_s013
                                 + 1.25152 w_k1_AXP_s013
                                 + 1.69132 w_k1_DIS_s013 + 1.3551 w_k1_FDX_s013
                                 + 1.13208 w_k1_IBM_s013
                                 + 1.1127 w_k1_INTC_s013 + 1.35492 w_k1_KO_s013
                                 + 1.21527 w_k1_MAT_s013
                                 + 1.53422 w_k1_MSFT_s013
                                 + 1.29424 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.840461 w_k1_ADM_s013
                                 + 0.5621 w_k1_AXP_s013 + 0.56592 w_k1_DIS_s013
                                 + 0.719812 w_k1_FDX_s013
                                 + 0.853221 w_k1_IBM_s013
                                 + 0.764575 w_k1_INTC_s013
                                 + 0.747622 w_k1_KO_s013
                                 + 0.779446 w_k1_MAT_s013
                                 + 0.758981 w_k1_MSFT_s013
                                 + 0.768146 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.31793 w_k1_ADM_s013
                                 + 0.899088 w_k1_AXP_s013
                                 + 0.868847 w_k1_DIS_s013
                                 + 0.973435 w_k1_FDX_s013
                                 + 0.732779 w_k1_IBM_s013
                                 + 0.699384 w_k1_INTC_s013
                                 + 0.770029 w_k1_KO_s013
                                 + 0.714146 w_k1_MAT_s013
                                 + 0.892394 w_k1_MSFT_s013
                                 + 0.894269 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.04927 w_k1_ADM_s014
                                 + 0.87436 w_k1_AXP_s014
                                 + 0.836739 w_k1_DIS_s014
                                 + 0.853255 w_k1_FDX_s014
                                 + 1.12444 w_k1_IBM_s014
                                 + 0.910043 w_k1_INTC_s014
                                 + 0.853943 w_k1_KO_s014
                                 + 1.21009 w_k1_MAT_s014
                                 + 0.886219 w_k1_MSFT_s014
                                 + 1.00754 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.758566 w_k1_ADM_s014
                                 + 0.607419 w_k1_AXP_s014
                                 + 0.447975 w_k1_DIS_s014
                                 + 0.567892 w_k1_FDX_s014
                                 + 0.761143 w_k1_IBM_s014
                                 + 0.512583 w_k1_INTC_s014
                                 + 0.560085 w_k1_KO_s014
                                 + 1.00161 w_k1_MAT_s014
                                 + 0.575861 w_k1_MSFT_s014
                                 + 0.692455 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.853411 w_k1_ADM_s014
                                 + 0.807919 w_k1_AXP_s014
                                 + 1.09294 w_k1_DIS_s014
                                 + 0.823505 w_k1_FDX_s014
                                 + 1.30262 w_k1_IBM_s014
                                 + 0.844526 w_k1_INTC_s014
                                 + 1.07243 w_k1_KO_s014 + 1.20386 w_k1_MAT_s014
                                 + 1.0053 w_k1_MSFT_s014
                                 + 0.938065 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.906347 w_k1_ADM_s014
                                 + 0.662793 w_k1_AXP_s014
                                 + 0.630643 w_k1_DIS_s014
                                 + 0.755409 w_k1_FDX_s014
                                 + 0.757964 w_k1_IBM_s014
                                 + 0.654714 w_k1_INTC_s014
                                 + 0.665904 w_k1_KO_s014
                                 + 0.878523 w_k1_MAT_s014
                                 + 0.702773 w_k1_MSFT_s014
                                 + 0.825274 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.13482 w_k1_ADM_s014
                                 + 1.09171 w_k1_AXP_s014
                                 + 1.07224 w_k1_DIS_s014
                                 + 1.10869 w_k1_FDX_s014
                                 + 0.788481 w_k1_IBM_s014
                                 + 1.00077 w_k1_INTC_s014
                                 + 0.918771 w_k1_KO_s014
                                 + 1.07682 w_k1_MAT_s014
                                 + 1.05968 w_k1_MSFT_s014
                                 + 0.991399 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.30289 w_k1_ADM_s014
                                 + 1.14685 w_k1_AXP_s014
                                 + 1.62238 w_k1_DIS_s014
                                 + 1.11406 w_k1_FDX_s014
                                 + 1.52544 w_k1_IBM_s014
                                 + 0.933103 w_k1_INTC_s014
                                 + 1.37348 w_k1_KO_s014 + 1.82448 w_k1_MAT_s014
                                 + 1.11783 w_k1_MSFT_s014
                                 + 1.17297 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.48034 w_k1_ADM_s014
                                 + 1.62358 w_k1_AXP_s014
                                 + 1.70631 w_k1_DIS_s014
                                 + 1.53143 w_k1_FDX_s014
                                 + 1.10804 w_k1_IBM_s014
                                 + 1.51627 w_k1_INTC_s014
                                 + 1.19205 w_k1_KO_s014 + 1.50424 w_k1_MAT_s014
                                 + 1.36366 w_k1_MSFT_s014
                                 + 1.21888 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.06662 w_k1_ADM_s014
                                 + 0.99253 w_k1_AXP_s014
                                 + 1.06739 w_k1_DIS_s014
                                 + 0.969638 w_k1_FDX_s014
                                 + 1.0529 w_k1_IBM_s014
                                 + 0.807864 w_k1_INTC_s014
                                 + 0.885369 w_k1_KO_s014 + 1.1853 w_k1_MAT_s014
                                 + 0.893595 w_k1_MSFT_s014
                                 + 1.03112 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.702745 w_k1_ADM_s015
                                 + 0.751916 w_k1_AXP_s015
                                 + 0.683183 w_k1_DIS_s015
                                 + 0.770679 w_k1_FDX_s015
                                 + 1.01871 w_k1_IBM_s015
                                 + 0.791662 w_k1_INTC_s015
                                 + 0.920562 w_k1_KO_s015
                                 + 0.800591 w_k1_MAT_s015
                                 + 0.8008 w_k1_MSFT_s015
                                 + 0.879134 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.919332 w_k1_ADM_s015
                                 + 1.351 w_k1_AXP_s015 + 1.31674 w_k1_DIS_s015
                                 + 1.2442 w_k1_FDX_s015 + 1.48276 w_k1_IBM_s015
                                 + 1.23452 w_k1_INTC_s015
                                 + 1.24598 w_k1_KO_s015 + 1.68296 w_k1_MAT_s015
                                 + 1.21926 w_k1_MSFT_s015
                                 + 1.22062 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.10905 w_k1_ADM_s015
                                 + 0.922124 w_k1_AXP_s015
                                 + 0.987928 w_k1_DIS_s015
                                 + 1.02363 w_k1_FDX_s015
                                 + 0.828279 w_k1_IBM_s015
                                 + 0.890524 w_k1_INTC_s015
                                 + 0.799891 w_k1_KO_s015
                                 + 0.691581 w_k1_MAT_s015
                                 + 1.04448 w_k1_MSFT_s015
                                 + 1.04126 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.11728 w_k1_ADM_s015
                                 + 0.842511 w_k1_AXP_s015
                                 + 0.975816 w_k1_DIS_s015
                                 + 1.07492 w_k1_FDX_s015
                                 + 1.06077 w_k1_IBM_s015
                                 + 1.11985 w_k1_INTC_s015
                                 + 0.918712 w_k1_KO_s015
                                 + 0.695326 w_k1_MAT_s015
                                 + 1.05054 w_k1_MSFT_s015
                                 + 0.91171 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.908498 w_k1_ADM_s015
                                 + 0.967163 w_k1_AXP_s015
                                 + 0.810435 w_k1_DIS_s015
                                 + 1.04336 w_k1_FDX_s015
                                 + 0.905028 w_k1_IBM_s015
                                 + 1.05222 w_k1_INTC_s015
                                 + 0.797128 w_k1_KO_s015
                                 + 0.799929 w_k1_MAT_s015
                                 + 1.00408 w_k1_MSFT_s015
                                 + 0.952172 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.06795 w_k1_ADM_s015
                                 + 0.78532 w_k1_AXP_s015
                                 + 0.670366 w_k1_DIS_s015
                                 + 0.8712 w_k1_FDX_s015
                                 + 0.635755 w_k1_IBM_s015
                                 + 0.647936 w_k1_INTC_s015
                                 + 0.613537 w_k1_KO_s015
                                 + 0.573351 w_k1_MAT_s015
                                 + 0.769187 w_k1_MSFT_s015
                                 + 0.799335 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.4496 w_k1_ADM_s015
                                 + 1.11331 w_k1_AXP_s015
                                 + 1.37957 w_k1_DIS_s015 + 1.2821 w_k1_FDX_s015
                                 + 1.05719 w_k1_IBM_s015
                                 + 1.10454 w_k1_INTC_s015
                                 + 1.18378 w_k1_KO_s015
                                 + 0.987635 w_k1_MAT_s015
                                 + 1.29694 w_k1_MSFT_s015 + 1.1745 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.02943 w_k1_ADM_s015
                                 + 0.871496 w_k1_AXP_s015
                                 + 0.844279 w_k1_DIS_s015
                                 + 0.90194 w_k1_FDX_s015
                                 + 0.991115 w_k1_IBM_s015
                                 + 1.0351 w_k1_INTC_s015
                                 + 0.912823 w_k1_KO_s015
                                 + 0.791924 w_k1_MAT_s015
                                 + 0.91106 w_k1_MSFT_s015
                                 + 0.95955 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.767202 w_k1_ADM_s016
                                 + 0.81495 w_k1_AXP_s016
                                 + 0.889132 w_k1_DIS_s016
                                 + 0.894722 w_k1_FDX_s016
                                 + 0.908925 w_k1_IBM_s016
                                 + 1.06246 w_k1_INTC_s016
                                 + 0.770703 w_k1_KO_s016
                                 + 0.767316 w_k1_MAT_s016
                                 + 1.01528 w_k1_MSFT_s016
                                 + 0.850345 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.24189 w_k1_ADM_s016
                                 + 1.37249 w_k1_AXP_s016
                                 + 1.54518 w_k1_DIS_s016
                                 + 1.32128 w_k1_FDX_s016
                                 + 1.34804 w_k1_IBM_s016
                                 + 1.23781 w_k1_INTC_s016
                                 + 1.21914 w_k1_KO_s016 + 1.57221 w_k1_MAT_s016
                                 + 1.37784 w_k1_MSFT_s016
                                 + 1.03951 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.22156 w_k1_ADM_s016
                                 + 0.837318 w_k1_AXP_s016
                                 + 1.06179 w_k1_DIS_s016
                                 + 1.09827 w_k1_FDX_s016
                                 + 0.818049 w_k1_IBM_s016
                                 + 0.854173 w_k1_INTC_s016
                                 + 0.824456 w_k1_KO_s016
                                 + 0.772621 w_k1_MAT_s016
                                 + 1.07968 w_k1_MSFT_s016
                                 + 0.960066 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.663995 w_k1_ADM_s016
                                 + 0.694761 w_k1_AXP_s016
                                 + 0.550937 w_k1_DIS_s016
                                 + 0.648623 w_k1_FDX_s016
                                 + 0.917737 w_k1_IBM_s016
                                 + 0.71098 w_k1_INTC_s016
                                 + 0.661214 w_k1_KO_s016
                                 + 1.07396 w_k1_MAT_s016
                                 + 0.787611 w_k1_MSFT_s016
                                 + 0.78968 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.17587 w_k1_ADM_s016
                                 + 1.30181 w_k1_AXP_s016
                                 + 1.24572 w_k1_DIS_s016
                                 + 1.12647 w_k1_FDX_s016
                                 + 1.24903 w_k1_IBM_s016
                                 + 1.07715 w_k1_INTC_s016
                                 + 1.08123 w_k1_KO_s016 + 1.61396 w_k1_MAT_s016
                                 + 1.13306 w_k1_MSFT_s016
                                 + 1.14767 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.917705 w_k1_ADM_s016
                                 + 1.11518 w_k1_AXP_s016
                                 + 1.23692 w_k1_DIS_s016
                                 + 1.18126 w_k1_FDX_s016
                                 + 0.949614 w_k1_IBM_s016
                                 + 1.18719 w_k1_INTC_s016
                                 + 0.938549 w_k1_KO_s016
                                 + 1.08802 w_k1_MAT_s016
                                 + 1.19448 w_k1_MSFT_s016
                                 + 1.00629 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.870068 w_k1_ADM_s016
                                 + 0.83777 w_k1_AXP_s016
                                 + 0.609985 w_k1_DIS_s016
                                 + 0.850155 w_k1_FDX_s016
                                 + 0.60933 w_k1_IBM_s016
                                 + 0.767734 w_k1_INTC_s016
                                 + 0.578101 w_k1_KO_s016
                                 + 0.699447 w_k1_MAT_s016
                                 + 0.801456 w_k1_MSFT_s016
                                 + 0.810605 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.01263 w_k1_ADM_s016
                                 + 0.841281 w_k1_AXP_s016
                                 + 1.02567 w_k1_DIS_s016
                                 + 0.912723 w_k1_FDX_s016
                                 + 1.29367 w_k1_IBM_s016
                                 + 0.877664 w_k1_INTC_s016
                                 + 0.892064 w_k1_KO_s016
                                 + 1.17354 w_k1_MAT_s016
                                 + 0.881838 w_k1_MSFT_s016
                                 + 0.987346 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.07241 w_k1_ADM_s017
                                 + 1.28789 w_k1_AXP_s017
                                 + 1.07102 w_k1_DIS_s017 + 1.0301 w_k1_FDX_s017
                                 + 1.20007 w_k1_IBM_s017
                                 + 0.973951 w_k1_INTC_s017
                                 + 1.1081 w_k1_KO_s017 + 1.41613 w_k1_MAT_s017
                                 + 0.958204 w_k1_MSFT_s017
                                 + 1.15005 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.81134 w_k1_ADM_s017
                                 + 0.753978 w_k1_AXP_s017
                                 + 0.859069 w_k1_DIS_s017
                                 + 0.846154 w_k1_FDX_s017
                                 + 1.25235 w_k1_IBM_s017
                                 + 0.746133 w_k1_INTC_s017
                                 + 1.01044 w_k1_KO_s017 + 1.37383 w_k1_MAT_s017
                                 + 0.80382 w_k1_MSFT_s017
                                 + 0.895116 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.675176 w_k1_ADM_s017
                                 + 0.623205 w_k1_AXP_s017
                                 + 0.623621 w_k1_DIS_s017
                                 + 0.789367 w_k1_FDX_s017
                                 + 0.710528 w_k1_IBM_s017
                                 + 0.966638 w_k1_INTC_s017
                                 + 0.723179 w_k1_KO_s017
                                 + 0.704159 w_k1_MAT_s017
                                 + 0.736954 w_k1_MSFT_s017
                                 + 0.855425 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.712439 w_k1_ADM_s017
                                 + 0.791581 w_k1_AXP_s017
                                 + 0.525928 w_k1_DIS_s017
                                 + 0.689795 w_k1_FDX_s017
                                 + 0.935232 w_k1_IBM_s017
                                 + 0.680549 w_k1_INTC_s017
                                 + 0.69558 w_k1_KO_s017 + 0.85004 w_k1_MAT_s017
                                 + 0.637012 w_k1_MSFT_s017
                                 + 0.791428 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.03063 w_k1_ADM_s017
                                 + 0.969741 w_k1_AXP_s017
                                 + 0.697806 w_k1_DIS_s017
                                 + 0.93655 w_k1_FDX_s017
                                 + 0.915932 w_k1_IBM_s017
                                 + 0.895878 w_k1_INTC_s017
                                 + 0.764722 w_k1_KO_s017
                                 + 0.941592 w_k1_MAT_s017
                                 + 0.840685 w_k1_MSFT_s017
                                 + 0.827982 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.07589 w_k1_ADM_s017
                                 + 1.01946 w_k1_AXP_s017
                                 + 1.16747 w_k1_DIS_s017
                                 + 1.18726 w_k1_FDX_s017
                                 + 0.825427 w_k1_IBM_s017
                                 + 0.959055 w_k1_INTC_s017
                                 + 1.09287 w_k1_KO_s017 + 0.89354 w_k1_MAT_s017
                                 + 1.21227 w_k1_MSFT_s017
                                 + 1.10132 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.987678 w_k1_ADM_s017
                                 + 1.62233 w_k1_AXP_s017
                                 + 1.61266 w_k1_DIS_s017 + 1.3337 w_k1_FDX_s017
                                 + 1.50899 w_k1_IBM_s017
                                 + 1.5295 w_k1_INTC_s017 + 1.4453 w_k1_KO_s017
                                 + 1.70241 w_k1_MAT_s017
                                 + 1.31045 w_k1_MSFT_s017 + 1.2828 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.788857 w_k1_ADM_s017
                                 + 0.808095 w_k1_AXP_s017
                                 + 0.83102 w_k1_DIS_s017
                                 + 0.830362 w_k1_FDX_s017
                                 + 0.894563 w_k1_IBM_s017
                                 + 0.824514 w_k1_INTC_s017
                                 + 0.913905 w_k1_KO_s017
                                 + 1.14921 w_k1_MAT_s017
                                 + 0.831683 w_k1_MSFT_s017
                                 + 0.915605 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.833391 w_k1_ADM_s100
                                 + 0.773902 w_k1_AXP_s100
                                 + 0.894284 w_k1_DIS_s100
                                 + 0.949588 w_k1_FDX_s100
                                 + 0.648541 w_k1_IBM_s100
                                 + 0.951686 w_k1_INTC_s100
                                 + 0.909709 w_k1_KO_s100
                                 + 0.892369 w_k1_MAT_s100
                                 + 1.0084 w_k1_MSFT_s100
                                 + 0.939544 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.35687 w_k1_ADM_s100
                                 + 1.35993 w_k1_AXP_s100
                                 + 1.42919 w_k1_DIS_s100
                                 + 1.31342 w_k1_FDX_s100
                                 + 0.988279 w_k1_IBM_s100
                                 + 1.40153 w_k1_INTC_s100
                                 + 1.20063 w_k1_KO_s100
                                 + 0.941607 w_k1_MAT_s100
                                 + 1.4942 w_k1_MSFT_s100 + 1.29103 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.09705 w_k1_ADM_s100
                                 + 0.922029 w_k1_AXP_s100
                                 + 1.36761 w_k1_DIS_s100
                                 + 1.13038 w_k1_FDX_s100
                                 + 1.04131 w_k1_IBM_s100
                                 + 1.08856 w_k1_INTC_s100
                                 + 1.21411 w_k1_KO_s100 + 1.23268 w_k1_MAT_s100
                                 + 1.32149 w_k1_MSFT_s100
                                 + 1.19977 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.761481 w_k1_ADM_s100
                                 + 0.789658 w_k1_AXP_s100
                                 + 0.755552 w_k1_DIS_s100
                                 + 0.78397 w_k1_FDX_s100
                                 + 1.12049 w_k1_IBM_s100
                                 + 0.78219 w_k1_INTC_s100
                                 + 0.884243 w_k1_KO_s100
                                 + 0.877548 w_k1_MAT_s100
                                 + 0.841993 w_k1_MSFT_s100
                                 + 0.900218 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.967941 w_k1_ADM_s100
                                 + 1.11966 w_k1_AXP_s100
                                 + 0.762365 w_k1_DIS_s100
                                 + 0.940317 w_k1_FDX_s100
                                 + 1.01392 w_k1_IBM_s100
                                 + 0.78574 w_k1_INTC_s100
                                 + 0.876248 w_k1_KO_s100
                                 + 1.09012 w_k1_MAT_s100
                                 + 0.941285 w_k1_MSFT_s100
                                 + 0.941824 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.94167 w_k1_ADM_s100
                                 + 0.894668 w_k1_AXP_s100
                                 + 0.948779 w_k1_DIS_s100
                                 + 0.932354 w_k1_FDX_s100
                                 + 0.913513 w_k1_IBM_s100
                                 + 1.04159 w_k1_INTC_s100
                                 + 0.981572 w_k1_KO_s100
                                 + 0.700081 w_k1_MAT_s100
                                 + 1.18509 w_k1_MSFT_s100
                                 + 0.992241 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.736574 w_k1_ADM_s100
                                 + 0.592608 w_k1_AXP_s100
                                 + 0.610479 w_k1_DIS_s100
                                 + 0.740318 w_k1_FDX_s100
                                 + 0.712028 w_k1_IBM_s100
                                 + 0.825826 w_k1_INTC_s100
                                 + 0.824874 w_k1_KO_s100
                                 + 0.720693 w_k1_MAT_s100
                                 + 0.896698 w_k1_MSFT_s100
                                 + 0.799483 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.13924 w_k1_ADM_s100
                                 + 1.35764 w_k1_AXP_s100
                                 + 1.20855 w_k1_DIS_s100
                                 + 1.05912 w_k1_FDX_s100
                                 + 1.42742 w_k1_IBM_s100
                                 + 0.983409 w_k1_INTC_s100
                                 + 1.30542 w_k1_KO_s100 + 1.70505 w_k1_MAT_s100
                                 + 1.20948 w_k1_MSFT_s100
                                 + 1.18238 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.943631 w_k1_ADM_s101
                                 + 1.29593 w_k1_AXP_s101
                                 + 0.953899 w_k1_DIS_s101
                                 + 0.984104 w_k1_FDX_s101
                                 + 1.53096 w_k1_IBM_s101
                                 + 1.27091 w_k1_INTC_s101
                                 + 1.06196 w_k1_KO_s101 + 1.68827 w_k1_MAT_s101
                                 + 1.04095 w_k1_MSFT_s101
                                 + 1.06028 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.00556 w_k1_ADM_s101
                                 + 1.05872 w_k1_AXP_s101
                                 + 1.15233 w_k1_DIS_s101
                                 + 1.12803 w_k1_FDX_s101
                                 + 1.16479 w_k1_IBM_s101
                                 + 1.35144 w_k1_INTC_s101
                                 + 1.14504 w_k1_KO_s101 + 1.06139 w_k1_MAT_s101
                                 + 1.16778 w_k1_MSFT_s101
                                 + 1.08695 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.91708 w_k1_ADM_s101
                                 + 1.05401 w_k1_AXP_s101
                                 + 0.984937 w_k1_DIS_s101
                                 + 1.07134 w_k1_FDX_s101
                                 + 0.848166 w_k1_IBM_s101
                                 + 1.32026 w_k1_INTC_s101
                                 + 0.840509 w_k1_KO_s101
                                 + 0.831219 w_k1_MAT_s101
                                 + 1.13779 w_k1_MSFT_s101
                                 + 0.947612 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.24486 w_k1_ADM_s101
                                 + 1.4022 w_k1_AXP_s101 + 1.54135 w_k1_DIS_s101
                                 + 1.25459 w_k1_FDX_s101
                                 + 1.48111 w_k1_IBM_s101
                                 + 1.35883 w_k1_INTC_s101
                                 + 1.38769 w_k1_KO_s101 + 1.91842 w_k1_MAT_s101
                                 + 1.3015 w_k1_MSFT_s101 + 1.09375 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.13188 w_k1_ADM_s101
                                 + 0.941607 w_k1_AXP_s101
                                 + 0.910076 w_k1_DIS_s101
                                 + 1.08814 w_k1_FDX_s101
                                 + 0.746385 w_k1_IBM_s101
                                 + 1.06959 w_k1_INTC_s101
                                 + 0.787133 w_k1_KO_s101
                                 + 0.646769 w_k1_MAT_s101
                                 + 1.10213 w_k1_MSFT_s101
                                 + 0.918833 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.924402 w_k1_ADM_s101
                                 + 0.805791 w_k1_AXP_s101
                                 + 0.738309 w_k1_DIS_s101
                                 + 0.789224 w_k1_FDX_s101
                                 + 0.914628 w_k1_IBM_s101
                                 + 1.16146 w_k1_INTC_s101
                                 + 0.785332 w_k1_KO_s101
                                 + 0.839683 w_k1_MAT_s101
                                 + 0.799657 w_k1_MSFT_s101
                                 + 0.838324 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.50128 w_k1_ADM_s101
                                 + 1.28994 w_k1_AXP_s101
                                 + 1.42319 w_k1_DIS_s101
                                 + 1.37527 w_k1_FDX_s101
                                 + 0.832771 w_k1_IBM_s101
                                 + 1.27921 w_k1_INTC_s101
                                 + 1.01139 w_k1_KO_s101
                                 + 0.941453 w_k1_MAT_s101
                                 + 1.43518 w_k1_MSFT_s101
                                 + 1.19451 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.992112 w_k1_ADM_s101
                                 + 0.88935 w_k1_AXP_s101
                                 + 1.01881 w_k1_DIS_s101
                                 + 0.910749 w_k1_FDX_s101
                                 + 1.16902 w_k1_IBM_s101
                                 + 1.07901 w_k1_INTC_s101
                                 + 0.893261 w_k1_KO_s101
                                 + 0.752136 w_k1_MAT_s101
                                 + 1.03218 w_k1_MSFT_s101
                                 + 0.915275 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.07018 w_k1_ADM_s102
                                 + 1.46176 w_k1_AXP_s102
                                 + 1.56568 w_k1_DIS_s102
                                 + 1.31153 w_k1_FDX_s102
                                 + 1.50961 w_k1_IBM_s102
                                 + 1.16058 w_k1_INTC_s102
                                 + 1.32649 w_k1_KO_s102 + 1.65583 w_k1_MAT_s102
                                 + 1.20184 w_k1_MSFT_s102
                                 + 1.26548 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.04004 w_k1_ADM_s102
                                 + 1.0265 w_k1_AXP_s102
                                 + 0.951748 w_k1_DIS_s102
                                 + 0.988492 w_k1_FDX_s102
                                 + 0.821224 w_k1_IBM_s102
                                 + 1.00863 w_k1_INTC_s102
                                 + 0.817257 w_k1_KO_s102
                                 + 0.624097 w_k1_MAT_s102
                                 + 0.904263 w_k1_MSFT_s102
                                 + 0.903861 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.05329 w_k1_ADM_s102
                                 + 1.15114 w_k1_AXP_s102
                                 + 1.16699 w_k1_DIS_s102
                                 + 1.11328 w_k1_FDX_s102 + 1.1998 w_k1_IBM_s102
                                 + 1.11816 w_k1_INTC_s102
                                 + 1.04092 w_k1_KO_s102 + 1.44504 w_k1_MAT_s102
                                 + 0.988675 w_k1_MSFT_s102
                                 + 1.10553 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05193 w_k1_ADM_s102
                                 + 1.16575 w_k1_AXP_s102
                                 + 1.41748 w_k1_DIS_s102
                                 + 1.23139 w_k1_FDX_s102
                                 + 0.907325 w_k1_IBM_s102
                                 + 1.24775 w_k1_INTC_s102
                                 + 1.08035 w_k1_KO_s102
                                 + 0.913553 w_k1_MAT_s102
                                 + 1.19317 w_k1_MSFT_s102
                                 + 1.12001 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.820862 w_k1_ADM_s102
                                 + 0.634951 w_k1_AXP_s102
                                 + 0.540662 w_k1_DIS_s102
                                 + 0.642387 w_k1_FDX_s102
                                 + 1.02822 w_k1_IBM_s102
                                 + 0.793525 w_k1_INTC_s102
                                 + 0.722639 w_k1_KO_s102
                                 + 0.95776 w_k1_MAT_s102
                                 + 0.676666 w_k1_MSFT_s102
                                 + 0.858608 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.871983 w_k1_ADM_s102
                                 + 0.951073 w_k1_AXP_s102
                                 + 0.798983 w_k1_DIS_s102
                                 + 0.837829 w_k1_FDX_s102
                                 + 1.21871 w_k1_IBM_s102
                                 + 0.848231 w_k1_INTC_s102
                                 + 0.886608 w_k1_KO_s102
                                 + 1.09338 w_k1_MAT_s102
                                 + 0.803822 w_k1_MSFT_s102
                                 + 0.919785 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.956266 w_k1_ADM_s102
                                 + 0.685432 w_k1_AXP_s102
                                 + 0.600707 w_k1_DIS_s102
                                 + 0.791997 w_k1_FDX_s102
                                 + 0.617562 w_k1_IBM_s102
                                 + 0.806569 w_k1_INTC_s102
                                 + 0.639756 w_k1_KO_s102
                                 + 0.509447 w_k1_MAT_s102
                                 + 0.71107 w_k1_MSFT_s102
                                 + 0.784258 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.996935 w_k1_ADM_s102
                                 + 1.20796 w_k1_AXP_s102
                                 + 1.18201 w_k1_DIS_s102
                                 + 0.94693 w_k1_FDX_s102
                                 + 1.61719 w_k1_IBM_s102
                                 + 0.894682 w_k1_INTC_s102
                                 + 1.24767 w_k1_KO_s102 + 1.75492 w_k1_MAT_s102
                                 + 0.820935 w_k1_MSFT_s102
                                 + 1.20271 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.709075 w_k1_ADM_s103
                                 + 0.751127 w_k1_AXP_s103
                                 + 0.697827 w_k1_DIS_s103
                                 + 0.73328 w_k1_FDX_s103
                                 + 0.959593 w_k1_IBM_s103
                                 + 1.14452 w_k1_INTC_s103
                                 + 0.933088 w_k1_KO_s103
                                 + 1.10162 w_k1_MAT_s103
                                 + 0.813174 w_k1_MSFT_s103
                                 + 0.852297 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.85138 w_k1_ADM_s103
                                 + 0.966855 w_k1_AXP_s103
                                 + 0.983728 w_k1_DIS_s103
                                 + 1.07307 w_k1_FDX_s103
                                 + 0.914247 w_k1_IBM_s103
                                 + 1.22893 w_k1_INTC_s103
                                 + 0.899693 w_k1_KO_s103
                                 + 0.738326 w_k1_MAT_s103
                                 + 1.02234 w_k1_MSFT_s103
                                 + 0.903957 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.829913 w_k1_ADM_s103
                                 + 1.12576 w_k1_AXP_s103
                                 + 0.804204 w_k1_DIS_s103
                                 + 0.894335 w_k1_FDX_s103
                                 + 1.31014 w_k1_IBM_s103
                                 + 1.15543 w_k1_INTC_s103
                                 + 1.23676 w_k1_KO_s103 + 1.65836 w_k1_MAT_s103
                                 + 0.992956 w_k1_MSFT_s103
                                 + 0.984558 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.11767 w_k1_ADM_s103
                                 + 1.74736 w_k1_AXP_s103
                                 + 1.05318 w_k1_DIS_s103
                                 + 1.21198 w_k1_FDX_s103 + 1.2755 w_k1_IBM_s103
                                 + 1.39226 w_k1_INTC_s103 + 1.0712 w_k1_KO_s103
                                 + 1.75849 w_k1_MAT_s103
                                 + 1.08304 w_k1_MSFT_s103
                                 + 1.08477 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.998178 w_k1_ADM_s103
                                 + 1.12209 w_k1_AXP_s103
                                 + 1.21071 w_k1_DIS_s103
                                 + 1.14251 w_k1_FDX_s103
                                 + 0.879034 w_k1_IBM_s103
                                 + 1.35377 w_k1_INTC_s103
                                 + 1.11183 w_k1_KO_s103
                                 + 0.855377 w_k1_MAT_s103
                                 + 1.2192 w_k1_MSFT_s103 + 1.01181 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.661448 w_k1_ADM_s103
                                 + 0.740537 w_k1_AXP_s103
                                 + 0.596821 w_k1_DIS_s103
                                 + 0.791741 w_k1_FDX_s103
                                 + 0.692394 w_k1_IBM_s103
                                 + 1.15989 w_k1_INTC_s103
                                 + 0.723407 w_k1_KO_s103
                                 + 0.894045 w_k1_MAT_s103
                                 + 0.781364 w_k1_MSFT_s103
                                 + 0.829745 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.82379 w_k1_ADM_s103
                                 + 0.887809 w_k1_AXP_s103
                                 + 1.11083 w_k1_DIS_s103
                                 + 0.969517 w_k1_FDX_s103
                                 + 1.23906 w_k1_IBM_s103
                                 + 1.09835 w_k1_INTC_s103
                                 + 1.17476 w_k1_KO_s103 + 1.71283 w_k1_MAT_s103
                                 + 1.0599 w_k1_MSFT_s103 + 1.02108 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.17905 w_k1_ADM_s103
                                 + 1.32048 w_k1_AXP_s103
                                 + 1.47862 w_k1_DIS_s103
                                 + 1.36337 w_k1_FDX_s103
                                 + 1.04434 w_k1_IBM_s103
                                 + 1.35966 w_k1_INTC_s103
                                 + 1.14336 w_k1_KO_s103 + 1.12435 w_k1_MAT_s103
                                 + 1.27541 w_k1_MSFT_s103
                                 + 1.12083 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.06558 w_k1_ADM_s104
                                 + 1.33879 w_k1_AXP_s104
                                 + 1.05051 w_k1_DIS_s104
                                 + 1.17193 w_k1_FDX_s104
                                 + 0.742145 w_k1_IBM_s104
                                 + 1.1375 w_k1_INTC_s104
                                 + 0.881966 w_k1_KO_s104
                                 + 0.714786 w_k1_MAT_s104
                                 + 0.925924 w_k1_MSFT_s104
                                 + 1.08885 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.12167 w_k1_ADM_s104
                                 + 1.40643 w_k1_AXP_s104
                                 + 0.890166 w_k1_DIS_s104
                                 + 1.04303 w_k1_FDX_s104
                                 + 1.09532 w_k1_IBM_s104
                                 + 1.17912 w_k1_INTC_s104
                                 + 1.06372 w_k1_KO_s104 + 1.26013 w_k1_MAT_s104
                                 + 0.997669 w_k1_MSFT_s104
                                 + 1.17112 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.897435 w_k1_ADM_s104
                                 + 0.777141 w_k1_AXP_s104
                                 + 0.808393 w_k1_DIS_s104
                                 + 0.918819 w_k1_FDX_s104
                                 + 0.680975 w_k1_IBM_s104
                                 + 0.934587 w_k1_INTC_s104
                                 + 0.738651 w_k1_KO_s104
                                 + 0.617961 w_k1_MAT_s104
                                 + 0.843968 w_k1_MSFT_s104
                                 + 0.91633 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.09445 w_k1_ADM_s104
                                 + 1.63845 w_k1_AXP_s104
                                 + 1.46616 w_k1_DIS_s104
                                 + 1.20986 w_k1_FDX_s104
                                 + 1.46588 w_k1_IBM_s104
                                 + 1.07469 w_k1_INTC_s104
                                 + 1.13864 w_k1_KO_s104 + 1.46538 w_k1_MAT_s104
                                 + 1.10788 w_k1_MSFT_s104
                                 + 1.04797 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.98962 w_k1_ADM_s104
                                 + 0.958458 w_k1_AXP_s104
                                 + 1.14963 w_k1_DIS_s104
                                 + 1.10376 w_k1_FDX_s104
                                 + 0.905366 w_k1_IBM_s104
                                 + 1.13926 w_k1_INTC_s104
                                 + 0.862748 w_k1_KO_s104
                                 + 0.453781 w_k1_MAT_s104
                                 + 1.04612 w_k1_MSFT_s104
                                 + 0.997584 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.82896 w_k1_ADM_s104
                                 + 0.800731 w_k1_AXP_s104
                                 + 0.773276 w_k1_DIS_s104
                                 + 0.826111 w_k1_FDX_s104
                                 + 1.08104 w_k1_IBM_s104
                                 + 0.865086 w_k1_INTC_s104
                                 + 0.852985 w_k1_KO_s104
                                 + 0.891339 w_k1_MAT_s104
                                 + 0.796959 w_k1_MSFT_s104
                                 + 0.92998 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.17455 w_k1_ADM_s104
                                 + 1.01463 w_k1_AXP_s104
                                 + 1.38681 w_k1_DIS_s104
                                 + 1.10314 w_k1_FDX_s104
                                 + 1.46043 w_k1_IBM_s104
                                 + 1.27181 w_k1_INTC_s104
                                 + 1.17619 w_k1_KO_s104 + 1.05394 w_k1_MAT_s104
                                 + 1.12278 w_k1_MSFT_s104
                                 + 1.05399 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.45668 w_k1_ADM_s104
                                 + 1.67895 w_k1_AXP_s104
                                 + 1.59677 w_k1_DIS_s104
                                 + 1.41258 w_k1_FDX_s104
                                 + 1.11883 w_k1_IBM_s104
                                 + 1.20311 w_k1_INTC_s104 + 1.2307 w_k1_KO_s104
                                 + 1.43566 w_k1_MAT_s104
                                 + 1.26523 w_k1_MSFT_s104 + 1.224 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.18658 w_k1_ADM_s105
                                 + 1.19038 w_k1_AXP_s105 + 1.5603 w_k1_DIS_s105
                                 + 1.02897 w_k1_FDX_s105
                                 + 1.57773 w_k1_IBM_s105
                                 + 1.07649 w_k1_INTC_s105
                                 + 1.36617 w_k1_KO_s105 + 1.20841 w_k1_MAT_s105
                                 + 1.03837 w_k1_MSFT_s105
                                 + 1.22267 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.10207 w_k1_ADM_s105
                                 + 1.48951 w_k1_AXP_s105
                                 + 1.08327 w_k1_DIS_s105
                                 + 1.06454 w_k1_FDX_s105
                                 + 1.11457 w_k1_IBM_s105
                                 + 1.09548 w_k1_INTC_s105
                                 + 1.02856 w_k1_KO_s105 + 1.16111 w_k1_MAT_s105
                                 + 1.02656 w_k1_MSFT_s105
                                 + 1.19088 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.908275 w_k1_ADM_s105
                                 + 1.01916 w_k1_AXP_s105
                                 + 0.825484 w_k1_DIS_s105
                                 + 0.950986 w_k1_FDX_s105
                                 + 0.935219 w_k1_IBM_s105
                                 + 0.974163 w_k1_INTC_s105
                                 + 0.863765 w_k1_KO_s105
                                 + 0.800312 w_k1_MAT_s105
                                 + 0.908968 w_k1_MSFT_s105
                                 + 0.984474 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.28526 w_k1_ADM_s105
                                 + 1.46396 w_k1_AXP_s105
                                 + 1.61534 w_k1_DIS_s105
                                 + 1.34674 w_k1_FDX_s105
                                 + 1.14174 w_k1_IBM_s105
                                 + 1.22775 w_k1_INTC_s105
                                 + 1.10406 w_k1_KO_s105
                                 + 0.876983 w_k1_MAT_s105
                                 + 1.14549 w_k1_MSFT_s105 + 1.0845 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.18197 w_k1_ADM_s105
                                 + 1.02751 w_k1_AXP_s105
                                 + 1.18328 w_k1_DIS_s105
                                 + 1.05471 w_k1_FDX_s105
                                 + 0.707724 w_k1_IBM_s105
                                 + 1.05353 w_k1_INTC_s105
                                 + 0.951614 w_k1_KO_s105
                                 + 0.681001 w_k1_MAT_s105
                                 + 1.02554 w_k1_MSFT_s105
                                 + 1.00731 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.851788 w_k1_ADM_s105
                                 + 0.706319 w_k1_AXP_s105
                                 + 1.00997 w_k1_DIS_s105
                                 + 0.808798 w_k1_FDX_s105
                                 + 0.968495 w_k1_IBM_s105
                                 + 0.881195 w_k1_INTC_s105
                                 + 0.947921 w_k1_KO_s105
                                 + 0.504501 w_k1_MAT_s105
                                 + 0.860835 w_k1_MSFT_s105
                                 + 0.869493 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.8083 w_k1_ADM_s105
                                 + 0.750885 w_k1_AXP_s105
                                 + 0.664834 w_k1_DIS_s105
                                 + 0.75983 w_k1_FDX_s105
                                 + 0.842516 w_k1_IBM_s105
                                 + 0.931472 w_k1_INTC_s105
                                 + 0.783111 w_k1_KO_s105
                                 + 0.645422 w_k1_MAT_s105
                                 + 0.79664 w_k1_MSFT_s105
                                 + 0.843183 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.997678 w_k1_ADM_s105
                                 + 0.927269 w_k1_AXP_s105
                                 + 1.13265 w_k1_DIS_s105
                                 + 1.01708 w_k1_FDX_s105 + 1.0919 w_k1_IBM_s105
                                 + 0.968241 w_k1_INTC_s105
                                 + 1.01374 w_k1_KO_s105
                                 + 0.781097 w_k1_MAT_s105
                                 + 1.10849 w_k1_MSFT_s105
                                 + 1.07191 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.27692 w_k1_ADM_s106
                                 + 1.11425 w_k1_AXP_s106
                                 + 1.15092 w_k1_DIS_s106
                                 + 1.17243 w_k1_FDX_s106
                                 + 0.861939 w_k1_IBM_s106
                                 + 1.12953 w_k1_INTC_s106
                                 + 1.20632 w_k1_KO_s106
                                 + 0.962644 w_k1_MAT_s106
                                 + 1.13867 w_k1_MSFT_s106
                                 + 1.16769 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.09656 w_k1_ADM_s106
                                 + 1.41472 w_k1_AXP_s106 + 1.5772 w_k1_DIS_s106
                                 + 1.22647 w_k1_FDX_s106
                                 + 1.54535 w_k1_IBM_s106
                                 + 1.06453 w_k1_INTC_s106
                                 + 1.44431 w_k1_KO_s106 + 1.5032 w_k1_MAT_s106
                                 + 1.23291 w_k1_MSFT_s106
                                 + 1.24893 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.88017 w_k1_ADM_s106
                                 + 0.789853 w_k1_AXP_s106
                                 + 0.985028 w_k1_DIS_s106
                                 + 0.962151 w_k1_FDX_s106
                                 + 0.856129 w_k1_IBM_s106
                                 + 0.841996 w_k1_INTC_s106
                                 + 1.01322 w_k1_KO_s106 + 0.97741 w_k1_MAT_s106
                                 + 0.922829 w_k1_MSFT_s106
                                 + 0.992322 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.859051 w_k1_ADM_s106
                                 + 0.734755 w_k1_AXP_s106
                                 + 0.842827 w_k1_DIS_s106
                                 + 0.986514 w_k1_FDX_s106
                                 + 0.578915 w_k1_IBM_s106
                                 + 0.961163 w_k1_INTC_s106
                                 + 0.786873 w_k1_KO_s106
                                 + 0.479554 w_k1_MAT_s106
                                 + 1.05371 w_k1_MSFT_s106
                                 + 0.918115 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.967216 w_k1_ADM_s106
                                 + 1.25756 w_k1_AXP_s106
                                 + 1.42148 w_k1_DIS_s106
                                 + 1.22092 w_k1_FDX_s106
                                 + 1.10915 w_k1_IBM_s106
                                 + 1.27923 w_k1_INTC_s106
                                 + 1.28472 w_k1_KO_s106 + 1.14968 w_k1_MAT_s106
                                 + 1.20388 w_k1_MSFT_s106
                                 + 1.18447 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.836857 w_k1_ADM_s106
                                 + 0.856275 w_k1_AXP_s106
                                 + 1.12268 w_k1_DIS_s106
                                 + 0.930187 w_k1_FDX_s106
                                 + 1.5443 w_k1_IBM_s106
                                 + 0.908812 w_k1_INTC_s106
                                 + 1.11399 w_k1_KO_s106 + 1.30465 w_k1_MAT_s106
                                 + 0.995297 w_k1_MSFT_s106
                                 + 1.05213 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.6051 w_k1_ADM_s106
                                 + 0.530167 w_k1_AXP_s106
                                 + 0.509767 w_k1_DIS_s106
                                 + 0.672382 w_k1_FDX_s106
                                 + 0.676813 w_k1_IBM_s106
                                 + 0.850013 w_k1_INTC_s106
                                 + 0.646872 w_k1_KO_s106
                                 + 0.578585 w_k1_MAT_s106
                                 + 0.752894 w_k1_MSFT_s106
                                 + 0.752605 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.824133 w_k1_ADM_s106
                                 + 0.637707 w_k1_AXP_s106
                                 + 0.647218 w_k1_DIS_s106
                                 + 0.782355 w_k1_FDX_s106
                                 + 1.01314 w_k1_IBM_s106
                                 + 0.956897 w_k1_INTC_s106
                                 + 0.985608 w_k1_KO_s106
                                 + 0.724734 w_k1_MAT_s106
                                 + 0.793829 w_k1_MSFT_s106
                                 + 0.953151 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.86176 w_k1_ADM_s107
                                 + 0.747368 w_k1_AXP_s107
                                 + 0.615483 w_k1_DIS_s107
                                 + 0.803424 w_k1_FDX_s107
                                 + 0.767808 w_k1_IBM_s107
                                 + 0.75852 w_k1_INTC_s107
                                 + 0.634933 w_k1_KO_s107
                                 + 0.836154 w_k1_MAT_s107
                                 + 0.749568 w_k1_MSFT_s107
                                 + 0.805123 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.23766 w_k1_ADM_s107
                                 + 1.47883 w_k1_AXP_s107
                                 + 1.94533 w_k1_DIS_s107
                                 + 1.44204 w_k1_FDX_s107
                                 + 1.16082 w_k1_IBM_s107
                                 + 1.30672 w_k1_INTC_s107
                                 + 1.25788 w_k1_KO_s107 + 1.1123 w_k1_MAT_s107
                                 + 1.41781 w_k1_MSFT_s107
                                 + 1.08178 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.931353 w_k1_ADM_s107
                                 + 0.983887 w_k1_AXP_s107
                                 + 1.06593 w_k1_DIS_s107
                                 + 1.02866 w_k1_FDX_s107
                                 + 1.53394 w_k1_IBM_s107
                                 + 1.09633 w_k1_INTC_s107
                                 + 1.13853 w_k1_KO_s107 + 1.75621 w_k1_MAT_s107
                                 + 1.01294 w_k1_MSFT_s107
                                 + 0.942374 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.19118 w_k1_ADM_s107
                                 + 1.32721 w_k1_AXP_s107
                                 + 1.23389 w_k1_DIS_s107
                                 + 1.17594 w_k1_FDX_s107
                                 + 1.23925 w_k1_IBM_s107
                                 + 1.15217 w_k1_INTC_s107
                                 + 1.03167 w_k1_KO_s107
                                 + 0.946888 w_k1_MAT_s107
                                 + 0.989067 w_k1_MSFT_s107
                                 + 1.06538 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.895388 w_k1_ADM_s107
                                 + 0.561639 w_k1_AXP_s107
                                 + 0.745556 w_k1_DIS_s107
                                 + 0.830082 w_k1_FDX_s107
                                 + 0.531608 w_k1_IBM_s107
                                 + 0.799077 w_k1_INTC_s107
                                 + 0.670904 w_k1_KO_s107
                                 + 0.385495 w_k1_MAT_s107
                                 + 0.825762 w_k1_MSFT_s107
                                 + 0.682632 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.902987 w_k1_ADM_s107
                                 + 0.879489 w_k1_AXP_s107
                                 + 1.04774 w_k1_DIS_s107
                                 + 1.01902 w_k1_FDX_s107
                                 + 0.759386 w_k1_IBM_s107
                                 + 1.15963 w_k1_INTC_s107
                                 + 0.799007 w_k1_KO_s107
                                 + 0.598639 w_k1_MAT_s107
                                 + 1.01202 w_k1_MSFT_s107
                                 + 0.858566 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.818781 w_k1_ADM_s107
                                 + 0.871537 w_k1_AXP_s107
                                 + 1.03993 w_k1_DIS_s107
                                 + 0.890555 w_k1_FDX_s107
                                 + 1.13149 w_k1_IBM_s107
                                 + 0.990931 w_k1_INTC_s107 + 0.972 w_k1_KO_s107
                                 + 0.800587 w_k1_MAT_s107
                                 + 0.950867 w_k1_MSFT_s107
                                 + 0.893644 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.935003 w_k1_ADM_s107
                                 + 1.15614 w_k1_AXP_s107 + 1.1622 w_k1_DIS_s107
                                 + 1.18972 w_k1_FDX_s107
                                 + 0.906595 w_k1_IBM_s107
                                 + 1.0743 w_k1_INTC_s107 + 0.95696 w_k1_KO_s107
                                 + 1.17816 w_k1_MAT_s107
                                 + 1.04371 w_k1_MSFT_s107
                                 + 0.858815 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.878305 w_k1_ADM_s110
                                 + 1.17493 w_k1_AXP_s110
                                 + 1.02099 w_k1_DIS_s110
                                 + 1.05452 w_k1_FDX_s110 + 1.0306 w_k1_IBM_s110
                                 + 0.802882 w_k1_INTC_s110
                                 + 0.913687 w_k1_KO_s110
                                 + 1.03795 w_k1_MAT_s110
                                 + 0.844515 w_k1_MSFT_s110
                                 + 0.939081 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.18145 w_k1_ADM_s110
                                 + 1.13877 w_k1_AXP_s110 + 1.4614 w_k1_DIS_s110
                                 + 1.23666 w_k1_FDX_s110
                                 + 0.999906 w_k1_IBM_s110
                                 + 1.15675 w_k1_INTC_s110
                                 + 1.32033 w_k1_KO_s110
                                 + 0.713389 w_k1_MAT_s110
                                 + 1.1874 w_k1_MSFT_s110 + 1.05717 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.24582 w_k1_ADM_s110
                                 + 1.48141 w_k1_AXP_s110
                                 + 1.28208 w_k1_DIS_s110
                                 + 1.27652 w_k1_FDX_s110
                                 + 1.63312 w_k1_IBM_s110
                                 + 1.17356 w_k1_INTC_s110
                                 + 1.36917 w_k1_KO_s110 + 1.24773 w_k1_MAT_s110
                                 + 0.991967 w_k1_MSFT_s110
                                 + 1.12161 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.732054 w_k1_ADM_s110
                                 + 0.872769 w_k1_AXP_s110
                                 + 0.927611 w_k1_DIS_s110
                                 + 1.03292 w_k1_FDX_s110
                                 + 0.728963 w_k1_IBM_s110
                                 + 0.925002 w_k1_INTC_s110
                                 + 0.793797 w_k1_KO_s110
                                 + 0.724319 w_k1_MAT_s110
                                 + 0.98816 w_k1_MSFT_s110
                                 + 0.876288 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.19475 w_k1_ADM_s110
                                 + 1.18375 w_k1_AXP_s110
                                 + 1.50639 w_k1_DIS_s110
                                 + 1.25979 w_k1_FDX_s110
                                 + 1.21678 w_k1_IBM_s110
                                 + 1.23226 w_k1_INTC_s110
                                 + 1.17276 w_k1_KO_s110
                                 + 0.945916 w_k1_MAT_s110
                                 + 1.12581 w_k1_MSFT_s110
                                 + 1.06374 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.837886 w_k1_ADM_s110
                                 + 1.05016 w_k1_AXP_s110
                                 + 1.00734 w_k1_DIS_s110
                                 + 0.879454 w_k1_FDX_s110
                                 + 1.25968 w_k1_IBM_s110
                                 + 1.06667 w_k1_INTC_s110
                                 + 1.10658 w_k1_KO_s110 + 1.17343 w_k1_MAT_s110
                                 + 0.922361 w_k1_MSFT_s110
                                 + 0.989346 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.904346 w_k1_ADM_s110
                                 + 0.751725 w_k1_AXP_s110
                                 + 1.14752 w_k1_DIS_s110
                                 + 1.08157 w_k1_FDX_s110
                                 + 0.815567 w_k1_IBM_s110
                                 + 0.908482 w_k1_INTC_s110
                                 + 0.971535 w_k1_KO_s110
                                 + 0.508652 w_k1_MAT_s110
                                 + 0.886413 w_k1_MSFT_s110
                                 + 0.954548 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.776043 w_k1_ADM_s110
                                 + 0.750589 w_k1_AXP_s110
                                 + 0.698965 w_k1_DIS_s110
                                 + 0.827525 w_k1_FDX_s110
                                 + 0.83348 w_k1_IBM_s110
                                 + 0.630905 w_k1_INTC_s110
                                 + 0.744606 w_k1_KO_s110
                                 + 0.643827 w_k1_MAT_s110
                                 + 0.727266 w_k1_MSFT_s110
                                 + 0.747085 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.35258 w_k1_ADM_s111
                                 + 1.02563 w_k1_AXP_s111
                                 + 1.17094 w_k1_DIS_s111
                                 + 0.970035 w_k1_FDX_s111
                                 + 1.67043 w_k1_IBM_s111
                                 + 0.974704 w_k1_INTC_s111
                                 + 1.36613 w_k1_KO_s111 + 1.59097 w_k1_MAT_s111
                                 + 0.954039 w_k1_MSFT_s111
                                 + 1.24008 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.35594 w_k1_ADM_s111
                                 + 1.05675 w_k1_AXP_s111
                                 + 0.939329 w_k1_DIS_s111
                                 + 1.02087 w_k1_FDX_s111
                                 + 0.962574 w_k1_IBM_s111
                                 + 1.08355 w_k1_INTC_s111
                                 + 0.989167 w_k1_KO_s111
                                 + 0.803529 w_k1_MAT_s111
                                 + 0.973125 w_k1_MSFT_s111
                                 + 1.02947 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.11121 w_k1_ADM_s111
                                 + 0.813922 w_k1_AXP_s111
                                 + 0.889391 w_k1_DIS_s111
                                 + 0.999099 w_k1_FDX_s111
                                 + 0.818143 w_k1_IBM_s111
                                 + 1.05001 w_k1_INTC_s111
                                 + 0.826108 w_k1_KO_s111
                                 + 0.668689 w_k1_MAT_s111
                                 + 0.890321 w_k1_MSFT_s111
                                 + 0.839552 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.76956 w_k1_ADM_s111
                                 + 1.21445 w_k1_AXP_s111
                                 + 1.54426 w_k1_DIS_s111 + 1.4215 w_k1_FDX_s111
                                 + 0.912148 w_k1_IBM_s111
                                 + 1.19987 w_k1_INTC_s111
                                 + 1.30132 w_k1_KO_s111 + 0.75894 w_k1_MAT_s111
                                 + 1.17447 w_k1_MSFT_s111
                                 + 1.20898 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.864453 w_k1_ADM_s111
                                 + 0.680437 w_k1_AXP_s111
                                 + 0.624258 w_k1_DIS_s111
                                 + 0.72466 w_k1_FDX_s111
                                 + 0.700589 w_k1_IBM_s111
                                 + 0.810674 w_k1_INTC_s111
                                 + 0.677762 w_k1_KO_s111
                                 + 0.917785 w_k1_MAT_s111
                                 + 0.728894 w_k1_MSFT_s111
                                 + 0.77365 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.68576 w_k1_ADM_s111
                                 + 1.52085 w_k1_AXP_s111
                                 + 1.67322 w_k1_DIS_s111
                                 + 1.43382 w_k1_FDX_s111 + 1.2917 w_k1_IBM_s111
                                 + 1.37548 w_k1_INTC_s111 + 1.3001 w_k1_KO_s111
                                 + 1.36474 w_k1_MAT_s111
                                 + 1.30878 w_k1_MSFT_s111
                                 + 1.40631 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.818422 w_k1_ADM_s111
                                 + 0.868206 w_k1_AXP_s111
                                 + 0.687327 w_k1_DIS_s111
                                 + 0.772539 w_k1_FDX_s111
                                 + 1.24168 w_k1_IBM_s111
                                 + 0.842231 w_k1_INTC_s111
                                 + 0.92618 w_k1_KO_s111 + 1.27818 w_k1_MAT_s111
                                 + 0.766547 w_k1_MSFT_s111
                                 + 0.942269 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.21037 w_k1_ADM_s111
                                 + 1.1243 w_k1_AXP_s111 + 1.15534 w_k1_DIS_s111
                                 + 1.04924 w_k1_FDX_s111
                                 + 1.22405 w_k1_IBM_s111
                                 + 1.29083 w_k1_INTC_s111
                                 + 1.12663 w_k1_KO_s111 + 1.34507 w_k1_MAT_s111
                                 + 1.06716 w_k1_MSFT_s111
                                 + 0.977987 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.66726 w_k1_ADM_s112
                                 + 0.626301 w_k1_AXP_s112
                                 + 0.83175 w_k1_DIS_s112
                                 + 0.810516 w_k1_FDX_s112
                                 + 0.856177 w_k1_IBM_s112
                                 + 0.992083 w_k1_INTC_s112
                                 + 0.871407 w_k1_KO_s112
                                 + 0.827525 w_k1_MAT_s112
                                 + 0.896389 w_k1_MSFT_s112
                                 + 0.866513 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.03442 w_k1_ADM_s112
                                 + 1.20835 w_k1_AXP_s112
                                 + 1.24717 w_k1_DIS_s112
                                 + 1.06274 w_k1_FDX_s112 + 1.1106 w_k1_IBM_s112
                                 + 1.11964 w_k1_INTC_s112
                                 + 1.06485 w_k1_KO_s112 + 1.72179 w_k1_MAT_s112
                                 + 1.21431 w_k1_MSFT_s112
                                 + 1.09659 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.698701 w_k1_ADM_s112
                                 + 0.745861 w_k1_AXP_s112
                                 + 0.599829 w_k1_DIS_s112
                                 + 0.73367 w_k1_FDX_s112
                                 + 0.762241 w_k1_IBM_s112
                                 + 0.916894 w_k1_INTC_s112
                                 + 0.643332 w_k1_KO_s112
                                 + 0.594034 w_k1_MAT_s112
                                 + 0.725899 w_k1_MSFT_s112
                                 + 0.817735 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.935245 w_k1_ADM_s112
                                 + 0.79792 w_k1_AXP_s112
                                 + 0.896167 w_k1_DIS_s112
                                 + 0.995702 w_k1_FDX_s112
                                 + 0.916092 w_k1_IBM_s112
                                 + 1.28154 w_k1_INTC_s112
                                 + 0.84429 w_k1_KO_s112 + 0.71928 w_k1_MAT_s112
                                 + 1.05344 w_k1_MSFT_s112
                                 + 1.03885 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.15579 w_k1_ADM_s112
                                 + 0.95978 w_k1_AXP_s112
                                 + 1.28447 w_k1_DIS_s112
                                 + 0.965446 w_k1_FDX_s112
                                 + 1.37229 w_k1_IBM_s112
                                 + 1.04533 w_k1_INTC_s112
                                 + 1.17557 w_k1_KO_s112 + 1.38012 w_k1_MAT_s112
                                 + 1.12637 w_k1_MSFT_s112
                                 + 0.970936 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.695126 w_k1_ADM_s112
                                 + 0.869644 w_k1_AXP_s112
                                 + 0.982589 w_k1_DIS_s112
                                 + 1.04429 w_k1_FDX_s112
                                 + 0.64143 w_k1_IBM_s112
                                 + 1.22291 w_k1_INTC_s112
                                 + 0.820574 w_k1_KO_s112
                                 + 0.759183 w_k1_MAT_s112
                                 + 1.09671 w_k1_MSFT_s112
                                 + 0.871119 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.36969 w_k1_ADM_s112
                                 + 1.64295 w_k1_AXP_s112
                                 + 1.59149 w_k1_DIS_s112
                                 + 1.35556 w_k1_FDX_s112 + 1.3004 w_k1_IBM_s112
                                 + 1.48576 w_k1_INTC_s112
                                 + 1.13649 w_k1_KO_s112 + 1.13132 w_k1_MAT_s112
                                 + 1.22416 w_k1_MSFT_s112 + 1.129 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.33428 w_k1_ADM_s112
                                 + 1.14853 w_k1_AXP_s112
                                 + 1.12854 w_k1_DIS_s112
                                 + 1.24088 w_k1_FDX_s112
                                 + 0.791296 w_k1_IBM_s112
                                 + 1.1152 w_k1_INTC_s112
                                 + 0.881204 w_k1_KO_s112
                                 + 0.915005 w_k1_MAT_s112
                                 + 0.958268 w_k1_MSFT_s112
                                 + 1.07059 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.0834 w_k1_ADM_s113
                                 + 0.83504 w_k1_AXP_s113
                                 + 1.23555 w_k1_DIS_s113
                                 + 1.05684 w_k1_FDX_s113
                                 + 0.84702 w_k1_IBM_s113
                                 + 1.00444 w_k1_INTC_s113
                                 + 0.999346 w_k1_KO_s113
                                 + 0.780398 w_k1_MAT_s113
                                 + 1.1007 w_k1_MSFT_s113 + 1.01931 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.16218 w_k1_ADM_s113
                                 + 1.08255 w_k1_AXP_s113
                                 + 1.42887 w_k1_DIS_s113
                                 + 1.17357 w_k1_FDX_s113
                                 + 1.25426 w_k1_IBM_s113
                                 + 1.2847 w_k1_INTC_s113 + 1.40537 w_k1_KO_s113
                                 + 1.08663 w_k1_MAT_s113
                                 + 1.25985 w_k1_MSFT_s113
                                 + 1.19383 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.53675 w_k1_ADM_s113
                                 + 1.65495 w_k1_AXP_s113
                                 + 1.55255 w_k1_DIS_s113
                                 + 1.15497 w_k1_FDX_s113 + 1.7543 w_k1_IBM_s113
                                 + 1.42426 w_k1_INTC_s113
                                 + 1.57191 w_k1_KO_s113 + 2.04162 w_k1_MAT_s113
                                 + 1.27785 w_k1_MSFT_s113
                                 + 1.29139 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.05708 w_k1_ADM_s113
                                 + 1.09277 w_k1_AXP_s113
                                 + 0.984592 w_k1_DIS_s113
                                 + 0.970149 w_k1_FDX_s113
                                 + 1.45515 w_k1_IBM_s113
                                 + 1.0212 w_k1_INTC_s113 + 1.06055 w_k1_KO_s113
                                 + 1.10378 w_k1_MAT_s113
                                 + 0.975068 w_k1_MSFT_s113
                                 + 1.04884 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.809338 w_k1_ADM_s113
                                 + 0.569194 w_k1_AXP_s113
                                 + 0.720047 w_k1_DIS_s113
                                 + 0.745762 w_k1_FDX_s113
                                 + 1.00312 w_k1_IBM_s113
                                 + 0.936423 w_k1_INTC_s113
                                 + 0.883353 w_k1_KO_s113
                                 + 0.70653 w_k1_MAT_s113
                                 + 0.915535 w_k1_MSFT_s113
                                 + 0.856324 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.37984 w_k1_ADM_s113
                                 + 1.57895 w_k1_AXP_s113
                                 + 1.41662 w_k1_DIS_s113
                                 + 1.46018 w_k1_FDX_s113
                                 + 1.08063 w_k1_IBM_s113
                                 + 1.54467 w_k1_INTC_s113
                                 + 1.22768 w_k1_KO_s113
                                 + 0.952059 w_k1_MAT_s113
                                 + 1.39535 w_k1_MSFT_s113
                                 + 1.12529 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.996095 w_k1_ADM_s113
                                 + 0.931877 w_k1_AXP_s113
                                 + 0.768707 w_k1_DIS_s113
                                 + 0.968976 w_k1_FDX_s113
                                 + 0.68899 w_k1_IBM_s113 + 1.072 w_k1_INTC_s113
                                 + 0.840826 w_k1_KO_s113
                                 + 0.673262 w_k1_MAT_s113
                                 + 0.928699 w_k1_MSFT_s113
                                 + 0.81917 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.14057 w_k1_ADM_s113
                                 + 1.04612 w_k1_AXP_s113
                                 + 1.08565 w_k1_DIS_s113
                                 + 1.14664 w_k1_FDX_s113 + 1.0479 w_k1_IBM_s113
                                 + 1.18818 w_k1_INTC_s113
                                 + 1.28179 w_k1_KO_s113 + 0.88247 w_k1_MAT_s113
                                 + 1.05683 w_k1_MSFT_s113
                                 + 0.981943 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.924802 w_k1_ADM_s114
                                 + 1.24026 w_k1_AXP_s114
                                 + 0.929633 w_k1_DIS_s114
                                 + 0.969701 w_k1_FDX_s114
                                 + 1.3033 w_k1_IBM_s114
                                 + 1.12724 w_k1_INTC_s114
                                 + 0.979487 w_k1_KO_s114
                                 + 1.12661 w_k1_MAT_s114
                                 + 0.850101 w_k1_MSFT_s114
                                 + 0.939908 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.1105 w_k1_ADM_s114
                                 + 1.33124 w_k1_AXP_s114
                                 + 1.19583 w_k1_DIS_s114
                                 + 1.21998 w_k1_FDX_s114
                                 + 0.984816 w_k1_IBM_s114
                                 + 0.998838 w_k1_INTC_s114
                                 + 1.11865 w_k1_KO_s114
                                 + 0.994493 w_k1_MAT_s114
                                 + 1.06254 w_k1_MSFT_s114
                                 + 1.17986 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.05765 w_k1_ADM_s114
                                 + 0.867852 w_k1_AXP_s114
                                 + 0.647143 w_k1_DIS_s114
                                 + 0.761953 w_k1_FDX_s114
                                 + 0.972563 w_k1_IBM_s114
                                 + 0.671537 w_k1_INTC_s114
                                 + 0.784846 w_k1_KO_s114
                                 + 0.905461 w_k1_MAT_s114
                                 + 0.757683 w_k1_MSFT_s114
                                 + 0.851731 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.01611 w_k1_ADM_s114
                                 + 0.933215 w_k1_AXP_s114
                                 + 0.649042 w_k1_DIS_s114
                                 + 0.961929 w_k1_FDX_s114
                                 + 0.690392 w_k1_IBM_s114
                                 + 1.03938 w_k1_INTC_s114
                                 + 0.826369 w_k1_KO_s114
                                 + 0.645464 w_k1_MAT_s114
                                 + 0.875571 w_k1_MSFT_s114
                                 + 0.842076 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.28607 w_k1_ADM_s114
                                 + 1.34407 w_k1_AXP_s114
                                 + 1.50168 w_k1_DIS_s114
                                 + 1.17475 w_k1_FDX_s114
                                 + 1.52966 w_k1_IBM_s114
                                 + 1.27921 w_k1_INTC_s114 + 1.4519 w_k1_KO_s114
                                 + 1.18005 w_k1_MAT_s114
                                 + 1.24911 w_k1_MSFT_s114
                                 + 1.03211 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.949709 w_k1_ADM_s114
                                 + 1.01034 w_k1_AXP_s114
                                 + 0.949377 w_k1_DIS_s114
                                 + 0.953298 w_k1_FDX_s114
                                 + 0.795943 w_k1_IBM_s114
                                 + 0.840449 w_k1_INTC_s114
                                 + 0.873786 w_k1_KO_s114
                                 + 0.74379 w_k1_MAT_s114
                                 + 0.830179 w_k1_MSFT_s114
                                 + 0.823389 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.686758 w_k1_ADM_s114
                                 + 0.786394 w_k1_AXP_s114
                                 + 1.01739 w_k1_DIS_s114
                                 + 0.819654 w_k1_FDX_s114
                                 + 0.903224 w_k1_IBM_s114
                                 + 0.901956 w_k1_INTC_s114
                                 + 1.05977 w_k1_KO_s114
                                 + 0.980014 w_k1_MAT_s114
                                 + 0.967965 w_k1_MSFT_s114
                                 + 0.935428 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.08386 w_k1_ADM_s114
                                 + 0.918951 w_k1_AXP_s114
                                 + 1.13862 w_k1_DIS_s114
                                 + 0.957533 w_k1_FDX_s114
                                 + 1.21413 w_k1_IBM_s114
                                 + 0.900551 w_k1_INTC_s114
                                 + 1.21946 w_k1_KO_s114
                                 + 0.959311 w_k1_MAT_s114
                                 + 1.0625 w_k1_MSFT_s114 + 1.02977 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.973306 w_k1_ADM_s115
                                 + 0.912737 w_k1_AXP_s115
                                 + 0.707985 w_k1_DIS_s115
                                 + 0.8451 w_k1_FDX_s115 + 0.69264 w_k1_IBM_s115
                                 + 0.865669 w_k1_INTC_s115
                                 + 0.766746 w_k1_KO_s115
                                 + 0.656149 w_k1_MAT_s115
                                 + 0.861807 w_k1_MSFT_s115
                                 + 0.965176 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.7573 w_k1_ADM_s115
                                 + 0.947232 w_k1_AXP_s115
                                 + 0.723904 w_k1_DIS_s115
                                 + 0.749183 w_k1_FDX_s115
                                 + 1.16652 w_k1_IBM_s115
                                 + 0.782592 w_k1_INTC_s115
                                 + 0.839137 w_k1_KO_s115
                                 + 0.994786 w_k1_MAT_s115
                                 + 0.810897 w_k1_MSFT_s115
                                 + 0.943536 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.21082 w_k1_ADM_s115
                                 + 1.06143 w_k1_AXP_s115 + 1.0327 w_k1_DIS_s115
                                 + 1.16206 w_k1_FDX_s115
                                 + 0.723553 w_k1_IBM_s115
                                 + 1.15285 w_k1_INTC_s115
                                 + 0.937894 w_k1_KO_s115
                                 + 0.873448 w_k1_MAT_s115
                                 + 1.17605 w_k1_MSFT_s115
                                 + 1.08735 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.650953 w_k1_ADM_s115
                                 + 0.925516 w_k1_AXP_s115
                                 + 0.981683 w_k1_DIS_s115
                                 + 0.854132 w_k1_FDX_s115
                                 + 0.975121 w_k1_IBM_s115
                                 + 0.97759 w_k1_INTC_s115
                                 + 0.958078 w_k1_KO_s115
                                 + 0.917401 w_k1_MAT_s115
                                 + 1.02064 w_k1_MSFT_s115
                                 + 0.993114 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.26153 w_k1_ADM_s115
                                 + 1.25293 w_k1_AXP_s115
                                 + 1.12142 w_k1_DIS_s115
                                 + 1.05607 w_k1_FDX_s115
                                 + 0.933886 w_k1_IBM_s115
                                 + 0.928858 w_k1_INTC_s115
                                 + 0.876649 w_k1_KO_s115
                                 + 0.725258 w_k1_MAT_s115
                                 + 0.912575 w_k1_MSFT_s115
                                 + 0.972195 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.3224 w_k1_ADM_s115
                                 + 1.59924 w_k1_AXP_s115
                                 + 1.63476 w_k1_DIS_s115
                                 + 1.31764 w_k1_FDX_s115
                                 + 0.906884 w_k1_IBM_s115
                                 + 1.11466 w_k1_INTC_s115
                                 + 1.20666 w_k1_KO_s115 + 1.57759 w_k1_MAT_s115
                                 + 1.35689 w_k1_MSFT_s115
                                 + 1.24127 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.998435 w_k1_ADM_s115
                                 + 1.53447 w_k1_AXP_s115
                                 + 1.29826 w_k1_DIS_s115
                                 + 1.12808 w_k1_FDX_s115
                                 + 1.37493 w_k1_IBM_s115
                                 + 1.57279 w_k1_INTC_s115
                                 + 1.36729 w_k1_KO_s115 + 1.49797 w_k1_MAT_s115
                                 + 1.32229 w_k1_MSFT_s115
                                 + 1.23616 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.09442 w_k1_ADM_s115
                                 + 1.15573 w_k1_AXP_s115
                                 + 1.20349 w_k1_DIS_s115 + 1.1124 w_k1_FDX_s115
                                 + 1.08836 w_k1_IBM_s115
                                 + 1.19483 w_k1_INTC_s115
                                 + 1.06317 w_k1_KO_s115 + 1.18996 w_k1_MAT_s115
                                 + 1.2819 w_k1_MSFT_s115 + 1.20362 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.857636 w_k1_ADM_s116
                                 + 1.23729 w_k1_AXP_s116
                                 + 0.759989 w_k1_DIS_s116
                                 + 0.951779 w_k1_FDX_s116
                                 + 0.964538 w_k1_IBM_s116
                                 + 1.0316 w_k1_INTC_s116
                                 + 0.844222 w_k1_KO_s116
                                 + 1.11753 w_k1_MAT_s116
                                 + 0.880039 w_k1_MSFT_s116
                                 + 0.983113 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.98245 w_k1_ADM_s116
                                 + 1.08018 w_k1_AXP_s116
                                 + 1.50479 w_k1_DIS_s116
                                 + 1.27219 w_k1_FDX_s116
                                 + 0.991179 w_k1_IBM_s116
                                 + 1.1835 w_k1_INTC_s116 + 1.1483 w_k1_KO_s116
                                 + 0.906686 w_k1_MAT_s116
                                 + 1.23343 w_k1_MSFT_s116
                                 + 1.14334 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.07354 w_k1_ADM_s116
                                 + 1.03384 w_k1_AXP_s116
                                 + 1.11543 w_k1_DIS_s116
                                 + 1.19587 w_k1_FDX_s116
                                 + 0.815575 w_k1_IBM_s116
                                 + 1.19958 w_k1_INTC_s116
                                 + 0.962444 w_k1_KO_s116
                                 + 0.551366 w_k1_MAT_s116
                                 + 1.04722 w_k1_MSFT_s116
                                 + 0.908489 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.12452 w_k1_ADM_s116
                                 + 1.20263 w_k1_AXP_s116
                                 + 1.25967 w_k1_DIS_s116
                                 + 1.04721 w_k1_FDX_s116
                                 + 1.43996 w_k1_IBM_s116
                                 + 1.06407 w_k1_INTC_s116
                                 + 1.21405 w_k1_KO_s116 + 1.46159 w_k1_MAT_s116
                                 + 1.1593 w_k1_MSFT_s116
                                 + 0.993317 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.881696 w_k1_ADM_s116
                                 + 0.738246 w_k1_AXP_s116
                                 + 0.740529 w_k1_DIS_s116
                                 + 0.788302 w_k1_FDX_s116
                                 + 0.838365 w_k1_IBM_s116
                                 + 0.779814 w_k1_INTC_s116
                                 + 0.847677 w_k1_KO_s116
                                 + 0.902506 w_k1_MAT_s116
                                 + 0.806874 w_k1_MSFT_s116
                                 + 0.946599 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.817031 w_k1_ADM_s116
                                 + 0.768585 w_k1_AXP_s116
                                 + 0.615398 w_k1_DIS_s116
                                 + 0.72281 w_k1_FDX_s116
                                 + 1.08584 w_k1_IBM_s116
                                 + 0.789378 w_k1_INTC_s116
                                 + 0.833908 w_k1_KO_s116
                                 + 1.13685 w_k1_MAT_s116
                                 + 0.683013 w_k1_MSFT_s116
                                 + 0.844653 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.00785 w_k1_ADM_s116
                                 + 0.738138 w_k1_AXP_s116
                                 + 1.05445 w_k1_DIS_s116
                                 + 1.01594 w_k1_FDX_s116
                                 + 0.848295 w_k1_IBM_s116
                                 + 0.960744 w_k1_INTC_s116
                                 + 1.04176 w_k1_KO_s116
                                 + 0.732441 w_k1_MAT_s116
                                 + 0.99629 w_k1_MSFT_s116
                                 + 1.08753 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.43869 w_k1_ADM_s116
                                 + 1.4709 w_k1_AXP_s116 + 1.66168 w_k1_DIS_s116
                                 + 1.35802 w_k1_FDX_s116
                                 + 1.25749 w_k1_IBM_s116
                                 + 1.09438 w_k1_INTC_s116
                                 + 1.46145 w_k1_KO_s116 + 1.59255 w_k1_MAT_s116
                                 + 1.26787 w_k1_MSFT_s116
                                 + 1.33747 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.19485 w_k1_ADM_s117
                                 + 1.02178 w_k1_AXP_s117
                                 + 1.07311 w_k1_DIS_s117
                                 + 1.06187 w_k1_FDX_s117 + 1.0429 w_k1_IBM_s117
                                 + 1.07094 w_k1_INTC_s117
                                 + 0.938482 w_k1_KO_s117
                                 + 0.960185 w_k1_MAT_s117
                                 + 1.05367 w_k1_MSFT_s117
                                 + 1.00047 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.995359 w_k1_ADM_s117
                                 + 1.02341 w_k1_AXP_s117
                                 + 1.05609 w_k1_DIS_s117
                                 + 1.07199 w_k1_FDX_s117
                                 + 1.27803 w_k1_IBM_s117
                                 + 0.881723 w_k1_INTC_s117
                                 + 1.06589 w_k1_KO_s117 + 1.49608 w_k1_MAT_s117
                                 + 1.09999 w_k1_MSFT_s117
                                 + 1.03435 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.14768 w_k1_ADM_s117
                                 + 1.19817 w_k1_AXP_s117
                                 + 1.14745 w_k1_DIS_s117
                                 + 1.12374 w_k1_FDX_s117
                                 + 1.69124 w_k1_IBM_s117
                                 + 0.95477 w_k1_INTC_s117
                                 + 0.97798 w_k1_KO_s117 + 1.79322 w_k1_MAT_s117
                                 + 0.932852 w_k1_MSFT_s117
                                 + 1.23309 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.992204 w_k1_ADM_s117
                                 + 0.772637 w_k1_AXP_s117
                                 + 0.768875 w_k1_DIS_s117
                                 + 0.926584 w_k1_FDX_s117
                                 + 0.69321 w_k1_IBM_s117
                                 + 0.689479 w_k1_INTC_s117
                                 + 0.664118 w_k1_KO_s117
                                 + 0.720004 w_k1_MAT_s117
                                 + 0.854078 w_k1_MSFT_s117
                                 + 0.918551 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.866991 w_k1_ADM_s117
                                 + 0.918349 w_k1_AXP_s117
                                 + 0.567326 w_k1_DIS_s117
                                 + 0.828855 w_k1_FDX_s117
                                 + 1.12518 w_k1_IBM_s117
                                 + 0.684318 w_k1_INTC_s117
                                 + 0.752882 w_k1_KO_s117
                                 + 1.26429 w_k1_MAT_s117
                                 + 0.687021 w_k1_MSFT_s117
                                 + 0.941293 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.835019 w_k1_ADM_s117
                                 + 0.565718 w_k1_AXP_s117
                                 + 0.75074 w_k1_DIS_s117
                                 + 0.861174 w_k1_FDX_s117
                                 + 0.897834 w_k1_IBM_s117
                                 + 0.698434 w_k1_INTC_s117
                                 + 0.716272 w_k1_KO_s117
                                 + 0.713848 w_k1_MAT_s117
                                 + 0.779255 w_k1_MSFT_s117
                                 + 0.860975 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.11612 w_k1_ADM_s117
                                 + 1.57706 w_k1_AXP_s117
                                 + 1.50264 w_k1_DIS_s117
                                 + 1.53599 w_k1_FDX_s117
                                 + 0.885299 w_k1_IBM_s117
                                 + 1.02702 w_k1_INTC_s117
                                 + 1.01507 w_k1_KO_s117 + 1.17212 w_k1_MAT_s117
                                 + 1.22814 w_k1_MSFT_s117
                                 + 1.30107 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.956405 w_k1_ADM_s117
                                 + 1.75816 w_k1_AXP_s117
                                 + 1.90418 w_k1_DIS_s117
                                 + 1.52719 w_k1_FDX_s117
                                 + 1.31987 w_k1_IBM_s117
                                 + 1.10988 w_k1_INTC_s117
                                 + 1.30956 w_k1_KO_s117 + 2.00527 w_k1_MAT_s117
                                 + 1.51459 w_k1_MSFT_s117
                                 + 1.32278 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.010304727749416 w_k1_ADM_s000
                                 + 0.0163543580492291 w_k1_ADM_s001
                                 + 0.0095461367731811 w_k1_ADM_s002
                                 + 0.025401958647539 w_k1_ADM_s003
                                 + 0.00955148651143511 w_k1_ADM_s004
                                 + 0.0301801285492686 w_k1_ADM_s005
                                 + 0.0122371954384327 w_k1_ADM_s006
                                 + 0.0132389189306596 w_k1_ADM_s007
                                 + 0.025449109059535 w_k1_ADM_s010
                                 + 0.068499898154917 w_k1_ADM_s011
                                 + 0.0183694504858476 w_k1_ADM_s012
                                 + 0.0165495104618717 w_k1_ADM_s013
                                 + 0.0455782065746427 w_k1_ADM_s014
                                 + 0.0552579068720198 w_k1_ADM_s015
                                 + 0.0160339912221304 w_k1_ADM_s016
                                 + 0.0555095615097098 w_k1_ADM_s017
                                 + 0.0893055424182431 w_k1_ADM_s100
                                 + 0.0351695334777559 w_k1_ADM_s101
                                 + 0.0950892433187581 w_k1_ADM_s102
                                 + 0.0217551392351834 w_k1_ADM_s103
                                 + 0.0429145966171582 w_k1_ADM_s104
                                 + 0.0311072240951202 w_k1_ADM_s105
                                 + 0.0792468921818844 w_k1_ADM_s106
                                 + 0.0260180583269863 w_k1_ADM_s107
                                 + 0.011079404705094 w_k1_ADM_s110
                                 + 0.0536879671405326 w_k1_ADM_s111
                                 + 0.00850162907925864 w_k1_ADM_s112
                                 + 0.0220088364987411 w_k1_ADM_s113
                                 + 0.0180354876096834 w_k1_ADM_s114
                                 + 0.0244178029845255 w_k1_ADM_s115
                                 + 0.020912650323889 w_k1_ADM_s116
                                 + 0.0148516219864288 w_k1_ADM_s117
                                 + 0.00989156094083068 w_k1_AXP_s000
                                 + 0.0166704342907365 w_k1_AXP_s001
                                 + 0.00942354811926607 w_k1_AXP_s002
                                 + 0.0249865764945165 w_k1_AXP_s003
                                 + 0.0105362441314522 w_k1_AXP_s004
                                 + 0.0321634680879202 w_k1_AXP_s005
                                 + 0.0148541046165934 w_k1_AXP_s006
                                 + 0.0121273920400562 w_k1_AXP_s007
                                 + 0.0268622988024469 w_k1_AXP_s010
                                 + 0.0648009373132276 w_k1_AXP_s011
                                 + 0.0188447874132182 w_k1_AXP_s012
                                 + 0.0131662201533154 w_k1_AXP_s013
                                 + 0.0416756493271963 w_k1_AXP_s014
                                 + 0.0489212731482852 w_k1_AXP_s015
                                 + 0.0160483331012397 w_k1_AXP_s016
                                 + 0.0623450053995368 w_k1_AXP_s017
                                 + 0.0883789670261829 w_k1_AXP_s100
                                 + 0.0349777822778096 w_k1_AXP_s101
                                 + 0.0996510238672288 w_k1_AXP_s102
                                 + 0.0256357862023433 w_k1_AXP_s103
                                 + 0.0456388491701543 w_k1_AXP_s104
                                 + 0.0315874948982231 w_k1_AXP_s105
                                 + 0.0786041918670646 w_k1_AXP_s106
                                 + 0.0270716182800932 w_k1_AXP_s107
                                 + 0.0123663100005477 w_k1_AXP_s110
                                 + 0.0433191922642748 w_k1_AXP_s111
                                 + 0.00858699110914502 w_k1_AXP_s112
                                 + 0.0206761894119011 w_k1_AXP_s113
                                 + 0.0187639702986742 w_k1_AXP_s114
                                 + 0.0278462508069784 w_k1_AXP_s115
                                 + 0.021540256618785 w_k1_AXP_s116
                                 + 0.0156196390635248 w_k1_AXP_s117
                                 + 0.01026912200331 w_k1_DIS_s000
                                 + 0.0179973214445005 w_k1_DIS_s001
                                 + 0.0097527623555758 w_k1_DIS_s002
                                 + 0.0247605453204487 w_k1_DIS_s003
                                 + 0.00943106157423322 w_k1_DIS_s004
                                 + 0.0290095146095111 w_k1_DIS_s005
                                 + 0.0141263402841769 w_k1_DIS_s006
                                 + 0.0129072585588574 w_k1_DIS_s007
                                 + 0.0280007465511214 w_k1_DIS_s010
                                 + 0.064929379759677 w_k1_DIS_s011
                                 + 0.01712329064685 w_k1_DIS_s012
                                 + 0.0141680845428828 w_k1_DIS_s013
                                 + 0.0445061650795936 w_k1_DIS_s014
                                 + 0.0514007500363596 w_k1_DIS_s015
                                 + 0.0174788279909648 w_k1_DIS_s016
                                 + 0.0602600610290622 w_k1_DIS_s017
                                 + 0.0945102419957363 w_k1_DIS_s100
                                 + 0.0352492688653207 w_k1_DIS_s101
                                 + 0.0991798721018381 w_k1_DIS_s102
                                 + 0.0246660779615359 w_k1_DIS_s103
                                 + 0.0434082018319888 w_k1_DIS_s104
                                 + 0.0336564278921049 w_k1_DIS_s105
                                 + 0.0892949835900992 w_k1_DIS_s106
                                 + 0.0294085071595638 w_k1_DIS_s107
                                 + 0.0129811881947032 w_k1_DIS_s110
                                 + 0.0447510332407762 w_k1_DIS_s111
                                 + 0.00932706223624186 w_k1_DIS_s112
                                 + 0.0206344747205773 w_k1_DIS_s113
                                 + 0.0168655263084278 w_k1_DIS_s114
                                 + 0.0250474356520155 w_k1_DIS_s115
                                 + 0.0220594122538054 w_k1_DIS_s116
                                 + 0.0154583180797656 w_k1_DIS_s117
                                 + 0.00958384690351797 w_k1_FDX_s000
                                 + 0.0172093557911607 w_k1_FDX_s001
                                 + 0.00988410731826097 w_k1_FDX_s002
                                 + 0.0230878035145017 w_k1_FDX_s003
                                 + 0.0091756367742868 w_k1_FDX_s004
                                 + 0.0297698022112914 w_k1_FDX_s005
                                 + 0.013951029357898 w_k1_FDX_s006
                                 + 0.0123486765723998 w_k1_FDX_s007
                                 + 0.0266972151979144 w_k1_FDX_s010
                                 + 0.0655088858370342 w_k1_FDX_s011
                                 + 0.0182928547281082 w_k1_FDX_s012
                                 + 0.0134311472195673 w_k1_FDX_s013
                                 + 0.0415659914793382 w_k1_FDX_s014
                                 + 0.053077449740746 w_k1_FDX_s015
                                 + 0.0167477348225823 w_k1_FDX_s016
                                 + 0.0600924906371518 w_k1_FDX_s017
                                 + 0.0896013729196789 w_k1_FDX_s100
                                 + 0.0348659537510939 w_k1_FDX_s101
                                 + 0.0958822849751716 w_k1_FDX_s102
                                 + 0.0252154088998848 w_k1_FDX_s103
                                 + 0.0437294122836049 w_k1_FDX_s104
                                 + 0.0299494526799617 w_k1_FDX_s105
                                 + 0.0848610634722161 w_k1_FDX_s106
                                 + 0.0284664791503151 w_k1_FDX_s107
                                 + 0.012405113823283 w_k1_FDX_s110
                                 + 0.0443657678441565 w_k1_FDX_s111
                                 + 0.00886752036740879 w_k1_FDX_s112
                                 + 0.0209369790962235 w_k1_FDX_s113
                                 + 0.0172853922434646 w_k1_FDX_s114
                                 + 0.0243313363376092 w_k1_FDX_s115
                                 + 0.0209522627964901 w_k1_FDX_s116
                                 + 0.0162070686185168 w_k1_FDX_s117
                                 + 0.00958302207382301 w_k1_IBM_s000
                                 + 0.0193465815487249 w_k1_IBM_s001
                                 + 0.0102041020602232 w_k1_IBM_s002
                                 + 0.0248569244116676 w_k1_IBM_s003
                                 + 0.0096144199805653 w_k1_IBM_s004
                                 + 0.0307399005905755 w_k1_IBM_s005
                                 + 0.0136660628321145 w_k1_IBM_s006
                                 + 0.0119937922929972 w_k1_IBM_s007
                                 + 0.0261223120674774 w_k1_IBM_s010
                                 + 0.0643823513028851 w_k1_IBM_s011
                                 + 0.0182199527271449 w_k1_IBM_s012
                                 + 0.0140188416229603 w_k1_IBM_s013
                                 + 0.0424994891863784 w_k1_IBM_s014
                                 + 0.0499743859542629 w_k1_IBM_s015
                                 + 0.0169926718391674 w_k1_IBM_s016
                                 + 0.063650587975628 w_k1_IBM_s017
                                 + 0.0882288740329581 w_k1_IBM_s100
                                 + 0.0354330930951676 w_k1_IBM_s101
                                 + 0.100088910517131 w_k1_IBM_s102
                                 + 0.0249951379734196 w_k1_IBM_s103
                                 + 0.0427507192220019 w_k1_IBM_s104
                                 + 0.0306794670351481 w_k1_IBM_s105
                                 + 0.0867775709017408 w_k1_IBM_s106
                                 + 0.027473786002698 w_k1_IBM_s107
                                 + 0.012651110021896 w_k1_IBM_s110
                                 + 0.0449303465563432 w_k1_IBM_s111
                                 + 0.00862213611086974 w_k1_IBM_s112
                                 + 0.0214635968787152 w_k1_IBM_s113
                                 + 0.0177147810234357 w_k1_IBM_s114
                                 + 0.0247907888981053 w_k1_IBM_s115
                                 + 0.0218208483684749 w_k1_IBM_s116
                                 + 0.0151291933289543 w_k1_IBM_s117
                                 + 0.0107227362831309 w_k1_INTC_s000
                                 + 0.0205439367496146 w_k1_INTC_s001
                                 + 0.00916437972365895 w_k1_INTC_s002
                                 + 0.0255193154644058 w_k1_INTC_s003
                                 + 0.0102978779812555 w_k1_INTC_s004
                                 + 0.0317991874399748 w_k1_INTC_s005
                                 + 0.0151478309223613 w_k1_INTC_s006
                                 + 0.0124757406983837 w_k1_INTC_s007
                                 + 0.0257442659152079 w_k1_INTC_s010
                                 + 0.064570897755342 w_k1_INTC_s011
                                 + 0.0191036216393098 w_k1_INTC_s012
                                 + 0.011881031408439 w_k1_INTC_s013
                                 + 0.0384806639451454 w_k1_INTC_s014
                                 + 0.0490864826379586 w_k1_INTC_s015
                                 + 0.0165556275412909 w_k1_INTC_s016
                                 + 0.0585910716405758 w_k1_INTC_s017
                                 + 0.0902876900472935 w_k1_INTC_s100
                                 + 0.0415305623956538 w_k1_INTC_s101
                                 + 0.0954879560865156 w_k1_INTC_s102
                                 + 0.0299736445846418 w_k1_INTC_s103
                                 + 0.0449006193249958 w_k1_INTC_s104
                                 + 0.0306670728286798 w_k1_INTC_s105
                                 + 0.0835512771045002 w_k1_INTC_s106
                                 + 0.0289794460409885 w_k1_INTC_s107
                                 + 0.0116498826735458 w_k1_INTC_s110
                                 + 0.046221454316817 w_k1_INTC_s111
                                 + 0.0100503773244888 w_k1_INTC_s112
                                 + 0.023137822533963 w_k1_INTC_s113
                                 + 0.0165758519134594 w_k1_INTC_s114
                                 + 0.0263549353879451 w_k1_INTC_s115
                                 + 0.0203956993883982 w_k1_INTC_s116
                                 + 0.0127244342768505 w_k1_INTC_s117
                                 + 0.00952512124061625 w_k1_KO_s000
                                 + 0.0163209204779446 w_k1_KO_s001
                                 + 0.00981987201904013 w_k1_KO_s002
                                 + 0.0235998907425955 w_k1_KO_s003
                                 + 0.00990532304968678 w_k1_KO_s004
                                 + 0.0299319526836326 w_k1_KO_s005
                                 + 0.0140863921535049 w_k1_KO_s006
                                 + 0.0124865161912415 w_k1_KO_s007
                                 + 0.0255471756124541 w_k1_KO_s010
                                 + 0.0627309016920092 w_k1_KO_s011
                                 + 0.0187609639525525 w_k1_KO_s012
                                 + 0.0137034158641542 w_k1_KO_s013
                                 + 0.0387651212865938 w_k1_KO_s014
                                 + 0.0482667550426858 w_k1_KO_s015
                                 + 0.0145873562179401 w_k1_KO_s016
                                 + 0.0615216675944683 w_k1_KO_s017
                                 + 0.0943489120706493 w_k1_KO_s100
                                 + 0.0327654968806665 w_k1_KO_s101
                                 + 0.0906748755328311 w_k1_KO_s102
                                 + 0.0248507266226431 w_k1_KO_s103
                                 + 0.0395294518851788 w_k1_KO_s104
                                 + 0.0298280707081978 w_k1_KO_s105
                                 + 0.0919255655971918 w_k1_KO_s106
                                 + 0.0251797078784315 w_k1_KO_s107
                                 + 0.0122071182936048 w_k1_KO_s110
                                 + 0.043836426599053 w_k1_KO_s111
                                 + 0.00821265511877917 w_k1_KO_s112
                                 + 0.0217967223266637 w_k1_KO_s113
                                 + 0.0174711951086975 w_k1_KO_s114
                                 + 0.0243661230533571 w_k1_KO_s115
                                 + 0.0214840194041937 w_k1_KO_s116
                                 + 0.0129975989181862 w_k1_KO_s117
                                 + 0.0095289803506188 w_k1_MAT_s000
                                 + 0.0136939947862403 w_k1_MAT_s001
                                 + 0.0118614650653646 w_k1_MAT_s002
                                 + 0.0226681631229238 w_k1_MAT_s003
                                 + 0.00974507658789549 w_k1_MAT_s004
                                 + 0.0313708426959749 w_k1_MAT_s005
                                 + 0.0171672670099445 w_k1_MAT_s006
                                 + 0.012202530479565 w_k1_MAT_s007
                                 + 0.0282597982773484 w_k1_MAT_s010
                                 + 0.0623669377320393 w_k1_MAT_s011
                                 + 0.0170049114675702 w_k1_MAT_s012
                                 + 0.0129476017483283 w_k1_MAT_s013
                                 + 0.0505501091202361 w_k1_MAT_s014
                                 + 0.0452010524067582 w_k1_MAT_s015
                                 + 0.0176975070471699 w_k1_MAT_s016
                                 + 0.0704867898761798 w_k1_MAT_s017
                                 + 0.0935801349764179 w_k1_MAT_s100
                                 + 0.0351135044931615 w_k1_MAT_s101
                                 + 0.0975635981156445 w_k1_MAT_s102
                                 + 0.0286568445926729 w_k1_MAT_s103
                                 + 0.0375110981491584 w_k1_MAT_s104
                                 + 0.0243879845464135 w_k1_MAT_s105
                                 + 0.0838963167103212 w_k1_MAT_s106
                                 + 0.0265278345211541 w_k1_MAT_s107
                                 + 0.0108387357462965 w_k1_MAT_s110
                                 + 0.0429715591073791 w_k1_MAT_s111
                                 + 0.00895719333450369 w_k1_MAT_s112
                                 + 0.0187864362511575 w_k1_MAT_s113
                                 + 0.0160800859428163 w_k1_MAT_s114
                                 + 0.0254747751259421 w_k1_MAT_s115
                                 + 0.0229153953582287 w_k1_MAT_s116
                                 + 0.0167989028001449 w_k1_MAT_s117
                                 + 0.00946279974827959 w_k1_MSFT_s000
                                 + 0.0189472661271832 w_k1_MSFT_s001
                                 + 0.0109832345449521 w_k1_MSFT_s002
                                 + 0.0242693386441833 w_k1_MSFT_s003
                                 + 0.00908300882493946 w_k1_MSFT_s004
                                 + 0.0299624315204894 w_k1_MSFT_s005
                                 + 0.0141312394428911 w_k1_MSFT_s006
                                 + 0.0131426895302924 w_k1_MSFT_s007
                                 + 0.0270250716091037 w_k1_MSFT_s010
                                 + 0.0674028506399939 w_k1_MSFT_s011
                                 + 0.0187853728305314 w_k1_MSFT_s012
                                 + 0.0142397168600907 w_k1_MSFT_s013
                                 + 0.0400146267254541 w_k1_MSFT_s014
                                 + 0.0525781629567761 w_k1_MSFT_s015
                                 + 0.017317344002707 w_k1_MSFT_s016
                                 + 0.0579770788626196 w_k1_MSFT_s017
                                 + 0.102628100415152 w_k1_MSFT_s100
                                 + 0.0363040295659739 w_k1_MSFT_s101
                                 + 0.0887980474482325 w_k1_MSFT_s102
                                 + 0.0251744733810344 w_k1_MSFT_s103
                                 + 0.0409310298615194 w_k1_MSFT_s104
                                 + 0.0294936463241441 w_k1_MSFT_s105
                                 + 0.0849897907862442 w_k1_MSFT_s106
                                 + 0.0271913497630805 w_k1_MSFT_s107
                                 + 0.0113025030244762 w_k1_MSFT_s110
                                 + 0.0414905247508813 w_k1_MSFT_s111
                                 + 0.00914314590038669 w_k1_MSFT_s112
                                 + 0.0213805012719614 w_k1_MSFT_s113
                                 + 0.0163071587767843 w_k1_MSFT_s114
                                 + 0.026407738912709 w_k1_MSFT_s115
                                 + 0.020639091891247 w_k1_MSFT_s116
                                 + 0.0145463311589677 w_k1_MSFT_s117
                                 + 0.00912316442808773 w_k1_PG_s000
                                 + 0.0173138844792887 w_k1_PG_s001
                                 + 0.0099633088685405 w_k1_PG_s002
                                 + 0.0227400640567208 w_k1_PG_s003
                                 + 0.00932664193664339 w_k1_PG_s004
                                 + 0.0318062385461841 w_k1_PG_s005
                                 + 0.0138918776966633 w_k1_PG_s006
                                 + 0.0134458580556279 w_k1_PG_s007
                                 + 0.0245507661854294 w_k1_PG_s010
                                 + 0.061676794514011 w_k1_PG_s011
                                 + 0.0170758989082587 w_k1_PG_s012
                                 + 0.0140780246898505 w_k1_PG_s013
                                 + 0.0414571995163022 w_k1_PG_s014
                                 + 0.0511774680693586 w_k1_PG_s015
                                 + 0.0155580039006888 w_k1_PG_s016
                                 + 0.0611132707190243 w_k1_PG_s017
                                 + 0.0942088794501706 w_k1_PG_s100
                                 + 0.032989337261696 w_k1_PG_s101
                                 + 0.0957877637928475 w_k1_PG_s102
                                 + 0.023660780117936 w_k1_PG_s103
                                 + 0.0433638478044425 w_k1_PG_s104
                                 + 0.0306583735218646 w_k1_PG_s105
                                 + 0.0882820641522807 w_k1_PG_s106
                                 + 0.0245714373194272 w_k1_PG_s107
                                 + 0.0113985451842575 w_k1_PG_s110
                                 + 0.0433933259262017 w_k1_PG_s111
                                 + 0.00862170848231986 w_k1_PG_s112
                                 + 0.0198868534525103 w_k1_PG_s113
                                 + 0.0167696680888971 w_k1_PG_s114
                                 + 0.0266254580624741 w_k1_PG_s115
                                 + 0.0211629631402918 w_k1_PG_s116
                                 + 0.0155259856126949 w_k1_PG_s117 >= 1.12
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
