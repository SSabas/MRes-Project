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
 w_k1_ADM_s0_return_constraint:  0.84567 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.0912 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.05595 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.922699 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.17781 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.886028 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.907024 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.38019 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.860085 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.07143 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.938732 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.792759 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.816842 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.973621 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.01698 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.982274 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.05209 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.26608 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.28915 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.08783 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.814166 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.820732 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.17347 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.840485 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.897168 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.950108 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.926641 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.53973 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.11198 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.06645 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.873759 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.0331 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.786139 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.16677 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.06635 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.30685 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.942616 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.900827 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.786093 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.12087 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.0855 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.825517 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.36071 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.862197 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.948396 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.58036 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.951386 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.11349 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.91761 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.709553 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.65352 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.34269 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.01143 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.795575 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.15212 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.34567 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.54155 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.948867 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.698531 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.0245 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.37055 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.78551 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.896721 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.918711 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.19558 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.46177 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.18158 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.16821 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.700293 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.899942 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.918159 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.37813 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.16458 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.36475 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.975011 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.719349 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.702763 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.14266 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.25742 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.8603 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.49915 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.902731 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.810131 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.60546 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.781865 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.18443 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.867801 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.632672 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.596113 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.1871 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.952081 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.887538 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.752748 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.36489 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.45689 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.008 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.620692 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.897352 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.39502 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.711295 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.939794 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.781583 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.816523 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.53506 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.999295 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.01216 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.581162 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.960778 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.673996 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.30451 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.3115 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.20996 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.768951 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.622985 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.805812 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.09152 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.06546 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.896597 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.28559 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.929347 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.948617 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.46332 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.877608 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.19452 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.0329 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.671407 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.746672 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.01413 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.94541 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.943663 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.935991 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.32338 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.27043 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.15012 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.707461 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.839485 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.32661 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.88285 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.854403 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.919899 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.84936 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.1919 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.926015 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.01581 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.700402 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.838483 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.831014 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.0296 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.16436 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.23846 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.947913 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.789192 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.888233 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.09239 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.23154 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.761165 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.31565 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.910805 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.01909 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.19536 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.864512 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.733552 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.616181 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.19804 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.656153 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.45288 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.25111 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.824716 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.927787 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.95679 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.41692 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.579751 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.874802 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.72622 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.93861 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.617738 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.17921 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.878746 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.25732 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.69066 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.94546 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.886082 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.764519 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.24726 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.96564 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.75263 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.00325 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.879338 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.730691 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.525092 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.860614 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.0666 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.952126 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.888226 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.15155 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.937938 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.12447 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.25916 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.916205 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.24457 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.18724 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.785805 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.83552 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.17919 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.871928 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.898411 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.854181 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.2678 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.1825 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.21657 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.814637 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.866096 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.26215 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.844881 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.973824 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.10602 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.998328 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.14925 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.967614 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.12141 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.715899 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.02033 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.95869 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.956346 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.13543 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.10489 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.861038 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.756389 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.847199 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.10275 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.12062 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.850176 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.2277 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.00701 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.00418 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.22491 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.858007 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.99312 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.788274 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.903574 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.704487 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.24194 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.04473 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.883424 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.789567 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.14687 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.16623 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.800548 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.746564 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11319 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.17412 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.736883 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.02206 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.924942 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.995889 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.69686 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.42764 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.956038 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.698104 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.958444 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.753978 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.22784 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.998422 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.865431 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.716809 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.575475 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.852221 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.10725 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.10989 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.695931 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.53163 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.979856 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.01002 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.51142 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.984733 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.830872 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.854882 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.19303 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.618758 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.73446 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.1834 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.810735 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.0293 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.97054 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.45932 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.544127 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.80825 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.06793 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.858606 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.567866 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.959815 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.754569 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.2298 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.83247 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.54582 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.980276 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.639741 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.11759 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.97381 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.89585 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.780243 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.952818 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.840306 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.470508 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.88119 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.03944 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.00201 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.882345 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.25253 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.938902 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.945299 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.21623 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.962512 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.21409 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.01164 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.787085 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.702098 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.07712 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.920102 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.935624 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.750292 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.1794 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.08247 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.16544 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.742969 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.832522 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.34451 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.881581 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.961883 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.993328 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.86238 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.19466 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.02755 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.09485 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.778153 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.04212 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.832429 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.07176 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.21716 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.2566 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.951087 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.804704 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.866018 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.06197 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.1007 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.913164 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.20069 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.927256 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.945346 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.17941 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.897812 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.00029 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.902097 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.855346 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.778181 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.0538 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.03937 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.981607 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.901456 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.13851 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.18475 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.952731 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.768687 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.959645 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.14136 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.839422 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.933977 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.916577 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.910783 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.30222 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.11959 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.972419 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.80354 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.0034 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.950894 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.1198 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.04942 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.12306 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.906962 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.86726 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.289997971680945 z_k1_s0000
                                 - 0.0769995876526086 z_k1_s0001
                                 - 0.163999418354777 z_k1_s0002
                                 - 0.0949995273842133 z_k1_s0003
                                 - 0.345998630041864 z_k1_s0004
                                 - 0.267998549606555 z_k1_s0005
                                 - 0.103999570449663 z_k1_s0006
                                 - 0.119999403011638 z_k1_s0007
                                 - 0.191999752471745 z_k1_s0010
                                 - 0.120999919838854 z_k1_s0011
                                 - 0.433999368149779 z_k1_s0012
                                 - 0.27399864109135 z_k1_s0013
                                 - 0.261998978561398 z_k1_s0014
                                 - 0.310998907223469 z_k1_s0015
                                 - 0.1549997103393 z_k1_s0016
                                 - 0.49099832516624 z_k1_s0017
                                 - 0.302999543791842 z_k1_s0020
                                 - 0.284999635893284 z_k1_s0021
                                 - 0.0829999400320558 z_k1_s0022
                                 - 0.108999952885455 z_k1_s0023
                                 - 0.260000128533991 z_k1_s0024
                                 - 0.0759999466715202 z_k1_s0025
                                 - 0.0979999070474988 z_k1_s0026
                                 - 0.0989998873418705 z_k1_s0027
                                 - 0.180000090510288 z_k1_s0030
                                 - 0.307999574152775 z_k1_s0031
                                 - 0.507999674719762 z_k1_s0032
                                 - 0.122000019865835 z_k1_s0033
                                 - 0.237999953754604 z_k1_s0034
                                 - 0.171000168944829 z_k1_s0035
                                 - 0.168000125956296 z_k1_s0036
                                 - 0.379000937214935 z_k1_s0037
                                 - 0.336000514977618 z_k1_s0040
                                 - 0.101000094199787 z_k1_s0041
                                 - 0.0780000993483649 z_k1_s0042
                                 - 0.157000745629692 z_k1_s0043
                                 - 0.113999992924452 z_k1_s0044
                                 - 0.120000022320492 z_k1_s0045
                                 - 0.192000092272817 z_k1_s0046
                                 - 0.31599947432365 z_k1_s0047
                                 - 0.0700000045881423 z_k1_s0050
                                 - 0.139999837376263 z_k1_s0051
                                 - 0.141999589624465 z_k1_s0052
                                 - 0.140999713500364 z_k1_s0053
                                 - 0.0479999810575805 z_k1_s0054
                                 - 0.07400002448461 z_k1_s0055
                                 - 0.149000440493385 z_k1_s0056
                                 - 0.0950002577910815 z_k1_s0057
                                 - 0.16299963852871 z_k1_s0060
                                 - 0.160999658055968 z_k1_s0061
                                 - 0.128999970492107 z_k1_s0062
                                 - 0.236000155784059 z_k1_s0063
                                 - 0.166999599474192 z_k1_s0064
                                 - 0.111000023237406 z_k1_s0065
                                 - 0.17899948231064 z_k1_s0066
                                 - 0.0840000408553397 z_k1_s0067
                                 - 0.545998786694998 z_k1_s0070
                                 - 0.388001208324969 z_k1_s0071
                                 - 0.12299985357632 z_k1_s0072
                                 - 0.405001370921659 z_k1_s0073
                                 - 0.514001162100691 z_k1_s0074
                                 - 0.312998703339927 z_k1_s0075
                                 - 0.485999821706356 z_k1_s0076
                                 - 0.264000004449112 z_k1_s0077
                                 - 0.126999993315075 z_k1_s0100
                                 - 0.168000082208152 z_k1_s0101
                                 - 0.139000062722815 z_k1_s0102
                                 - 0.363999496167504 z_k1_s0103
                                 - 0.0870000584560112 z_k1_s0104
                                 - 0.2909999930873 z_k1_s0105
                                 - 0.423999843206204 z_k1_s0106
                                 - 0.178999463881755 z_k1_s0107
                                 - 0.139000046842651 z_k1_s0110
                                 - 0.249999815918333 z_k1_s0111
                                 - 0.525000271628757 z_k1_s0112
                                 - 0.317000214160621 z_k1_s0113
                                 - 0.380000948836347 z_k1_s0114
                                 - 0.211000134243186 z_k1_s0115
                                 - 0.141000098026019 z_k1_s0116
                                 - 0.230999768476515 z_k1_s0117
                                 - 0.0980003577039268 z_k1_s0120
                                 - 0.0630001105952516 z_k1_s0121
                                 - 0.0470000453741949 z_k1_s0122
                                 - 0.0380000295748732 z_k1_s0123
                                 - 0.053000055907076 z_k1_s0124
                                 - 0.0849999635489468 z_k1_s0125
                                 - 0.110000378769689 z_k1_s0126
                                 - 0.0630001105952516 z_k1_s0127
                                 - 0.201998831923928 z_k1_s0130
                                 - 0.301999547150662 z_k1_s0131
                                 - 0.233999753354912 z_k1_s0132
                                 - 0.345999043372281 z_k1_s0133
                                 - 0.128999710665235 z_k1_s0134
                                 - 0.079999785183172 z_k1_s0135
                                 - 0.0979998509641269 z_k1_s0136
                                 - 0.182999121009885 z_k1_s0137
                                 - 0.114000588537096 z_k1_s0140
                                 - 0.138000656281662 z_k1_s0141
                                 - 0.131999839094295 z_k1_s0142
                                 - 0.28100021521674 z_k1_s0143
                                 - 0.0630000444542812 z_k1_s0144
                                 - 0.0670001268784844 z_k1_s0145
                                 - 0.145000186998078 z_k1_s0146
                                 - 0.126999869439246 z_k1_s0147
                                 - 0.0460000725790245 z_k1_s0150
                                 - 0.0290000267780586 z_k1_s0151
                                 - 0.0799998731806175 z_k1_s0152
                                 - 0.0860003001696722 z_k1_s0153
                                 - 0.0890000286636346 z_k1_s0154
                                 - 0.0390000594256922 z_k1_s0155
                                 - 0.0260000072837572 z_k1_s0156
                                 - 0.0900000998284771 z_k1_s0157
                                 - 0.0940000856840107 z_k1_s0160
                                 - 0.219000661064364 z_k1_s0161
                                 - 0.25200035555303 z_k1_s0162
                                 - 0.177000601805526 z_k1_s0163
                                 - 0.153000121985656 z_k1_s0164
                                 - 0.180999668590958 z_k1_s0165
                                 - 0.4220001259811 z_k1_s0166
                                 - 0.145000181112628 z_k1_s0167
                                 - 0.0690001089371222 z_k1_s0170
                                 - 0.0409999978350803 z_k1_s0171
                                 - 0.0509998962143361 z_k1_s0172
                                 - 0.0389999898992202 z_k1_s0173
                                 - 0.14399981778169 z_k1_s0174
                                 - 0.0430000057709405 z_k1_s0175
                                 - 0.0459999941247232 z_k1_s0176
                                 - 0.0380000094812976 z_k1_s0177
                                 - 1.1590000656146 z_k1_s1000
                                 - 0.343999743212724 z_k1_s1001
                                 - 0.300999775311134 z_k1_s1002
                                 - 0.353999716152637 z_k1_s1003
                                 - 0.335999680600827 z_k1_s1004
                                 - 0.331999859944797 z_k1_s1005
                                 - 1.10699902668752 z_k1_s1006
                                 - 0.28000008491722 z_k1_s1007
                                 - 0.544999227945228 z_k1_s1010
                                 - 0.419000996431204 z_k1_s1011
                                 - 0.316001445733283 z_k1_s1012
                                 - 0.351000103931667 z_k1_s1013
                                 - 0.368000327056551 z_k1_s1014
                                 - 0.241999952142637 z_k1_s1015
                                 - 0.393999767718185 z_k1_s1016
                                 - 0.426998940019496 z_k1_s1017
                                 - 0.187999942824207 z_k1_s1020
                                 - 0.827000280821922 z_k1_s1021
                                 - 0.829999196884008 z_k1_s1022
                                 - 0.293999719680596 z_k1_s1023
                                 - 0.249999857885447 z_k1_s1024
                                 - 0.374999096628846 z_k1_s1025
                                 - 0.491998235009614 z_k1_s1026
                                 - 0.194999944366594 z_k1_s1027
                                 - 0.236999443484394 z_k1_s1030
                                 - 0.354000549648569 z_k1_s1031
                                 - 0.317000184032536 z_k1_s1032
                                 - 0.332999592541633 z_k1_s1033
                                 - 0.0760000563702699 z_k1_s1034
                                 - 0.136000149531044 z_k1_s1035
                                 - 0.279999818416504 z_k1_s1036
                                 - 0.116000056844075 z_k1_s1037
                                 - 0.0780001254243638 z_k1_s1040
                                 - 0.0910000906783647 z_k1_s1041
                                 - 0.109000102499353 z_k1_s1042
                                 - 0.103000135659064 z_k1_s1043
                                 - 0.260999893154037 z_k1_s1044
                                 - 0.131999715686852 z_k1_s1045
                                 - 0.148000220861595 z_k1_s1046
                                 - 0.190999909016925 z_k1_s1047
                                 - 0.104000099436971 z_k1_s1050
                                 - 0.20500018189979 z_k1_s1051
                                 - 0.073000021837458 z_k1_s1052
                                 - 0.27300007764693 z_k1_s1053
                                 - 0.182999568422349 z_k1_s1054
                                 - 0.089000014566208 z_k1_s1055
                                 - 0.230000812351385 z_k1_s1056
                                 - 0.310000775995134 z_k1_s1057
                                 - 0.379999716456248 z_k1_s1060
                                 - 0.094000072860266 z_k1_s1061
                                 - 0.206999637543219 z_k1_s1062
                                 - 0.088000070837271 z_k1_s1063
                                 - 0.0770000465451082 z_k1_s1064
                                 - 0.0979999918755751 z_k1_s1065
                                 - 0.17400042916695 z_k1_s1066
                                 - 0.116999977698388 z_k1_s1067
                                 - 0.283000386194175 z_k1_s1070
                                 - 0.165000091533579 z_k1_s1071
                                 - 0.578001695846214 z_k1_s1072
                                 - 0.21200011760678 z_k1_s1073
                                 - 0.141000078219604 z_k1_s1074
                                 - 0.485001415054341 z_k1_s1075
                                 - 0.201000111504542 z_k1_s1076
                                 - 0.227000125928015 z_k1_s1077
                                 - 0.615000943779335 z_k1_s1100
                                 - 1.13899832828391 z_k1_s1101
                                 - 0.529999735118446 z_k1_s1102
                                 - 0.450999767271927 z_k1_s1103
                                 - 0.455999968934273 z_k1_s1104
                                 - 0.828998033012944 z_k1_s1105
                                 - 0.758997429339096 z_k1_s1106
                                 - 0.770001091414818 z_k1_s1107
                                 - 0.663998997737908 z_k1_s1110
                                 - 0.956995907590019 z_k1_s1111
                                 - 0.576998954185096 z_k1_s1112
                                 - 1.21199367068055 z_k1_s1113
                                 - 1.08900227917485 z_k1_s1114
                                 - 0.716998120481072 z_k1_s1115
                                 - 1.11299346348283 z_k1_s1116
                                 - 2.81399037407496 z_k1_s1117
                                 - 0.342999768634848 z_k1_s1120
                                 - 0.218999743012071 z_k1_s1121
                                 - 0.832000769610759 z_k1_s1122
                                 - 0.587998640859958 z_k1_s1123
                                 - 0.466999794257626 z_k1_s1124
                                 - 0.309999853506992 z_k1_s1125
                                 - 0.55700147687862 z_k1_s1126
                                 - 0.895001526345551 z_k1_s1127
                                 - 1.11899830596104 z_k1_s1130
                                 - 0.88800184017851 z_k1_s1131
                                 - 0.8619990166029 z_k1_s1132
                                 - 0.782998365049053 z_k1_s1133
                                 - 0.467999293842683 z_k1_s1134
                                 - 1.48899616137337 z_k1_s1135
                                 - 1.94899774409547 z_k1_s1136
                                 - 1.17599629959749 z_k1_s1137
                                 - 0.868002096403584 z_k1_s1140
                                 - 1.81699963842783 z_k1_s1141
                                 - 1.5810007184419 z_k1_s1142
                                 - 0.765001712636473 z_k1_s1143
                                 - 1.80700676429298 z_k1_s1144
                                 - 1.16100705407082 z_k1_s1145
                                 - 2.00000703042832 z_k1_s1146
                                 - 0.851002275345519 z_k1_s1147
                                 - 0.74199830499259 z_k1_s1150
                                 - 0.895998382516641 z_k1_s1151
                                 - 1.05499738332342 z_k1_s1152
                                 - 0.776998682356455 z_k1_s1153
                                 - 1.98499695212624 z_k1_s1154
                                 - 2.38499985157313 z_k1_s1155
                                 - 1.13700030256056 z_k1_s1156
                                 - 0.915998032839937 z_k1_s1157
                                 - 0.803001395986245 z_k1_s1160
                                 - 0.840001017992916 z_k1_s1161
                                 - 0.582997746701106 z_k1_s1162
                                 - 0.790001803065086 z_k1_s1163
                                 - 0.395999972510754 z_k1_s1164
                                 - 0.669002079259124 z_k1_s1165
                                 - 0.567998997023141 z_k1_s1166
                                 - 0.424999923089049 z_k1_s1167
                                 - 1.04999911568473 z_k1_s1170
                                 - 0.266000103113516 z_k1_s1171
                                 - 0.444999986347394 z_k1_s1172
                                 - 0.409999867124547 z_k1_s1173
                                 - 0.478000685060455 z_k1_s1174
                                 - 0.5419982009761 z_k1_s1175
                                 - 0.705997969385892 z_k1_s1176
                                 - 0.563999856384992 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.06452 w_k1_ADM_s000
                                 + 0.79686 w_k1_AXP_s000
                                 + 0.823202 w_k1_DIS_s000
                                 + 0.866875 w_k1_FDX_s000
                                 + 0.769673 w_k1_IBM_s000
                                 + 0.89923 w_k1_INTC_s000
                                 + 0.800352 w_k1_KO_s000
                                 + 0.930219 w_k1_MAT_s000
                                 + 0.847894 w_k1_MSFT_s000
                                 + 0.883976 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.706093 w_k1_ADM_s000
                                 + 0.610922 w_k1_AXP_s000
                                 + 0.510704 w_k1_DIS_s000
                                 + 0.619273 w_k1_FDX_s000
                                 + 0.796652 w_k1_IBM_s000
                                 + 0.655126 w_k1_INTC_s000
                                 + 0.599617 w_k1_KO_s000
                                 + 0.725533 w_k1_MAT_s000
                                 + 0.609793 w_k1_MSFT_s000
                                 + 0.786272 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.730939 w_k1_ADM_s000
                                 + 1.00115 w_k1_AXP_s000
                                 + 0.860189 w_k1_DIS_s000
                                 + 0.869682 w_k1_FDX_s000
                                 + 1.07838 w_k1_IBM_s000
                                 + 0.716744 w_k1_INTC_s000
                                 + 0.804571 w_k1_KO_s000
                                 + 1.37727 w_k1_MAT_s000
                                 + 0.831392 w_k1_MSFT_s000
                                 + 0.852018 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.0286 w_k1_ADM_s000
                                 + 1.22396 w_k1_AXP_s000
                                 + 1.60005 w_k1_DIS_s000
                                 + 1.21129 w_k1_FDX_s000
                                 + 1.29813 w_k1_IBM_s000
                                 + 1.17014 w_k1_INTC_s000
                                 + 1.23737 w_k1_KO_s000 + 1.40186 w_k1_MAT_s000
                                 + 1.27165 w_k1_MSFT_s000
                                 + 1.11606 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.11324 w_k1_ADM_s000
                                 + 1.23418 w_k1_AXP_s000
                                 + 1.08692 w_k1_DIS_s000
                                 + 1.07385 w_k1_FDX_s000
                                 + 0.92905 w_k1_IBM_s000
                                 + 1.02071 w_k1_INTC_s000
                                 + 0.903135 w_k1_KO_s000 + 1.0166 w_k1_MAT_s000
                                 + 0.918747 w_k1_MSFT_s000
                                 + 1.11552 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.0181 w_k1_ADM_s000
                                 + 0.763028 w_k1_AXP_s000
                                 + 1.06187 w_k1_DIS_s000
                                 + 0.884574 w_k1_FDX_s000
                                 + 1.19702 w_k1_IBM_s000
                                 + 0.915347 w_k1_INTC_s000
                                 + 1.07366 w_k1_KO_s000 + 1.16794 w_k1_MAT_s000
                                 + 0.999122 w_k1_MSFT_s000
                                 + 0.945075 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.986952 w_k1_ADM_s000
                                 + 1.09034 w_k1_AXP_s000
                                 + 1.37382 w_k1_DIS_s000 + 1.2955 w_k1_FDX_s000
                                 + 0.758764 w_k1_IBM_s000
                                 + 1.08101 w_k1_INTC_s000
                                 + 0.976097 w_k1_KO_s000
                                 + 0.909261 w_k1_MAT_s000
                                 + 1.33779 w_k1_MSFT_s000
                                 + 1.01516 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.29891 w_k1_ADM_s000
                                 + 1.49546 w_k1_AXP_s000
                                 + 1.33325 w_k1_DIS_s000
                                 + 1.13501 w_k1_FDX_s000
                                 + 1.43664 w_k1_IBM_s000
                                 + 1.02556 w_k1_INTC_s000
                                 + 1.26978 w_k1_KO_s000 + 2.21341 w_k1_MAT_s000
                                 + 1.03881 w_k1_MSFT_s000
                                 + 1.18927 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.872339 w_k1_ADM_s001
                                 + 0.616881 w_k1_AXP_s001
                                 + 0.967595 w_k1_DIS_s001
                                 + 0.785349 w_k1_FDX_s001
                                 + 0.724071 w_k1_IBM_s001
                                 + 1.03425 w_k1_INTC_s001
                                 + 0.961919 w_k1_KO_s001
                                 + 0.615373 w_k1_MAT_s001
                                 + 1.00975 w_k1_MSFT_s001
                                 + 0.954683 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.928865 w_k1_ADM_s001
                                 + 0.683268 w_k1_AXP_s001
                                 + 0.565728 w_k1_DIS_s001
                                 + 0.754424 w_k1_FDX_s001
                                 + 0.508488 w_k1_IBM_s001
                                 + 0.882048 w_k1_INTC_s001
                                 + 0.644285 w_k1_KO_s001
                                 + 0.53703 w_k1_MAT_s001
                                 + 0.788006 w_k1_MSFT_s001
                                 + 0.898649 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.16252 w_k1_ADM_s001
                                 + 1.04245 w_k1_AXP_s001
                                 + 1.04554 w_k1_DIS_s001
                                 + 1.09116 w_k1_FDX_s001
                                 + 0.706994 w_k1_IBM_s001
                                 + 1.12759 w_k1_INTC_s001
                                 + 1.02658 w_k1_KO_s001
                                 + 0.737703 w_k1_MAT_s001
                                 + 1.23592 w_k1_MSFT_s001
                                 + 1.10641 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.09792 w_k1_ADM_s001
                                 + 1.26921 w_k1_AXP_s001 + 1.555 w_k1_DIS_s001
                                 + 1.27568 w_k1_FDX_s001
                                 + 1.10172 w_k1_IBM_s001
                                 + 1.61189 w_k1_INTC_s001
                                 + 1.43236 w_k1_KO_s001
                                 + 0.920626 w_k1_MAT_s001
                                 + 1.51538 w_k1_MSFT_s001
                                 + 1.21388 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.743017 w_k1_ADM_s001
                                 + 0.729795 w_k1_AXP_s001
                                 + 0.539181 w_k1_DIS_s001
                                 + 0.715919 w_k1_FDX_s001
                                 + 0.81167 w_k1_IBM_s001
                                 + 0.879646 w_k1_INTC_s001
                                 + 0.870092 w_k1_KO_s001
                                 + 0.92365 w_k1_MAT_s001
                                 + 0.833059 w_k1_MSFT_s001
                                 + 0.864663 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.865876 w_k1_ADM_s001
                                 + 0.931462 w_k1_AXP_s001
                                 + 0.841023 w_k1_DIS_s001
                                 + 0.887893 w_k1_FDX_s001
                                 + 0.850575 w_k1_IBM_s001
                                 + 0.956975 w_k1_INTC_s001
                                 + 0.905791 w_k1_KO_s001
                                 + 0.925337 w_k1_MAT_s001
                                 + 0.958734 w_k1_MSFT_s001
                                 + 0.968545 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.14454 w_k1_ADM_s001
                                 + 1.62043 w_k1_AXP_s001
                                 + 1.46349 w_k1_DIS_s001
                                 + 1.07352 w_k1_FDX_s001 + 1.9401 w_k1_IBM_s001
                                 + 1.3966 w_k1_INTC_s001 + 1.78807 w_k1_KO_s001
                                 + 2.00181 w_k1_MAT_s001
                                 + 1.27817 w_k1_MSFT_s001
                                 + 1.23957 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.907869 w_k1_ADM_s001
                                 + 0.934011 w_k1_AXP_s001
                                 + 0.945191 w_k1_DIS_s001
                                 + 0.91221 w_k1_FDX_s001
                                 + 1.17509 w_k1_IBM_s001
                                 + 1.10376 w_k1_INTC_s001
                                 + 1.26486 w_k1_KO_s001 + 1.58481 w_k1_MAT_s001
                                 + 1.02204 w_k1_MSFT_s001
                                 + 1.04207 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.794291 w_k1_ADM_s002
                                 + 0.823811 w_k1_AXP_s002
                                 + 0.739498 w_k1_DIS_s002
                                 + 0.778559 w_k1_FDX_s002
                                 + 1.00401 w_k1_IBM_s002
                                 + 0.808697 w_k1_INTC_s002
                                 + 0.883475 w_k1_KO_s002
                                 + 0.831874 w_k1_MAT_s002
                                 + 0.806961 w_k1_MSFT_s002
                                 + 1.01811 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.0691 w_k1_ADM_s002
                                 + 1.12936 w_k1_AXP_s002
                                 + 1.14519 w_k1_DIS_s002
                                 + 1.04863 w_k1_FDX_s002
                                 + 1.38383 w_k1_IBM_s002
                                 + 1.02417 w_k1_INTC_s002
                                 + 1.22653 w_k1_KO_s002 + 1.41513 w_k1_MAT_s002
                                 + 0.873677 w_k1_MSFT_s002
                                 + 1.14635 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.985409 w_k1_ADM_s002
                                 + 0.866664 w_k1_AXP_s002
                                 + 0.879566 w_k1_DIS_s002
                                 + 0.92395 w_k1_FDX_s002
                                 + 0.659286 w_k1_IBM_s002
                                 + 0.838463 w_k1_INTC_s002
                                 + 0.754215 w_k1_KO_s002
                                 + 0.657285 w_k1_MAT_s002
                                 + 0.780962 w_k1_MSFT_s002
                                 + 0.907499 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.06889 w_k1_ADM_s002
                                 + 1.11676 w_k1_AXP_s002
                                 + 1.13183 w_k1_DIS_s002 + 0.9428 w_k1_FDX_s002
                                 + 1.00097 w_k1_IBM_s002
                                 + 1.0906 w_k1_INTC_s002 + 1.09198 w_k1_KO_s002
                                 + 1.47692 w_k1_MAT_s002
                                 + 1.05125 w_k1_MSFT_s002
                                 + 1.10366 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.0474 w_k1_ADM_s002
                                 + 0.98575 w_k1_AXP_s002
                                 + 0.959523 w_k1_DIS_s002
                                 + 1.04843 w_k1_FDX_s002
                                 + 1.00806 w_k1_IBM_s002
                                 + 1.21299 w_k1_INTC_s002
                                 + 1.00632 w_k1_KO_s002 + 1.06946 w_k1_MAT_s002
                                 + 0.977185 w_k1_MSFT_s002
                                 + 1.04751 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.789891 w_k1_ADM_s002
                                 + 0.493295 w_k1_AXP_s002
                                 + 0.580519 w_k1_DIS_s002
                                 + 0.724326 w_k1_FDX_s002
                                 + 0.654323 w_k1_IBM_s002
                                 + 0.963428 w_k1_INTC_s002
                                 + 0.783952 w_k1_KO_s002
                                 + 0.573751 w_k1_MAT_s002
                                 + 0.749405 w_k1_MSFT_s002
                                 + 0.839287 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.39695 w_k1_ADM_s002
                                 + 1.03695 w_k1_AXP_s002
                                 + 1.37484 w_k1_DIS_s002
                                 + 1.22283 w_k1_FDX_s002
                                 + 0.911164 w_k1_IBM_s002
                                 + 1.11304 w_k1_INTC_s002
                                 + 1.09399 w_k1_KO_s002 + 0.90836 w_k1_MAT_s002
                                 + 1.19274 w_k1_MSFT_s002
                                 + 1.23633 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.13124 w_k1_ADM_s002
                                 + 1.24424 w_k1_AXP_s002
                                 + 1.72675 w_k1_DIS_s002
                                 + 1.22481 w_k1_FDX_s002
                                 + 1.38408 w_k1_IBM_s002
                                 + 1.09962 w_k1_INTC_s002
                                 + 1.50231 w_k1_KO_s002 + 1.70486 w_k1_MAT_s002
                                 + 1.14441 w_k1_MSFT_s002
                                 + 1.18149 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.05975 w_k1_ADM_s003
                                 + 0.942357 w_k1_AXP_s003 + 1.115 w_k1_DIS_s003
                                 + 0.98837 w_k1_FDX_s003
                                 + 0.915604 w_k1_IBM_s003
                                 + 1.00044 w_k1_INTC_s003
                                 + 0.923187 w_k1_KO_s003
                                 + 0.821266 w_k1_MAT_s003
                                 + 1.02556 w_k1_MSFT_s003
                                 + 1.00766 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.06055 w_k1_ADM_s003
                                 + 0.896339 w_k1_AXP_s003
                                 + 1.22923 w_k1_DIS_s003
                                 + 0.982766 w_k1_FDX_s003
                                 + 1.42039 w_k1_IBM_s003
                                 + 1.15111 w_k1_INTC_s003
                                 + 1.25872 w_k1_KO_s003 + 1.06604 w_k1_MAT_s003
                                 + 1.00307 w_k1_MSFT_s003
                                 + 1.01323 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.48144 w_k1_ADM_s003
                                 + 1.39237 w_k1_AXP_s003 + 1.2191 w_k1_DIS_s003
                                 + 1.3207 w_k1_FDX_s003 + 1.20214 w_k1_IBM_s003
                                 + 1.30013 w_k1_INTC_s003
                                 + 1.05974 w_k1_KO_s003
                                 + 0.820799 w_k1_MAT_s003
                                 + 1.04171 w_k1_MSFT_s003
                                 + 1.23337 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.12157 w_k1_ADM_s003
                                 + 1.11272 w_k1_AXP_s003
                                 + 1.22381 w_k1_DIS_s003
                                 + 1.09994 w_k1_FDX_s003
                                 + 0.709066 w_k1_IBM_s003
                                 + 0.906897 w_k1_INTC_s003
                                 + 0.907638 w_k1_KO_s003
                                 + 0.915506 w_k1_MAT_s003
                                 + 1.05259 w_k1_MSFT_s003
                                 + 0.972512 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.83234 w_k1_ADM_s003
                                 + 0.885216 w_k1_AXP_s003
                                 + 0.881365 w_k1_DIS_s003
                                 + 0.935635 w_k1_FDX_s003
                                 + 0.988639 w_k1_IBM_s003
                                 + 0.870356 w_k1_INTC_s003
                                 + 0.924743 w_k1_KO_s003
                                 + 1.02916 w_k1_MAT_s003
                                 + 0.881698 w_k1_MSFT_s003
                                 + 0.915632 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.87754 w_k1_ADM_s003
                                 + 0.766042 w_k1_AXP_s003
                                 + 0.644758 w_k1_DIS_s003
                                 + 0.807193 w_k1_FDX_s003
                                 + 0.618743 w_k1_IBM_s003
                                 + 0.891649 w_k1_INTC_s003
                                 + 0.610188 w_k1_KO_s003
                                 + 0.525021 w_k1_MAT_s003
                                 + 0.760871 w_k1_MSFT_s003
                                 + 0.800488 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.39415 w_k1_ADM_s003
                                 + 0.838282 w_k1_AXP_s003
                                 + 0.972681 w_k1_DIS_s003
                                 + 1.11251 w_k1_FDX_s003
                                 + 0.678216 w_k1_IBM_s003
                                 + 1.20968 w_k1_INTC_s003
                                 + 0.962467 w_k1_KO_s003
                                 + 0.733156 w_k1_MAT_s003
                                 + 1.10056 w_k1_MSFT_s003
                                 + 1.03044 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.775678 w_k1_ADM_s003
                                 + 0.873666 w_k1_AXP_s003
                                 + 0.692143 w_k1_DIS_s003
                                 + 0.795933 w_k1_FDX_s003
                                 + 1.06863 w_k1_IBM_s003
                                 + 0.813713 w_k1_INTC_s003
                                 + 0.8391 w_k1_KO_s003 + 1.16453 w_k1_MAT_s003
                                 + 0.798171 w_k1_MSFT_s003
                                 + 0.984105 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.09218 w_k1_ADM_s004
                                 + 1.06526 w_k1_AXP_s004
                                 + 1.07773 w_k1_DIS_s004
                                 + 1.06534 w_k1_FDX_s004
                                 + 0.960122 w_k1_IBM_s004
                                 + 1.00911 w_k1_INTC_s004
                                 + 0.984935 w_k1_KO_s004
                                 + 0.902253 w_k1_MAT_s004
                                 + 1.15538 w_k1_MSFT_s004
                                 + 1.03122 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.27748 w_k1_ADM_s004
                                 + 0.90619 w_k1_AXP_s004
                                 + 1.01574 w_k1_DIS_s004
                                 + 0.928163 w_k1_FDX_s004
                                 + 1.53571 w_k1_IBM_s004
                                 + 0.83039 w_k1_INTC_s004
                                 + 1.23922 w_k1_KO_s004
                                 + 0.959799 w_k1_MAT_s004
                                 + 0.925962 w_k1_MSFT_s004
                                 + 1.03089 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.01187 w_k1_ADM_s004
                                 + 0.919306 w_k1_AXP_s004
                                 + 0.832379 w_k1_DIS_s004
                                 + 0.832155 w_k1_FDX_s004
                                 + 1.23923 w_k1_IBM_s004
                                 + 0.934063 w_k1_INTC_s004
                                 + 1.04137 w_k1_KO_s004
                                 + 0.898936 w_k1_MAT_s004
                                 + 0.857558 w_k1_MSFT_s004
                                 + 1.02281 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.886568 w_k1_ADM_s004
                                 + 0.846205 w_k1_AXP_s004
                                 + 0.676446 w_k1_DIS_s004
                                 + 0.697056 w_k1_FDX_s004
                                 + 0.932682 w_k1_IBM_s004
                                 + 0.697702 w_k1_INTC_s004
                                 + 0.823193 w_k1_KO_s004 + 0.6233 w_k1_MAT_s004
                                 + 0.706577 w_k1_MSFT_s004
                                 + 0.831669 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.40874 w_k1_ADM_s004
                                 + 1.09598 w_k1_AXP_s004
                                 + 1.60487 w_k1_DIS_s004
                                 + 1.21255 w_k1_FDX_s004
                                 + 1.16895 w_k1_IBM_s004
                                 + 1.15355 w_k1_INTC_s004
                                 + 1.37628 w_k1_KO_s004
                                 + 0.899379 w_k1_MAT_s004
                                 + 1.31741 w_k1_MSFT_s004
                                 + 1.19255 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.09295 w_k1_ADM_s004
                                 + 1.61019 w_k1_AXP_s004
                                 + 1.41475 w_k1_DIS_s004
                                 + 1.21311 w_k1_FDX_s004
                                 + 1.58584 w_k1_IBM_s004
                                 + 0.950018 w_k1_INTC_s004
                                 + 1.23397 w_k1_KO_s004 + 1.20739 w_k1_MAT_s004
                                 + 0.968018 w_k1_MSFT_s004
                                 + 1.04968 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.849351 w_k1_ADM_s004
                                 + 1.02051 w_k1_AXP_s004
                                 + 1.06916 w_k1_DIS_s004
                                 + 0.97844 w_k1_FDX_s004
                                 + 1.26528 w_k1_IBM_s004
                                 + 0.902594 w_k1_INTC_s004
                                 + 1.07538 w_k1_KO_s004
                                 + 0.769572 w_k1_MAT_s004
                                 + 0.884072 w_k1_MSFT_s004
                                 + 1.00932 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.23607 w_k1_ADM_s004
                                 + 0.918374 w_k1_AXP_s004
                                 + 0.773533 w_k1_DIS_s004
                                 + 0.937013 w_k1_FDX_s004
                                 + 0.693337 w_k1_IBM_s004
                                 + 0.831821 w_k1_INTC_s004
                                 + 0.837879 w_k1_KO_s004
                                 + 0.688702 w_k1_MAT_s004
                                 + 0.963463 w_k1_MSFT_s004
                                 + 0.954962 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.808059 w_k1_ADM_s005
                                 + 0.698174 w_k1_AXP_s005
                                 + 0.752269 w_k1_DIS_s005
                                 + 0.719991 w_k1_FDX_s005
                                 + 0.840458 w_k1_IBM_s005
                                 + 0.694325 w_k1_INTC_s005
                                 + 0.904538 w_k1_KO_s005
                                 + 0.813929 w_k1_MAT_s005
                                 + 0.665016 w_k1_MSFT_s005
                                 + 0.84266 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.717686 w_k1_ADM_s005
                                 + 0.803793 w_k1_AXP_s005
                                 + 1.06461 w_k1_DIS_s005
                                 + 0.909799 w_k1_FDX_s005
                                 + 1.21866 w_k1_IBM_s005
                                 + 0.875842 w_k1_INTC_s005
                                 + 1.01322 w_k1_KO_s005 + 1.18002 w_k1_MAT_s005
                                 + 0.770797 w_k1_MSFT_s005
                                 + 0.807456 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.969417 w_k1_ADM_s005
                                 + 1.4727 w_k1_AXP_s005 + 1.38685 w_k1_DIS_s005
                                 + 1.14996 w_k1_FDX_s005
                                 + 1.47063 w_k1_IBM_s005
                                 + 1.25368 w_k1_INTC_s005 + 1.6098 w_k1_KO_s005
                                 + 1.35422 w_k1_MAT_s005
                                 + 1.20451 w_k1_MSFT_s005 + 1.179 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.937791 w_k1_ADM_s005
                                 + 1.22719 w_k1_AXP_s005
                                 + 0.789067 w_k1_DIS_s005
                                 + 1.05783 w_k1_FDX_s005
                                 + 0.825409 w_k1_IBM_s005
                                 + 1.26651 w_k1_INTC_s005
                                 + 0.83747 w_k1_KO_s005
                                 + 0.930891 w_k1_MAT_s005
                                 + 1.00813 w_k1_MSFT_s005
                                 + 0.921007 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.962993 w_k1_ADM_s005
                                 + 0.976144 w_k1_AXP_s005
                                 + 0.733933 w_k1_DIS_s005
                                 + 0.822255 w_k1_FDX_s005 + 1.002 w_k1_IBM_s005
                                 + 1.00087 w_k1_INTC_s005
                                 + 1.05362 w_k1_KO_s005 + 1.25658 w_k1_MAT_s005
                                 + 0.967863 w_k1_MSFT_s005
                                 + 0.962909 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.974154 w_k1_ADM_s005
                                 + 0.808047 w_k1_AXP_s005
                                 + 0.584198 w_k1_DIS_s005
                                 + 0.846986 w_k1_FDX_s005
                                 + 0.576635 w_k1_IBM_s005
                                 + 0.824417 w_k1_INTC_s005
                                 + 0.602025 w_k1_KO_s005
                                 + 0.542513 w_k1_MAT_s005
                                 + 0.689424 w_k1_MSFT_s005
                                 + 0.77447 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.10732 w_k1_ADM_s005
                                 + 1.19552 w_k1_AXP_s005
                                 + 1.26691 w_k1_DIS_s005
                                 + 1.20349 w_k1_FDX_s005
                                 + 0.947278 w_k1_IBM_s005
                                 + 1.36403 w_k1_INTC_s005 + 1.1483 w_k1_KO_s005
                                 + 0.744508 w_k1_MAT_s005
                                 + 1.29259 w_k1_MSFT_s005
                                 + 1.15885 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.967846 w_k1_ADM_s005
                                 + 0.695604 w_k1_AXP_s005
                                 + 0.871047 w_k1_DIS_s005
                                 + 0.931435 w_k1_FDX_s005
                                 + 0.719034 w_k1_IBM_s005
                                 + 1.14053 w_k1_INTC_s005
                                 + 0.929498 w_k1_KO_s005
                                 + 0.569462 w_k1_MAT_s005
                                 + 1.01469 w_k1_MSFT_s005
                                 + 0.902708 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00193 w_k1_ADM_s006
                                 + 1.26536 w_k1_AXP_s006
                                 + 0.866127 w_k1_DIS_s006
                                 + 0.951981 w_k1_FDX_s006
                                 + 1.08911 w_k1_IBM_s006
                                 + 0.881123 w_k1_INTC_s006
                                 + 0.918524 w_k1_KO_s006
                                 + 1.35436 w_k1_MAT_s006
                                 + 0.886275 w_k1_MSFT_s006
                                 + 0.906427 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.801106 w_k1_ADM_s006
                                 + 0.843343 w_k1_AXP_s006
                                 + 0.668769 w_k1_DIS_s006
                                 + 0.75822 w_k1_FDX_s006
                                 + 1.19346 w_k1_IBM_s006
                                 + 0.841688 w_k1_INTC_s006
                                 + 0.955917 w_k1_KO_s006
                                 + 1.31895 w_k1_MAT_s006
                                 + 0.852687 w_k1_MSFT_s006
                                 + 0.924713 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.08423 w_k1_ADM_s006
                                 + 1.36593 w_k1_AXP_s006
                                 + 1.42153 w_k1_DIS_s006 + 1.349 w_k1_FDX_s006
                                 + 0.985086 w_k1_IBM_s006
                                 + 1.36183 w_k1_INTC_s006
                                 + 1.23062 w_k1_KO_s006 + 0.91893 w_k1_MAT_s006
                                 + 1.47808 w_k1_MSFT_s006 + 1.132 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.07725 w_k1_ADM_s006
                                 + 0.991164 w_k1_AXP_s006
                                 + 0.929404 w_k1_DIS_s006
                                 + 0.985756 w_k1_FDX_s006
                                 + 0.867012 w_k1_IBM_s006
                                 + 0.976429 w_k1_INTC_s006
                                 + 0.836032 w_k1_KO_s006
                                 + 0.886019 w_k1_MAT_s006
                                 + 1.01278 w_k1_MSFT_s006
                                 + 0.960333 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.21822 w_k1_ADM_s006
                                 + 1.31111 w_k1_AXP_s006
                                 + 1.35142 w_k1_DIS_s006 + 1.1518 w_k1_FDX_s006
                                 + 1.35096 w_k1_IBM_s006
                                 + 1.30038 w_k1_INTC_s006
                                 + 1.53482 w_k1_KO_s006 + 1.48743 w_k1_MAT_s006
                                 + 1.41122 w_k1_MSFT_s006
                                 + 1.10915 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.947281 w_k1_ADM_s006
                                 + 1.464 w_k1_AXP_s006 + 1.15639 w_k1_DIS_s006
                                 + 0.990915 w_k1_FDX_s006
                                 + 1.73292 w_k1_IBM_s006
                                 + 1.0516 w_k1_INTC_s006 + 1.28284 w_k1_KO_s006
                                 + 1.61798 w_k1_MAT_s006
                                 + 1.03847 w_k1_MSFT_s006
                                 + 1.04925 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.669246 w_k1_ADM_s006
                                 + 0.822648 w_k1_AXP_s006
                                 + 0.711938 w_k1_DIS_s006
                                 + 0.854365 w_k1_FDX_s006
                                 + 0.674932 w_k1_IBM_s006
                                 + 0.998362 w_k1_INTC_s006
                                 + 0.792443 w_k1_KO_s006
                                 + 1.13754 w_k1_MAT_s006
                                 + 0.977326 w_k1_MSFT_s006
                                 + 0.861313 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.06674 w_k1_ADM_s006
                                 + 1.11629 w_k1_AXP_s006
                                 + 1.11461 w_k1_DIS_s006
                                 + 1.21657 w_k1_FDX_s006
                                 + 0.668352 w_k1_IBM_s006
                                 + 1.46452 w_k1_INTC_s006
                                 + 0.829253 w_k1_KO_s006
                                 + 0.750674 w_k1_MAT_s006
                                 + 1.36817 w_k1_MSFT_s006
                                 + 0.965188 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.03323 w_k1_ADM_s007
                                 + 0.783327 w_k1_AXP_s007
                                 + 1.0026 w_k1_DIS_s007 + 1.05464 w_k1_FDX_s007
                                 + 1.1109 w_k1_IBM_s007
                                 + 1.17094 w_k1_INTC_s007
                                 + 1.13662 w_k1_KO_s007 + 1.1723 w_k1_MAT_s007
                                 + 1.04098 w_k1_MSFT_s007
                                 + 1.05876 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.824409 w_k1_ADM_s007
                                 + 0.70947 w_k1_AXP_s007
                                 + 0.663713 w_k1_DIS_s007
                                 + 0.836717 w_k1_FDX_s007
                                 + 0.771955 w_k1_IBM_s007
                                 + 0.915817 w_k1_INTC_s007
                                 + 0.910378 w_k1_KO_s007
                                 + 0.604368 w_k1_MAT_s007
                                 + 0.798953 w_k1_MSFT_s007
                                 + 0.975696 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.747284 w_k1_ADM_s007
                                 + 0.560247 w_k1_AXP_s007
                                 + 0.565727 w_k1_DIS_s007
                                 + 0.748753 w_k1_FDX_s007
                                 + 0.606317 w_k1_IBM_s007
                                 + 0.763743 w_k1_INTC_s007
                                 + 0.639806 w_k1_KO_s007
                                 + 0.392986 w_k1_MAT_s007
                                 + 0.657961 w_k1_MSFT_s007
                                 + 0.751392 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.10049 w_k1_ADM_s007
                                 + 0.77711 w_k1_AXP_s007
                                 + 0.93615 w_k1_DIS_s007
                                 + 0.954424 w_k1_FDX_s007
                                 + 0.888031 w_k1_IBM_s007
                                 + 0.998446 w_k1_INTC_s007
                                 + 0.945201 w_k1_KO_s007
                                 + 0.663953 w_k1_MAT_s007
                                 + 1.09039 w_k1_MSFT_s007
                                 + 0.953658 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.12201 w_k1_ADM_s007
                                 + 1.11104 w_k1_AXP_s007
                                 + 1.20674 w_k1_DIS_s007
                                 + 1.15808 w_k1_FDX_s007 + 1.6049 w_k1_IBM_s007
                                 + 1.24994 w_k1_INTC_s007 + 1.4254 w_k1_KO_s007
                                 + 1.43834 w_k1_MAT_s007
                                 + 1.03804 w_k1_MSFT_s007
                                 + 1.23828 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.812399 w_k1_ADM_s007
                                 + 0.690223 w_k1_AXP_s007
                                 + 0.649474 w_k1_DIS_s007
                                 + 0.78771 w_k1_FDX_s007
                                 + 1.14533 w_k1_IBM_s007
                                 + 0.80847 w_k1_INTC_s007
                                 + 0.891933 w_k1_KO_s007
                                 + 0.923727 w_k1_MAT_s007
                                 + 0.800954 w_k1_MSFT_s007
                                 + 0.908795 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.21193 w_k1_ADM_s007
                                 + 0.799875 w_k1_AXP_s007
                                 + 0.956267 w_k1_DIS_s007
                                 + 1.09452 w_k1_FDX_s007
                                 + 0.705441 w_k1_IBM_s007
                                 + 1.10146 w_k1_INTC_s007
                                 + 0.905052 w_k1_KO_s007
                                 + 0.62287 w_k1_MAT_s007
                                 + 0.993778 w_k1_MSFT_s007
                                 + 1.0025 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.2568 w_k1_ADM_s007
                                 + 1.2632 w_k1_AXP_s007 + 1.78142 w_k1_DIS_s007
                                 + 1.43195 w_k1_FDX_s007 + 1.2583 w_k1_IBM_s007
                                 + 1.07614 w_k1_INTC_s007
                                 + 1.44137 w_k1_KO_s007 + 1.48536 w_k1_MAT_s007
                                 + 1.30288 w_k1_MSFT_s007
                                 + 1.17828 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.07994 w_k1_ADM_s010
                                 + 1.03379 w_k1_AXP_s010
                                 + 1.39125 w_k1_DIS_s010
                                 + 0.986771 w_k1_FDX_s010
                                 + 1.29882 w_k1_IBM_s010
                                 + 1.26461 w_k1_INTC_s010
                                 + 1.34046 w_k1_KO_s010 + 1.34846 w_k1_MAT_s010
                                 + 1.21384 w_k1_MSFT_s010
                                 + 1.01648 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.39758 w_k1_ADM_s010
                                 + 1.28991 w_k1_AXP_s010
                                 + 1.75113 w_k1_DIS_s010
                                 + 1.36548 w_k1_FDX_s010
                                 + 1.49792 w_k1_IBM_s010
                                 + 1.45235 w_k1_INTC_s010
                                 + 1.55815 w_k1_KO_s010 + 1.60336 w_k1_MAT_s010
                                 + 1.34551 w_k1_MSFT_s010
                                 + 1.20418 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.07674 w_k1_ADM_s010
                                 + 1.14101 w_k1_AXP_s010
                                 + 1.48671 w_k1_DIS_s010 + 1.3054 w_k1_FDX_s010
                                 + 0.983823 w_k1_IBM_s010
                                 + 1.40618 w_k1_INTC_s010
                                 + 1.10156 w_k1_KO_s010 + 1.15696 w_k1_MAT_s010
                                 + 1.4866 w_k1_MSFT_s010 + 1.12939 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.03557 w_k1_ADM_s010
                                 + 0.847611 w_k1_AXP_s010
                                 + 1.01241 w_k1_DIS_s010
                                 + 0.964431 w_k1_FDX_s010
                                 + 1.00199 w_k1_IBM_s010
                                 + 1.17366 w_k1_INTC_s010
                                 + 1.01243 w_k1_KO_s010
                                 + 0.959313 w_k1_MAT_s010
                                 + 1.18712 w_k1_MSFT_s010
                                 + 0.98818 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.38359 w_k1_ADM_s010
                                 + 1.61126 w_k1_AXP_s010
                                 + 1.04431 w_k1_DIS_s010
                                 + 1.26829 w_k1_FDX_s010
                                 + 1.00875 w_k1_IBM_s010
                                 + 1.44633 w_k1_INTC_s010
                                 + 1.09512 w_k1_KO_s010 + 1.08899 w_k1_MAT_s010
                                 + 1.11794 w_k1_MSFT_s010
                                 + 1.01355 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.895225 w_k1_ADM_s010
                                 + 0.892325 w_k1_AXP_s010
                                 + 0.76305 w_k1_DIS_s010
                                 + 0.796695 w_k1_FDX_s010
                                 + 1.14738 w_k1_IBM_s010
                                 + 0.91059 w_k1_INTC_s010
                                 + 0.830702 w_k1_KO_s010 + 1.1144 w_k1_MAT_s010
                                 + 0.736683 w_k1_MSFT_s010
                                 + 0.805773 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.817601 w_k1_ADM_s010
                                 + 0.972828 w_k1_AXP_s010
                                 + 0.837553 w_k1_DIS_s010
                                 + 0.92369 w_k1_FDX_s010
                                 + 0.726905 w_k1_IBM_s010
                                 + 1.13005 w_k1_INTC_s010
                                 + 0.935293 w_k1_KO_s010
                                 + 0.990154 w_k1_MAT_s010
                                 + 0.982162 w_k1_MSFT_s010
                                 + 0.936562 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.795161 w_k1_ADM_s010
                                 + 1.01475 w_k1_AXP_s010 + 1.0707 w_k1_DIS_s010
                                 + 1.01262 w_k1_FDX_s010
                                 + 1.46557 w_k1_IBM_s010
                                 + 1.21425 w_k1_INTC_s010
                                 + 1.28676 w_k1_KO_s010 + 1.48274 w_k1_MAT_s010
                                 + 1.02014 w_k1_MSFT_s010
                                 + 1.06636 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.870427 w_k1_ADM_s011
                                 + 0.877716 w_k1_AXP_s011
                                 + 0.786654 w_k1_DIS_s011
                                 + 0.826779 w_k1_FDX_s011
                                 + 1.2272 w_k1_IBM_s011
                                 + 1.03061 w_k1_INTC_s011
                                 + 0.934292 w_k1_KO_s011
                                 + 1.31159 w_k1_MAT_s011
                                 + 0.918061 w_k1_MSFT_s011
                                 + 0.877704 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.905085 w_k1_ADM_s011
                                 + 1.48424 w_k1_AXP_s011
                                 + 1.25762 w_k1_DIS_s011
                                 + 1.20407 w_k1_FDX_s011
                                 + 1.06913 w_k1_IBM_s011
                                 + 1.1557 w_k1_INTC_s011 + 1.12582 w_k1_KO_s011
                                 + 1.12492 w_k1_MAT_s011
                                 + 1.08542 w_k1_MSFT_s011
                                 + 1.10585 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.93977 w_k1_ADM_s011
                                 + 1.10295 w_k1_AXP_s011
                                 + 0.833029 w_k1_DIS_s011
                                 + 0.996843 w_k1_FDX_s011
                                 + 0.713315 w_k1_IBM_s011
                                 + 1.18093 w_k1_INTC_s011
                                 + 0.768094 w_k1_KO_s011
                                 + 0.79377 w_k1_MAT_s011
                                 + 0.933502 w_k1_MSFT_s011
                                 + 0.926932 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.780386 w_k1_ADM_s011
                                 + 0.799101 w_k1_AXP_s011
                                 + 0.672125 w_k1_DIS_s011
                                 + 0.789101 w_k1_FDX_s011
                                 + 0.878476 w_k1_IBM_s011
                                 + 0.894786 w_k1_INTC_s011
                                 + 0.782168 w_k1_KO_s011
                                 + 0.637703 w_k1_MAT_s011
                                 + 0.838247 w_k1_MSFT_s011
                                 + 0.773372 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.774857 w_k1_ADM_s011
                                 + 1.11585 w_k1_AXP_s011 + 1.0807 w_k1_DIS_s011
                                 + 0.992415 w_k1_FDX_s011
                                 + 1.14067 w_k1_IBM_s011
                                 + 1.05245 w_k1_INTC_s011
                                 + 1.09245 w_k1_KO_s011 + 0.97481 w_k1_MAT_s011
                                 + 0.978896 w_k1_MSFT_s011
                                 + 0.926279 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.0509 w_k1_ADM_s011
                                 + 1.31821 w_k1_AXP_s011 + 1.4245 w_k1_DIS_s011
                                 + 1.34528 w_k1_FDX_s011
                                 + 0.876379 w_k1_IBM_s011
                                 + 1.33983 w_k1_INTC_s011
                                 + 1.09368 w_k1_KO_s011
                                 + 0.892794 w_k1_MAT_s011
                                 + 1.29868 w_k1_MSFT_s011
                                 + 1.08935 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.02195 w_k1_ADM_s011
                                 + 1.04906 w_k1_AXP_s011
                                 + 0.931284 w_k1_DIS_s011
                                 + 0.913511 w_k1_FDX_s011
                                 + 1.33287 w_k1_IBM_s011
                                 + 1.06284 w_k1_INTC_s011
                                 + 1.03912 w_k1_KO_s011 + 1.23957 w_k1_MAT_s011
                                 + 0.982663 w_k1_MSFT_s011
                                 + 0.950137 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.37289 w_k1_ADM_s011
                                 + 1.55923 w_k1_AXP_s011
                                 + 1.47044 w_k1_DIS_s011
                                 + 1.31083 w_k1_FDX_s011
                                 + 1.55724 w_k1_IBM_s011
                                 + 1.12747 w_k1_INTC_s011
                                 + 1.35116 w_k1_KO_s011 + 1.66331 w_k1_MAT_s011
                                 + 1.2808 w_k1_MSFT_s011 + 1.25454 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.826508 w_k1_ADM_s012
                                 + 0.887078 w_k1_AXP_s012
                                 + 0.909984 w_k1_DIS_s012
                                 + 0.850241 w_k1_FDX_s012
                                 + 1.09438 w_k1_IBM_s012
                                 + 0.942366 w_k1_INTC_s012
                                 + 1.07663 w_k1_KO_s012 + 1.23144 w_k1_MAT_s012
                                 + 0.933812 w_k1_MSFT_s012
                                 + 1.08992 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.21486 w_k1_ADM_s012
                                 + 1.15493 w_k1_AXP_s012
                                 + 1.25968 w_k1_DIS_s012
                                 + 1.19692 w_k1_FDX_s012
                                 + 0.894198 w_k1_IBM_s012
                                 + 1.26244 w_k1_INTC_s012
                                 + 1.06484 w_k1_KO_s012
                                 + 0.799708 w_k1_MAT_s012
                                 + 1.30251 w_k1_MSFT_s012
                                 + 0.97326 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.940505 w_k1_ADM_s012
                                 + 1.03886 w_k1_AXP_s012
                                 + 1.25988 w_k1_DIS_s012
                                 + 0.994206 w_k1_FDX_s012
                                 + 1.6053 w_k1_IBM_s012 + 1.103 w_k1_INTC_s012
                                 + 1.37813 w_k1_KO_s012 + 1.68577 w_k1_MAT_s012
                                 + 1.21448 w_k1_MSFT_s012
                                 + 1.11451 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.11754 w_k1_ADM_s012
                                 + 0.916577 w_k1_AXP_s012
                                 + 1.00483 w_k1_DIS_s012
                                 + 1.12967 w_k1_FDX_s012
                                 + 0.778557 w_k1_IBM_s012
                                 + 1.01921 w_k1_INTC_s012
                                 + 0.961258 w_k1_KO_s012
                                 + 0.710522 w_k1_MAT_s012
                                 + 1.17856 w_k1_MSFT_s012 + 1.0175 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.712564 w_k1_ADM_s012
                                 + 0.856013 w_k1_AXP_s012
                                 + 0.749853 w_k1_DIS_s012
                                 + 0.845934 w_k1_FDX_s012
                                 + 0.77744 w_k1_IBM_s012
                                 + 0.989958 w_k1_INTC_s012
                                 + 0.869401 w_k1_KO_s012
                                 + 0.751032 w_k1_MAT_s012
                                 + 1.00876 w_k1_MSFT_s012
                                 + 0.963354 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.839846 w_k1_ADM_s012
                                 + 0.843658 w_k1_AXP_s012
                                 + 0.675644 w_k1_DIS_s012
                                 + 0.828691 w_k1_FDX_s012
                                 + 0.68986 w_k1_IBM_s012
                                 + 0.862544 w_k1_INTC_s012
                                 + 0.645066 w_k1_KO_s012
                                 + 0.605643 w_k1_MAT_s012
                                 + 0.788891 w_k1_MSFT_s012
                                 + 0.806172 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.929916 w_k1_ADM_s012
                                 + 0.890669 w_k1_AXP_s012
                                 + 0.870663 w_k1_DIS_s012
                                 + 0.962927 w_k1_FDX_s012
                                 + 0.956537 w_k1_IBM_s012
                                 + 0.972982 w_k1_INTC_s012
                                 + 1.01992 w_k1_KO_s012
                                 + 0.967689 w_k1_MAT_s012
                                 + 0.985066 w_k1_MSFT_s012
                                 + 0.894381 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.07617 w_k1_ADM_s012
                                 + 1.48798 w_k1_AXP_s012
                                 + 1.15061 w_k1_DIS_s012
                                 + 1.17898 w_k1_FDX_s012
                                 + 1.19002 w_k1_IBM_s012
                                 + 1.12325 w_k1_INTC_s012
                                 + 1.15613 w_k1_KO_s012 + 1.64197 w_k1_MAT_s012
                                 + 1.18773 w_k1_MSFT_s012
                                 + 1.17648 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.19439 w_k1_ADM_s013
                                 + 1.59274 w_k1_AXP_s013
                                 + 1.70846 w_k1_DIS_s013
                                 + 1.32275 w_k1_FDX_s013
                                 + 1.30296 w_k1_IBM_s013
                                 + 1.22664 w_k1_INTC_s013
                                 + 1.25325 w_k1_KO_s013 + 1.51881 w_k1_MAT_s013
                                 + 1.29242 w_k1_MSFT_s013
                                 + 1.18347 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.772844 w_k1_ADM_s013
                                 + 0.696739 w_k1_AXP_s013
                                 + 0.643431 w_k1_DIS_s013
                                 + 0.839442 w_k1_FDX_s013
                                 + 0.664159 w_k1_IBM_s013
                                 + 0.773521 w_k1_INTC_s013
                                 + 0.642658 w_k1_KO_s013
                                 + 0.544478 w_k1_MAT_s013
                                 + 0.856598 w_k1_MSFT_s013
                                 + 0.848828 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.891696 w_k1_ADM_s013
                                 + 1.0711 w_k1_AXP_s013 + 1.36259 w_k1_DIS_s013
                                 + 1.27323 w_k1_FDX_s013 + 1.0671 w_k1_IBM_s013
                                 + 1.20016 w_k1_INTC_s013
                                 + 1.10117 w_k1_KO_s013 + 1.137 w_k1_MAT_s013
                                 + 1.27142 w_k1_MSFT_s013
                                 + 1.05224 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.954781 w_k1_ADM_s013
                                 + 0.904314 w_k1_AXP_s013
                                 + 0.928063 w_k1_DIS_s013
                                 + 1.01783 w_k1_FDX_s013
                                 + 0.856945 w_k1_IBM_s013
                                 + 0.801204 w_k1_INTC_s013
                                 + 0.840498 w_k1_KO_s013
                                 + 1.28476 w_k1_MAT_s013
                                 + 0.990957 w_k1_MSFT_s013
                                 + 1.01722 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.19723 w_k1_ADM_s013
                                 + 1.33597 w_k1_AXP_s013
                                 + 1.15662 w_k1_DIS_s013
                                 + 1.33493 w_k1_FDX_s013
                                 + 0.858129 w_k1_IBM_s013
                                 + 1.07321 w_k1_INTC_s013
                                 + 1.01392 w_k1_KO_s013 + 1.18082 w_k1_MAT_s013
                                 + 1.21111 w_k1_MSFT_s013
                                 + 0.95007 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.715764 w_k1_ADM_s013
                                 + 0.73604 w_k1_AXP_s013
                                 + 0.963465 w_k1_DIS_s013
                                 + 0.834452 w_k1_FDX_s013
                                 + 1.05509 w_k1_IBM_s013
                                 + 0.959648 w_k1_INTC_s013
                                 + 0.974944 w_k1_KO_s013
                                 + 1.33585 w_k1_MAT_s013
                                 + 0.989309 w_k1_MSFT_s013
                                 + 0.937942 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.937324 w_k1_ADM_s013
                                 + 1.25503 w_k1_AXP_s013
                                 + 0.717711 w_k1_DIS_s013
                                 + 0.909929 w_k1_FDX_s013
                                 + 1.09572 w_k1_IBM_s013
                                 + 0.830399 w_k1_INTC_s013
                                 + 0.883227 w_k1_KO_s013 + 1.8681 w_k1_MAT_s013
                                 + 0.849893 w_k1_MSFT_s013
                                 + 0.931965 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.817061 w_k1_ADM_s013
                                 + 0.944463 w_k1_AXP_s013
                                 + 0.793865 w_k1_DIS_s013
                                 + 0.788405 w_k1_FDX_s013
                                 + 1.27389 w_k1_IBM_s013
                                 + 0.928753 w_k1_INTC_s013
                                 + 0.888363 w_k1_KO_s013
                                 + 1.16735 w_k1_MAT_s013
                                 + 0.918103 w_k1_MSFT_s013
                                 + 0.838441 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.863714 w_k1_ADM_s014
                                 + 0.906595 w_k1_AXP_s014
                                 + 0.770217 w_k1_DIS_s014
                                 + 0.952172 w_k1_FDX_s014
                                 + 0.790484 w_k1_IBM_s014
                                 + 0.929493 w_k1_INTC_s014
                                 + 0.856259 w_k1_KO_s014
                                 + 1.21104 w_k1_MAT_s014
                                 + 0.885577 w_k1_MSFT_s014
                                 + 0.918858 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.26731 w_k1_ADM_s014
                                 + 1.4889 w_k1_AXP_s014 + 1.64587 w_k1_DIS_s014
                                 + 1.39452 w_k1_FDX_s014 + 1.4404 w_k1_IBM_s014
                                 + 1.54749 w_k1_INTC_s014
                                 + 1.67941 w_k1_KO_s014 + 1.43889 w_k1_MAT_s014
                                 + 1.55524 w_k1_MSFT_s014
                                 + 1.20239 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.01277 w_k1_ADM_s014
                                 + 0.857008 w_k1_AXP_s014
                                 + 1.28704 w_k1_DIS_s014
                                 + 0.952364 w_k1_FDX_s014
                                 + 1.37097 w_k1_IBM_s014
                                 + 0.944852 w_k1_INTC_s014
                                 + 1.29125 w_k1_KO_s014 + 1.23008 w_k1_MAT_s014
                                 + 1.12515 w_k1_MSFT_s014
                                 + 1.03467 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.13893 w_k1_ADM_s014
                                 + 0.915599 w_k1_AXP_s014
                                 + 1.28095 w_k1_DIS_s014
                                 + 1.18378 w_k1_FDX_s014
                                 + 0.911898 w_k1_IBM_s014
                                 + 1.29139 w_k1_INTC_s014
                                 + 1.25641 w_k1_KO_s014 + 1.2559 w_k1_MAT_s014
                                 + 1.39815 w_k1_MSFT_s014 + 1.1331 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.691369 w_k1_ADM_s014
                                 + 1.03484 w_k1_AXP_s014
                                 + 0.931639 w_k1_DIS_s014
                                 + 0.85582 w_k1_FDX_s014
                                 + 1.22971 w_k1_IBM_s014
                                 + 0.981659 w_k1_INTC_s014
                                 + 1.17454 w_k1_KO_s014 + 1.4568 w_k1_MAT_s014
                                 + 0.945892 w_k1_MSFT_s014
                                 + 0.993396 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.91544 w_k1_ADM_s014
                                 + 0.683988 w_k1_AXP_s014
                                 + 0.669934 w_k1_DIS_s014
                                 + 0.749381 w_k1_FDX_s014
                                 + 1.01657 w_k1_IBM_s014
                                 + 0.824972 w_k1_INTC_s014
                                 + 0.866841 w_k1_KO_s014
                                 + 0.963412 w_k1_MAT_s014
                                 + 0.860944 w_k1_MSFT_s014
                                 + 0.782022 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.00073 w_k1_ADM_s014
                                 + 0.872369 w_k1_AXP_s014
                                 + 1.03742 w_k1_DIS_s014
                                 + 0.942433 w_k1_FDX_s014
                                 + 0.854315 w_k1_IBM_s014
                                 + 0.853226 w_k1_INTC_s014
                                 + 0.975553 w_k1_KO_s014
                                 + 0.836906 w_k1_MAT_s014
                                 + 1.03244 w_k1_MSFT_s014
                                 + 0.937307 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.75692 w_k1_ADM_s014
                                 + 0.583844 w_k1_AXP_s014
                                 + 0.616732 w_k1_DIS_s014
                                 + 0.775399 w_k1_FDX_s014
                                 + 0.752072 w_k1_IBM_s014
                                 + 0.807694 w_k1_INTC_s014
                                 + 0.788364 w_k1_KO_s014
                                 + 0.624835 w_k1_MAT_s014
                                 + 0.812057 w_k1_MSFT_s014
                                 + 0.841442 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.981611 w_k1_ADM_s015
                                 + 1.0289 w_k1_AXP_s015 + 1.82236 w_k1_DIS_s015
                                 + 1.29161 w_k1_FDX_s015
                                 + 1.47234 w_k1_IBM_s015
                                 + 1.10995 w_k1_INTC_s015 + 1.4325 w_k1_KO_s015
                                 + 1.22666 w_k1_MAT_s015
                                 + 1.18229 w_k1_MSFT_s015
                                 + 1.24461 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.07975 w_k1_ADM_s015
                                 + 0.994474 w_k1_AXP_s015
                                 + 1.14119 w_k1_DIS_s015
                                 + 1.11896 w_k1_FDX_s015
                                 + 1.15293 w_k1_IBM_s015
                                 + 0.92803 w_k1_INTC_s015 + 1.0847 w_k1_KO_s015
                                 + 1.62548 w_k1_MAT_s015
                                 + 0.958841 w_k1_MSFT_s015
                                 + 1.08227 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.957766 w_k1_ADM_s015
                                 + 0.833752 w_k1_AXP_s015
                                 + 0.621788 w_k1_DIS_s015
                                 + 0.788036 w_k1_FDX_s015
                                 + 0.760532 w_k1_IBM_s015
                                 + 0.776692 w_k1_INTC_s015
                                 + 0.787933 w_k1_KO_s015
                                 + 0.915066 w_k1_MAT_s015
                                 + 0.786675 w_k1_MSFT_s015
                                 + 0.879433 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.820795 w_k1_ADM_s015
                                 + 0.675177 w_k1_AXP_s015
                                 + 0.842986 w_k1_DIS_s015
                                 + 0.892625 w_k1_FDX_s015
                                 + 0.742863 w_k1_IBM_s015
                                 + 0.751073 w_k1_INTC_s015
                                 + 0.832022 w_k1_KO_s015
                                 + 0.700829 w_k1_MAT_s015
                                 + 0.837547 w_k1_MSFT_s015
                                 + 0.937179 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.987887 w_k1_ADM_s015
                                 + 0.92446 w_k1_AXP_s015
                                 + 0.994837 w_k1_DIS_s015
                                 + 0.923444 w_k1_FDX_s015
                                 + 1.33968 w_k1_IBM_s015
                                 + 0.905463 w_k1_INTC_s015
                                 + 1.09706 w_k1_KO_s015 + 1.53109 w_k1_MAT_s015
                                 + 0.840986 w_k1_MSFT_s015
                                 + 0.90906 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.00649 w_k1_ADM_s015
                                 + 1.28778 w_k1_AXP_s015
                                 + 1.37401 w_k1_DIS_s015
                                 + 1.07055 w_k1_FDX_s015
                                 + 1.63026 w_k1_IBM_s015
                                 + 0.991379 w_k1_INTC_s015
                                 + 1.49283 w_k1_KO_s015 + 2.47842 w_k1_MAT_s015
                                 + 1.10092 w_k1_MSFT_s015
                                 + 1.27142 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.31246 w_k1_ADM_s015
                                 + 1.49092 w_k1_AXP_s015
                                 + 0.944619 w_k1_DIS_s015
                                 + 1.11974 w_k1_FDX_s015
                                 + 1.06848 w_k1_IBM_s015
                                 + 0.934624 w_k1_INTC_s015
                                 + 1.02096 w_k1_KO_s015 + 1.43675 w_k1_MAT_s015
                                 + 0.937111 w_k1_MSFT_s015
                                 + 1.15804 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.11549 w_k1_ADM_s015
                                 + 1.12784 w_k1_AXP_s015 + 1.1392 w_k1_DIS_s015
                                 + 1.25425 w_k1_FDX_s015
                                 + 0.841408 w_k1_IBM_s015
                                 + 1.03086 w_k1_INTC_s015
                                 + 1.05728 w_k1_KO_s015 + 1.12067 w_k1_MAT_s015
                                 + 0.92941 w_k1_MSFT_s015 + 1.0879 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.703096 w_k1_ADM_s016
                                 + 0.499388 w_k1_AXP_s016
                                 + 0.629327 w_k1_DIS_s016
                                 + 0.832343 w_k1_FDX_s016
                                 + 0.489245 w_k1_IBM_s016
                                 + 0.815433 w_k1_INTC_s016
                                 + 0.631498 w_k1_KO_s016
                                 + 0.481925 w_k1_MAT_s016
                                 + 0.857417 w_k1_MSFT_s016
                                 + 0.831452 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.875688 w_k1_ADM_s016
                                 + 0.763393 w_k1_AXP_s016
                                 + 0.849972 w_k1_DIS_s016
                                 + 0.962666 w_k1_FDX_s016
                                 + 0.599112 w_k1_IBM_s016
                                 + 0.836695 w_k1_INTC_s016
                                 + 0.776082 w_k1_KO_s016
                                 + 0.554022 w_k1_MAT_s016
                                 + 0.979852 w_k1_MSFT_s016
                                 + 0.910585 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.02882 w_k1_ADM_s016
                                 + 1.12424 w_k1_AXP_s016
                                 + 0.894474 w_k1_DIS_s016
                                 + 1.10414 w_k1_FDX_s016
                                 + 0.844029 w_k1_IBM_s016
                                 + 0.908189 w_k1_INTC_s016
                                 + 0.876403 w_k1_KO_s016
                                 + 0.895976 w_k1_MAT_s016
                                 + 0.912614 w_k1_MSFT_s016
                                 + 0.964909 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.01522 w_k1_ADM_s016
                                 + 1.15596 w_k1_AXP_s016
                                 + 1.15394 w_k1_DIS_s016
                                 + 0.956815 w_k1_FDX_s016
                                 + 1.76528 w_k1_IBM_s016
                                 + 1.04453 w_k1_INTC_s016
                                 + 1.41185 w_k1_KO_s016 + 1.89323 w_k1_MAT_s016
                                 + 1.04498 w_k1_MSFT_s016
                                 + 1.25203 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.34497 w_k1_ADM_s016
                                 + 1.01874 w_k1_AXP_s016
                                 + 1.44243 w_k1_DIS_s016
                                 + 1.36997 w_k1_FDX_s016
                                 + 0.903377 w_k1_IBM_s016
                                 + 1.34856 w_k1_INTC_s016
                                 + 1.02128 w_k1_KO_s016
                                 + 0.912826 w_k1_MAT_s016
                                 + 1.49983 w_k1_MSFT_s016
                                 + 1.16553 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.747821 w_k1_ADM_s016
                                 + 0.634433 w_k1_AXP_s016
                                 + 0.602158 w_k1_DIS_s016
                                 + 0.705333 w_k1_FDX_s016
                                 + 0.841137 w_k1_IBM_s016
                                 + 0.778037 w_k1_INTC_s016
                                 + 0.774412 w_k1_KO_s016
                                 + 0.826642 w_k1_MAT_s016
                                 + 0.792559 w_k1_MSFT_s016
                                 + 0.890141 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.961494 w_k1_ADM_s016
                                 + 1.10314 w_k1_AXP_s016
                                 + 0.910798 w_k1_DIS_s016
                                 + 0.948814 w_k1_FDX_s016
                                 + 1.12269 w_k1_IBM_s016
                                 + 1.02153 w_k1_INTC_s016
                                 + 0.951135 w_k1_KO_s016
                                 + 1.12072 w_k1_MAT_s016
                                 + 0.969187 w_k1_MSFT_s016
                                 + 1.03562 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.97447 w_k1_ADM_s016
                                 + 1.41742 w_k1_AXP_s016
                                 + 1.33591 w_k1_DIS_s016
                                 + 1.26738 w_k1_FDX_s016
                                 + 1.26536 w_k1_IBM_s016
                                 + 1.13696 w_k1_INTC_s016
                                 + 1.19891 w_k1_KO_s016 + 1.70005 w_k1_MAT_s016
                                 + 1.21992 w_k1_MSFT_s016
                                 + 1.22596 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.28748 w_k1_ADM_s017
                                 + 1.2101 w_k1_AXP_s017 + 1.05801 w_k1_DIS_s017
                                 + 1.24961 w_k1_FDX_s017
                                 + 0.936207 w_k1_IBM_s017
                                 + 1.33468 w_k1_INTC_s017
                                 + 0.909212 w_k1_KO_s017
                                 + 0.801837 w_k1_MAT_s017
                                 + 1.15843 w_k1_MSFT_s017
                                 + 1.22078 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.21274 w_k1_ADM_s017
                                 + 1.09749 w_k1_AXP_s017
                                 + 1.37926 w_k1_DIS_s017
                                 + 1.16326 w_k1_FDX_s017
                                 + 1.41377 w_k1_IBM_s017
                                 + 1.27379 w_k1_INTC_s017
                                 + 1.40116 w_k1_KO_s017 + 1.36793 w_k1_MAT_s017
                                 + 1.1111 w_k1_MSFT_s017 + 1.32419 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.752274 w_k1_ADM_s017
                                 + 0.839808 w_k1_AXP_s017
                                 + 0.956725 w_k1_DIS_s017
                                 + 0.783086 w_k1_FDX_s017
                                 + 1.30529 w_k1_IBM_s017
                                 + 0.988558 w_k1_INTC_s017
                                 + 1.11719 w_k1_KO_s017
                                 + 0.984973 w_k1_MAT_s017
                                 + 0.86138 w_k1_MSFT_s017
                                 + 0.964942 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.06308 w_k1_ADM_s017
                                 + 0.862193 w_k1_AXP_s017
                                 + 1.33929 w_k1_DIS_s017
                                 + 1.05965 w_k1_FDX_s017
                                 + 0.895995 w_k1_IBM_s017
                                 + 1.29762 w_k1_INTC_s017
                                 + 1.12628 w_k1_KO_s017
                                 + 0.493451 w_k1_MAT_s017
                                 + 1.23746 w_k1_MSFT_s017
                                 + 0.997024 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.965153 w_k1_ADM_s017
                                 + 0.84768 w_k1_AXP_s017
                                 + 0.989179 w_k1_DIS_s017
                                 + 0.85075 w_k1_FDX_s017
                                 + 0.953491 w_k1_IBM_s017
                                 + 0.923524 w_k1_INTC_s017
                                 + 0.94299 w_k1_KO_s017
                                 + 0.728357 w_k1_MAT_s017
                                 + 0.870819 w_k1_MSFT_s017
                                 + 0.991457 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.12673 w_k1_ADM_s017
                                 + 1.12704 w_k1_AXP_s017
                                 + 1.17583 w_k1_DIS_s017
                                 + 1.07611 w_k1_FDX_s017 + 1.2068 w_k1_IBM_s017
                                 + 1.06274 w_k1_INTC_s017
                                 + 1.10771 w_k1_KO_s017
                                 + 0.842721 w_k1_MAT_s017
                                 + 0.945 w_k1_MSFT_s017 + 1.07557 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.955585 w_k1_ADM_s017
                                 + 0.67132 w_k1_AXP_s017
                                 + 0.737202 w_k1_DIS_s017
                                 + 0.762461 w_k1_FDX_s017
                                 + 0.658086 w_k1_IBM_s017
                                 + 0.669339 w_k1_INTC_s017
                                 + 0.727049 w_k1_KO_s017
                                 + 0.592438 w_k1_MAT_s017
                                 + 0.750357 w_k1_MSFT_s017
                                 + 0.843832 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.30682 w_k1_ADM_s017
                                 + 1.78753 w_k1_AXP_s017
                                 + 1.79548 w_k1_DIS_s017
                                 + 1.25745 w_k1_FDX_s017
                                 + 1.76611 w_k1_IBM_s017
                                 + 1.41298 w_k1_INTC_s017
                                 + 1.54995 w_k1_KO_s017 + 1.22183 w_k1_MAT_s017
                                 + 1.38609 w_k1_MSFT_s017
                                 + 1.53655 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.23241 w_k1_ADM_s100
                                 + 1.46718 w_k1_AXP_s100
                                 + 1.50412 w_k1_DIS_s100
                                 + 1.23578 w_k1_FDX_s100
                                 + 1.31248 w_k1_IBM_s100
                                 + 1.19018 w_k1_INTC_s100 + 1.1646 w_k1_KO_s100
                                 + 1.29857 w_k1_MAT_s100
                                 + 1.20352 w_k1_MSFT_s100
                                 + 1.18004 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.25624 w_k1_ADM_s100
                                 + 1.20843 w_k1_AXP_s100
                                 + 0.903713 w_k1_DIS_s100
                                 + 1.05824 w_k1_FDX_s100
                                 + 0.754455 w_k1_IBM_s100
                                 + 1.12345 w_k1_INTC_s100
                                 + 0.837116 w_k1_KO_s100
                                 + 1.06313 w_k1_MAT_s100
                                 + 1.08333 w_k1_MSFT_s100
                                 + 0.992577 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.893624 w_k1_ADM_s100
                                 + 0.879907 w_k1_AXP_s100
                                 + 0.912587 w_k1_DIS_s100
                                 + 0.894284 w_k1_FDX_s100
                                 + 0.903603 w_k1_IBM_s100
                                 + 0.8693 w_k1_INTC_s100
                                 + 0.947603 w_k1_KO_s100 + 1.2345 w_k1_MAT_s100
                                 + 0.835161 w_k1_MSFT_s100
                                 + 0.946291 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.09589 w_k1_ADM_s100
                                 + 1.07059 w_k1_AXP_s100
                                 + 0.707504 w_k1_DIS_s100
                                 + 0.873749 w_k1_FDX_s100
                                 + 1.15471 w_k1_IBM_s100
                                 + 0.820659 w_k1_INTC_s100
                                 + 0.955826 w_k1_KO_s100
                                 + 1.59184 w_k1_MAT_s100
                                 + 0.75084 w_k1_MSFT_s100
                                 + 0.890794 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.667494 w_k1_ADM_s100
                                 + 0.538194 w_k1_AXP_s100
                                 + 0.639142 w_k1_DIS_s100
                                 + 0.700792 w_k1_FDX_s100
                                 + 0.671424 w_k1_IBM_s100
                                 + 0.753471 w_k1_INTC_s100
                                 + 0.713554 w_k1_KO_s100
                                 + 0.811896 w_k1_MAT_s100
                                 + 0.660907 w_k1_MSFT_s100
                                 + 0.692147 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.993781 w_k1_ADM_s100
                                 + 0.713201 w_k1_AXP_s100
                                 + 0.77081 w_k1_DIS_s100
                                 + 0.942442 w_k1_FDX_s100
                                 + 0.597342 w_k1_IBM_s100
                                 + 0.843952 w_k1_INTC_s100
                                 + 0.74136 w_k1_KO_s100
                                 + 0.625581 w_k1_MAT_s100
                                 + 0.865957 w_k1_MSFT_s100
                                 + 0.871863 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.07103 w_k1_ADM_s100
                                 + 0.881105 w_k1_AXP_s100
                                 + 1.09908 w_k1_DIS_s100
                                 + 0.947194 w_k1_FDX_s100
                                 + 1.23129 w_k1_IBM_s100
                                 + 1.02072 w_k1_INTC_s100
                                 + 1.02672 w_k1_KO_s100
                                 + 0.823683 w_k1_MAT_s100
                                 + 0.966118 w_k1_MSFT_s100
                                 + 0.937328 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.982766 w_k1_ADM_s100
                                 + 0.787857 w_k1_AXP_s100
                                 + 1.07095 w_k1_DIS_s100
                                 + 1.08522 w_k1_FDX_s100
                                 + 0.527388 w_k1_IBM_s100
                                 + 1.06105 w_k1_INTC_s100
                                 + 0.803217 w_k1_KO_s100
                                 + 0.623328 w_k1_MAT_s100
                                 + 0.94232 w_k1_MSFT_s100
                                 + 0.86855 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.31405 w_k1_ADM_s101
                                 + 1.32759 w_k1_AXP_s101
                                 + 1.28826 w_k1_DIS_s101
                                 + 1.26556 w_k1_FDX_s101
                                 + 0.944739 w_k1_IBM_s101
                                 + 1.27682 w_k1_INTC_s101
                                 + 1.05858 w_k1_KO_s101
                                 + 0.997202 w_k1_MAT_s101
                                 + 1.12519 w_k1_MSFT_s101 + 1.1344 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.14635 w_k1_ADM_s101
                                 + 0.917182 w_k1_AXP_s101
                                 + 0.8349 w_k1_DIS_s101 + 0.7936 w_k1_FDX_s101
                                 + 1.08875 w_k1_IBM_s101
                                 + 0.943176 w_k1_INTC_s101
                                 + 1.01501 w_k1_KO_s101
                                 + 0.808158 w_k1_MAT_s101
                                 + 0.861726 w_k1_MSFT_s101
                                 + 1.0193 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.7736 w_k1_ADM_s101
                                 + 1.48346 w_k1_AXP_s101
                                 + 1.94291 w_k1_DIS_s101
                                 + 1.39808 w_k1_FDX_s101
                                 + 1.03426 w_k1_IBM_s101
                                 + 1.44246 w_k1_INTC_s101
                                 + 1.38289 w_k1_KO_s101 + 1.29097 w_k1_MAT_s101
                                 + 1.50603 w_k1_MSFT_s101
                                 + 1.23095 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.14217 w_k1_ADM_s101
                                 + 1.20448 w_k1_AXP_s101 + 1.0089 w_k1_DIS_s101
                                 + 0.927693 w_k1_FDX_s101
                                 + 1.50422 w_k1_IBM_s101
                                 + 1.01934 w_k1_INTC_s101
                                 + 1.23921 w_k1_KO_s101 + 1.58948 w_k1_MAT_s101
                                 + 1.04306 w_k1_MSFT_s101
                                 + 1.17651 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.867044 w_k1_ADM_s101
                                 + 0.655652 w_k1_AXP_s101
                                 + 0.601975 w_k1_DIS_s101
                                 + 0.708436 w_k1_FDX_s101
                                 + 0.815195 w_k1_IBM_s101
                                 + 0.905989 w_k1_INTC_s101
                                 + 0.789299 w_k1_KO_s101
                                 + 0.661425 w_k1_MAT_s101
                                 + 0.702151 w_k1_MSFT_s101
                                 + 0.770391 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.03372 w_k1_ADM_s101
                                 + 0.618269 w_k1_AXP_s101
                                 + 0.969847 w_k1_DIS_s101
                                 + 0.927651 w_k1_FDX_s101
                                 + 0.679522 w_k1_IBM_s101
                                 + 0.970237 w_k1_INTC_s101
                                 + 0.904053 w_k1_KO_s101
                                 + 0.579627 w_k1_MAT_s101
                                 + 1.05142 w_k1_MSFT_s101
                                 + 0.861529 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.23634 w_k1_ADM_s101
                                 + 0.988557 w_k1_AXP_s101
                                 + 0.774519 w_k1_DIS_s101
                                 + 0.965713 w_k1_FDX_s101
                                 + 0.669677 w_k1_IBM_s101
                                 + 1.06479 w_k1_INTC_s101
                                 + 0.748879 w_k1_KO_s101
                                 + 1.06479 w_k1_MAT_s101
                                 + 0.918258 w_k1_MSFT_s101
                                 + 0.952672 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.07067 w_k1_ADM_s101
                                 + 1.04579 w_k1_AXP_s101
                                 + 1.06264 w_k1_DIS_s101
                                 + 1.02597 w_k1_FDX_s101
                                 + 0.953288 w_k1_IBM_s101
                                 + 1.13757 w_k1_INTC_s101
                                 + 0.975464 w_k1_KO_s101
                                 + 1.06076 w_k1_MAT_s101
                                 + 0.994779 w_k1_MSFT_s101
                                 + 0.897039 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.960997 w_k1_ADM_s102
                                 + 0.99646 w_k1_AXP_s102
                                 + 0.935395 w_k1_DIS_s102
                                 + 1.05509 w_k1_FDX_s102
                                 + 0.711977 w_k1_IBM_s102
                                 + 1.06633 w_k1_INTC_s102
                                 + 0.915168 w_k1_KO_s102
                                 + 0.672529 w_k1_MAT_s102
                                 + 1.01434 w_k1_MSFT_s102
                                 + 0.906916 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.16679 w_k1_ADM_s102
                                 + 1.19689 w_k1_AXP_s102
                                 + 1.29654 w_k1_DIS_s102
                                 + 1.16796 w_k1_FDX_s102
                                 + 0.970037 w_k1_IBM_s102
                                 + 1.19362 w_k1_INTC_s102
                                 + 1.18058 w_k1_KO_s102 + 1.06733 w_k1_MAT_s102
                                 + 1.16969 w_k1_MSFT_s102
                                 + 1.06278 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.02876 w_k1_ADM_s102
                                 + 0.912237 w_k1_AXP_s102
                                 + 0.812412 w_k1_DIS_s102
                                 + 0.798464 w_k1_FDX_s102
                                 + 1.04902 w_k1_IBM_s102
                                 + 1.02899 w_k1_INTC_s102
                                 + 1.04633 w_k1_KO_s102
                                 + 0.971986 w_k1_MAT_s102
                                 + 0.76689 w_k1_MSFT_s102
                                 + 0.868474 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.41091 w_k1_ADM_s102
                                 + 1.22451 w_k1_AXP_s102 + 1.5053 w_k1_DIS_s102
                                 + 1.15411 w_k1_FDX_s102
                                 + 1.64362 w_k1_IBM_s102
                                 + 1.11424 w_k1_INTC_s102
                                 + 1.67595 w_k1_KO_s102 + 1.84538 w_k1_MAT_s102
                                 + 1.23458 w_k1_MSFT_s102
                                 + 1.42599 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.20753 w_k1_ADM_s102
                                 + 1.11172 w_k1_AXP_s102
                                 + 1.74154 w_k1_DIS_s102
                                 + 1.28862 w_k1_FDX_s102
                                 + 1.08381 w_k1_IBM_s102
                                 + 0.980028 w_k1_INTC_s102
                                 + 1.34199 w_k1_KO_s102 + 0.82397 w_k1_MAT_s102
                                 + 1.26546 w_k1_MSFT_s102
                                 + 1.20678 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.948894 w_k1_ADM_s102
                                 + 0.639934 w_k1_AXP_s102
                                 + 0.751195 w_k1_DIS_s102
                                 + 0.862328 w_k1_FDX_s102
                                 + 0.731473 w_k1_IBM_s102
                                 + 0.893104 w_k1_INTC_s102
                                 + 0.928686 w_k1_KO_s102
                                 + 0.78741 w_k1_MAT_s102
                                 + 0.866605 w_k1_MSFT_s102
                                 + 0.896271 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.06614 w_k1_ADM_s102
                                 + 0.86004 w_k1_AXP_s102
                                 + 0.914554 w_k1_DIS_s102
                                 + 1.00436 w_k1_FDX_s102
                                 + 0.809586 w_k1_IBM_s102
                                 + 0.988658 w_k1_INTC_s102
                                 + 0.97963 w_k1_KO_s102
                                 + 0.652835 w_k1_MAT_s102
                                 + 0.94944 w_k1_MSFT_s102
                                 + 0.945227 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.70298 w_k1_ADM_s102
                                 + 0.535859 w_k1_AXP_s102
                                 + 0.480193 w_k1_DIS_s102
                                 + 0.567949 w_k1_FDX_s102
                                 + 0.810632 w_k1_IBM_s102
                                 + 0.627594 w_k1_INTC_s102
                                 + 0.816832 w_k1_KO_s102
                                 + 0.751911 w_k1_MAT_s102
                                 + 0.626364 w_k1_MSFT_s102
                                 + 0.774554 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.847024 w_k1_ADM_s103
                                 + 0.816275 w_k1_AXP_s103
                                 + 0.748201 w_k1_DIS_s103
                                 + 0.822134 w_k1_FDX_s103
                                 + 1.10462 w_k1_IBM_s103
                                 + 0.768638 w_k1_INTC_s103
                                 + 0.949927 w_k1_KO_s103 + 1.242 w_k1_MAT_s103
                                 + 0.850513 w_k1_MSFT_s103
                                 + 0.896013 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.955689 w_k1_ADM_s103
                                 + 0.728374 w_k1_AXP_s103
                                 + 0.771769 w_k1_DIS_s103
                                 + 0.875541 w_k1_FDX_s103
                                 + 0.737687 w_k1_IBM_s103
                                 + 0.730912 w_k1_INTC_s103
                                 + 0.807551 w_k1_KO_s103
                                 + 0.913674 w_k1_MAT_s103
                                 + 0.7909 w_k1_MSFT_s103
                                 + 0.863247 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.16258 w_k1_ADM_s103
                                 + 1.13903 w_k1_AXP_s103
                                 + 1.31861 w_k1_DIS_s103
                                 + 1.18743 w_k1_FDX_s103
                                 + 1.25095 w_k1_IBM_s103
                                 + 0.945142 w_k1_INTC_s103
                                 + 1.11324 w_k1_KO_s103 + 1.09385 w_k1_MAT_s103
                                 + 0.99857 w_k1_MSFT_s103
                                 + 1.05554 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.02967 w_k1_ADM_s103
                                 + 0.908367 w_k1_AXP_s103
                                 + 0.98635 w_k1_DIS_s103
                                 + 1.02599 w_k1_FDX_s103
                                 + 0.871802 w_k1_IBM_s103
                                 + 0.957249 w_k1_INTC_s103
                                 + 0.964949 w_k1_KO_s103
                                 + 1.17685 w_k1_MAT_s103
                                 + 0.866707 w_k1_MSFT_s103
                                 + 0.896532 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.4834 w_k1_ADM_s103
                                 + 1.59201 w_k1_AXP_s103
                                 + 1.66525 w_k1_DIS_s103
                                 + 1.47743 w_k1_FDX_s103
                                 + 1.56432 w_k1_IBM_s103
                                 + 0.947678 w_k1_INTC_s103
                                 + 1.42968 w_k1_KO_s103 + 2.2502 w_k1_MAT_s103
                                 + 1.04394 w_k1_MSFT_s103
                                 + 1.24799 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.13502 w_k1_ADM_s103
                                 + 1.14773 w_k1_AXP_s103
                                 + 1.04939 w_k1_DIS_s103
                                 + 1.05925 w_k1_FDX_s103
                                 + 1.15804 w_k1_IBM_s103
                                 + 0.900016 w_k1_INTC_s103
                                 + 1.20599 w_k1_KO_s103 + 1.75484 w_k1_MAT_s103
                                 + 0.949415 w_k1_MSFT_s103
                                 + 1.1069 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.13204 w_k1_ADM_s103
                                 + 1.0771 w_k1_AXP_s103 + 1.29442 w_k1_DIS_s103
                                 + 1.28774 w_k1_FDX_s103
                                 + 0.867008 w_k1_IBM_s103
                                 + 0.87145 w_k1_INTC_s103
                                 + 0.960138 w_k1_KO_s103
                                 + 0.832309 w_k1_MAT_s103
                                 + 1.00383 w_k1_MSFT_s103
                                 + 1.04857 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.911492 w_k1_ADM_s103
                                 + 0.943898 w_k1_AXP_s103
                                 + 0.878116 w_k1_DIS_s103
                                 + 0.773313 w_k1_FDX_s103
                                 + 1.35472 w_k1_IBM_s103
                                 + 0.778611 w_k1_INTC_s103
                                 + 1.09283 w_k1_KO_s103 + 1.65077 w_k1_MAT_s103
                                 + 0.855136 w_k1_MSFT_s103
                                 + 1.07402 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.14068 w_k1_ADM_s104
                                 + 1.18946 w_k1_AXP_s104
                                 + 1.02807 w_k1_DIS_s104
                                 + 1.01713 w_k1_FDX_s104
                                 + 0.779095 w_k1_IBM_s104
                                 + 0.918404 w_k1_INTC_s104
                                 + 0.951232 w_k1_KO_s104
                                 + 0.918331 w_k1_MAT_s104
                                 + 0.879624 w_k1_MSFT_s104
                                 + 1.0604 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.12491 w_k1_ADM_s104
                                 + 1.29732 w_k1_AXP_s104
                                 + 1.15988 w_k1_DIS_s104
                                 + 1.21416 w_k1_FDX_s104
                                 + 0.814207 w_k1_IBM_s104
                                 + 1.15711 w_k1_INTC_s104
                                 + 1.06166 w_k1_KO_s104 + 0.97603 w_k1_MAT_s104
                                 + 1.2086 w_k1_MSFT_s104 + 1.16493 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.876692 w_k1_ADM_s104
                                 + 0.833145 w_k1_AXP_s104
                                 + 0.880476 w_k1_DIS_s104
                                 + 1.01253 w_k1_FDX_s104
                                 + 0.656115 w_k1_IBM_s104
                                 + 1.02346 w_k1_INTC_s104
                                 + 0.900925 w_k1_KO_s104
                                 + 0.560603 w_k1_MAT_s104
                                 + 0.982751 w_k1_MSFT_s104
                                 + 0.983035 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.25174 w_k1_ADM_s104
                                 + 1.86628 w_k1_AXP_s104
                                 + 1.37216 w_k1_DIS_s104
                                 + 1.38957 w_k1_FDX_s104
                                 + 1.14462 w_k1_IBM_s104
                                 + 1.08255 w_k1_INTC_s104
                                 + 1.43934 w_k1_KO_s104 + 1.97852 w_k1_MAT_s104
                                 + 1.18237 w_k1_MSFT_s104
                                 + 1.40466 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.06168 w_k1_ADM_s104
                                 + 1.23966 w_k1_AXP_s104
                                 + 1.40753 w_k1_DIS_s104
                                 + 1.15738 w_k1_FDX_s104
                                 + 1.23988 w_k1_IBM_s104
                                 + 1.03975 w_k1_INTC_s104
                                 + 1.11928 w_k1_KO_s104 + 1.30853 w_k1_MAT_s104
                                 + 1.12041 w_k1_MSFT_s104
                                 + 1.03793 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.696431 w_k1_ADM_s104
                                 + 0.641261 w_k1_AXP_s104
                                 + 0.492879 w_k1_DIS_s104
                                 + 0.725552 w_k1_FDX_s104
                                 + 0.764656 w_k1_IBM_s104
                                 + 0.922882 w_k1_INTC_s104
                                 + 0.704228 w_k1_KO_s104
                                 + 0.820482 w_k1_MAT_s104
                                 + 0.791241 w_k1_MSFT_s104
                                 + 0.797633 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.965224 w_k1_ADM_s104
                                 + 0.820976 w_k1_AXP_s104
                                 + 0.753293 w_k1_DIS_s104
                                 + 0.905595 w_k1_FDX_s104
                                 + 0.959031 w_k1_IBM_s104
                                 + 0.787097 w_k1_INTC_s104
                                 + 0.813118 w_k1_KO_s104
                                 + 1.00083 w_k1_MAT_s104
                                 + 0.826199 w_k1_MSFT_s104
                                 + 0.937504 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.995508 w_k1_ADM_s104
                                 + 0.876846 w_k1_AXP_s104
                                 + 0.816194 w_k1_DIS_s104
                                 + 0.895091 w_k1_FDX_s104
                                 + 1.25175 w_k1_IBM_s104
                                 + 0.816197 w_k1_INTC_s104
                                 + 1.15886 w_k1_KO_s104 + 1.45923 w_k1_MAT_s104
                                 + 0.841019 w_k1_MSFT_s104
                                 + 1.02016 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.01453 w_k1_ADM_s105
                                 + 1.59609 w_k1_AXP_s105
                                 + 2.17821 w_k1_DIS_s105
                                 + 1.51193 w_k1_FDX_s105
                                 + 1.59455 w_k1_IBM_s105
                                 + 1.28141 w_k1_INTC_s105
                                 + 1.23471 w_k1_KO_s105 + 1.63232 w_k1_MAT_s105
                                 + 1.30889 w_k1_MSFT_s105
                                 + 1.23737 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.606684 w_k1_ADM_s105
                                 + 1.06197 w_k1_AXP_s105
                                 + 0.954321 w_k1_DIS_s105
                                 + 0.881658 w_k1_FDX_s105
                                 + 1.33534 w_k1_IBM_s105
                                 + 1.11929 w_k1_INTC_s105
                                 + 1.00746 w_k1_KO_s105 + 1.11199 w_k1_MAT_s105
                                 + 1.01863 w_k1_MSFT_s105
                                 + 0.939831 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.495941 w_k1_ADM_s105
                                 + 0.928915 w_k1_AXP_s105
                                 + 0.812784 w_k1_DIS_s105
                                 + 0.803689 w_k1_FDX_s105
                                 + 1.09497 w_k1_IBM_s105
                                 + 0.942685 w_k1_INTC_s105
                                 + 0.826197 w_k1_KO_s105
                                 + 1.06832 w_k1_MAT_s105
                                 + 0.867133 w_k1_MSFT_s105
                                 + 0.909185 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.66531 w_k1_ADM_s105
                                 + 0.883846 w_k1_AXP_s105
                                 + 0.896957 w_k1_DIS_s105
                                 + 0.978147 w_k1_FDX_s105
                                 + 0.76352 w_k1_IBM_s105
                                 + 0.956062 w_k1_INTC_s105
                                 + 0.734788 w_k1_KO_s105
                                 + 0.662638 w_k1_MAT_s105
                                 + 0.914886 w_k1_MSFT_s105
                                 + 0.845274 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.784415 w_k1_ADM_s105
                                 + 1.4195 w_k1_AXP_s105 + 1.30345 w_k1_DIS_s105
                                 + 1.30651 w_k1_FDX_s105
                                 + 0.901968 w_k1_IBM_s105
                                 + 1.62247 w_k1_INTC_s105
                                 + 1.06662 w_k1_KO_s105 + 1.12274 w_k1_MAT_s105
                                 + 1.46377 w_k1_MSFT_s105
                                 + 1.01829 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.529336 w_k1_ADM_s105
                                 + 0.55713 w_k1_AXP_s105
                                 + 0.655413 w_k1_DIS_s105
                                 + 0.780264 w_k1_FDX_s105
                                 + 0.578946 w_k1_IBM_s105
                                 + 1.0385 w_k1_INTC_s105
                                 + 0.688837 w_k1_KO_s105
                                 + 0.551654 w_k1_MAT_s105
                                 + 0.983455 w_k1_MSFT_s105
                                 + 0.697489 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.617337 w_k1_ADM_s105
                                 + 0.769525 w_k1_AXP_s105
                                 + 0.742929 w_k1_DIS_s105
                                 + 0.747947 w_k1_FDX_s105
                                 + 0.93906 w_k1_IBM_s105
                                 + 1.05911 w_k1_INTC_s105
                                 + 0.864491 w_k1_KO_s105
                                 + 0.620839 w_k1_MAT_s105
                                 + 0.889639 w_k1_MSFT_s105
                                 + 0.810084 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.832502 w_k1_ADM_s105
                                 + 1.17295 w_k1_AXP_s105
                                 + 1.38648 w_k1_DIS_s105
                                 + 1.09322 w_k1_FDX_s105
                                 + 1.09539 w_k1_IBM_s105
                                 + 0.917835 w_k1_INTC_s105
                                 + 1.03049 w_k1_KO_s105 + 1.0041 w_k1_MAT_s105
                                 + 1.22895 w_k1_MSFT_s105
                                 + 1.09232 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.551649 w_k1_ADM_s106
                                 + 0.758897 w_k1_AXP_s106
                                 + 0.797738 w_k1_DIS_s106
                                 + 0.783971 w_k1_FDX_s106
                                 + 0.920868 w_k1_IBM_s106
                                 + 1.01038 w_k1_INTC_s106
                                 + 0.882373 w_k1_KO_s106
                                 + 0.929656 w_k1_MAT_s106
                                 + 0.863141 w_k1_MSFT_s106
                                 + 0.911217 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.00709 w_k1_ADM_s106
                                 + 1.21829 w_k1_AXP_s106 + 1.3092 w_k1_DIS_s106
                                 + 1.01989 w_k1_FDX_s106
                                 + 1.85432 w_k1_IBM_s106
                                 + 0.888935 w_k1_INTC_s106
                                 + 1.46555 w_k1_KO_s106 + 2.30726 w_k1_MAT_s106
                                 + 0.925266 w_k1_MSFT_s106
                                 + 1.04498 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.994448 w_k1_ADM_s106
                                 + 0.955909 w_k1_AXP_s106
                                 + 0.849979 w_k1_DIS_s106
                                 + 0.994892 w_k1_FDX_s106
                                 + 0.629908 w_k1_IBM_s106
                                 + 1.00129 w_k1_INTC_s106
                                 + 0.733809 w_k1_KO_s106
                                 + 0.713481 w_k1_MAT_s106
                                 + 0.890932 w_k1_MSFT_s106
                                 + 0.858915 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.682823 w_k1_ADM_s106
                                 + 0.873166 w_k1_AXP_s106
                                 + 1.15103 w_k1_DIS_s106
                                 + 0.805613 w_k1_FDX_s106
                                 + 1.33833 w_k1_IBM_s106
                                 + 0.969983 w_k1_INTC_s106
                                 + 1.14995 w_k1_KO_s106 + 1.40237 w_k1_MAT_s106
                                 + 0.847625 w_k1_MSFT_s106
                                 + 0.88404 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.12525 w_k1_ADM_s106
                                 + 1.59725 w_k1_AXP_s106
                                 + 1.55235 w_k1_DIS_s106
                                 + 1.33636 w_k1_FDX_s106
                                 + 1.42551 w_k1_IBM_s106
                                 + 1.24837 w_k1_INTC_s106
                                 + 1.51708 w_k1_KO_s106 + 1.88201 w_k1_MAT_s106
                                 + 1.20605 w_k1_MSFT_s106
                                 + 1.28638 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.00453 w_k1_ADM_s106
                                 + 1.26647 w_k1_AXP_s106
                                 + 1.37347 w_k1_DIS_s106
                                 + 1.39333 w_k1_FDX_s106
                                 + 1.03232 w_k1_IBM_s106
                                 + 1.08411 w_k1_INTC_s106
                                 + 1.06053 w_k1_KO_s106 + 1.4309 w_k1_MAT_s106
                                 + 1.20572 w_k1_MSFT_s106
                                 + 1.17042 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.03021 w_k1_ADM_s106
                                 + 1.14834 w_k1_AXP_s106
                                 + 1.19313 w_k1_DIS_s106
                                 + 1.14486 w_k1_FDX_s106
                                 + 0.84726 w_k1_IBM_s106
                                 + 0.986957 w_k1_INTC_s106
                                 + 0.886021 w_k1_KO_s106
                                 + 1.19853 w_k1_MAT_s106
                                 + 0.873071 w_k1_MSFT_s106
                                 + 0.962215 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.912367 w_k1_ADM_s106
                                 + 1.12863 w_k1_AXP_s106
                                 + 1.12772 w_k1_DIS_s106
                                 + 1.07628 w_k1_FDX_s106
                                 + 1.17547 w_k1_IBM_s106
                                 + 0.984695 w_k1_INTC_s106
                                 + 1.07599 w_k1_KO_s106 + 1.50113 w_k1_MAT_s106
                                 + 0.861938 w_k1_MSFT_s106
                                 + 1.0219 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.27884 w_k1_ADM_s107
                                 + 1.23751 w_k1_AXP_s107
                                 + 1.15164 w_k1_DIS_s107
                                 + 1.31576 w_k1_FDX_s107
                                 + 0.905666 w_k1_IBM_s107
                                 + 1.13722 w_k1_INTC_s107
                                 + 1.04081 w_k1_KO_s107
                                 + 0.596843 w_k1_MAT_s107
                                 + 1.02679 w_k1_MSFT_s107
                                 + 1.03341 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.860587 w_k1_ADM_s107
                                 + 0.774664 w_k1_AXP_s107
                                 + 0.52744 w_k1_DIS_s107
                                 + 0.73266 w_k1_FDX_s107
                                 + 0.744382 w_k1_IBM_s107
                                 + 0.690893 w_k1_INTC_s107
                                 + 0.690484 w_k1_KO_s107
                                 + 0.68589 w_k1_MAT_s107
                                 + 0.673154 w_k1_MSFT_s107
                                 + 0.828055 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.20503 w_k1_ADM_s107
                                 + 0.923273 w_k1_AXP_s107
                                 + 0.999445 w_k1_DIS_s107
                                 + 1.05834 w_k1_FDX_s107
                                 + 0.941676 w_k1_IBM_s107
                                 + 1.0918 w_k1_INTC_s107 + 1.02367 w_k1_KO_s107
                                 + 0.78512 w_k1_MAT_s107
                                 + 0.935735 w_k1_MSFT_s107
                                 + 1.10962 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.91308 w_k1_ADM_s107
                                 + 0.797687 w_k1_AXP_s107
                                 + 0.74647 w_k1_DIS_s107
                                 + 0.834209 w_k1_FDX_s107
                                 + 0.720359 w_k1_IBM_s107
                                 + 0.845382 w_k1_INTC_s107
                                 + 0.950207 w_k1_KO_s107
                                 + 0.868364 w_k1_MAT_s107
                                 + 0.849024 w_k1_MSFT_s107
                                 + 0.828924 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.07264 w_k1_ADM_s107
                                 + 0.832258 w_k1_AXP_s107
                                 + 0.749547 w_k1_DIS_s107
                                 + 1.01271 w_k1_FDX_s107
                                 + 0.714757 w_k1_IBM_s107
                                 + 0.931602 w_k1_INTC_s107
                                 + 0.748815 w_k1_KO_s107
                                 + 0.473137 w_k1_MAT_s107
                                 + 0.859433 w_k1_MSFT_s107
                                 + 0.77053 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.859855 w_k1_ADM_s107
                                 + 0.934107 w_k1_AXP_s107
                                 + 0.795305 w_k1_DIS_s107
                                 + 0.90581 w_k1_FDX_s107
                                 + 1.16365 w_k1_IBM_s107
                                 + 0.83666 w_k1_INTC_s107
                                 + 1.00089 w_k1_KO_s107 + 1.12654 w_k1_MAT_s107
                                 + 0.82875 w_k1_MSFT_s107
                                 + 1.09316 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.43327 w_k1_ADM_s107
                                 + 1.48979 w_k1_AXP_s107 + 1.466 w_k1_DIS_s107
                                 + 1.37405 w_k1_FDX_s107
                                 + 1.44904 w_k1_IBM_s107
                                 + 1.33399 w_k1_INTC_s107
                                 + 1.39481 w_k1_KO_s107 + 1.59211 w_k1_MAT_s107
                                 + 1.14134 w_k1_MSFT_s107
                                 + 1.28685 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.26639 w_k1_ADM_s107
                                 + 1.18407 w_k1_AXP_s107 + 1.3173 w_k1_DIS_s107
                                 + 1.20687 w_k1_FDX_s107
                                 + 1.63901 w_k1_IBM_s107
                                 + 1.12354 w_k1_INTC_s107
                                 + 1.53513 w_k1_KO_s107 + 1.33902 w_k1_MAT_s107
                                 + 1.15542 w_k1_MSFT_s107
                                 + 1.09736 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.05746 w_k1_ADM_s110
                                 + 1.38669 w_k1_AXP_s110
                                 + 1.70241 w_k1_DIS_s110
                                 + 1.23068 w_k1_FDX_s110
                                 + 1.46242 w_k1_IBM_s110
                                 + 1.27462 w_k1_INTC_s110
                                 + 1.28978 w_k1_KO_s110 + 1.81534 w_k1_MAT_s110
                                 + 1.10012 w_k1_MSFT_s110
                                 + 1.28111 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.916003 w_k1_ADM_s110
                                 + 0.929993 w_k1_AXP_s110
                                 + 0.742979 w_k1_DIS_s110
                                 + 0.894979 w_k1_FDX_s110
                                 + 0.824541 w_k1_IBM_s110
                                 + 0.883532 w_k1_INTC_s110
                                 + 0.840515 w_k1_KO_s110
                                 + 0.916241 w_k1_MAT_s110
                                 + 0.87618 w_k1_MSFT_s110
                                 + 0.925812 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.880741 w_k1_ADM_s110
                                 + 0.898062 w_k1_AXP_s110
                                 + 0.885212 w_k1_DIS_s110
                                 + 0.777357 w_k1_FDX_s110
                                 + 1.33088 w_k1_IBM_s110
                                 + 0.931014 w_k1_INTC_s110
                                 + 0.986354 w_k1_KO_s110
                                 + 1.31175 w_k1_MAT_s110
                                 + 0.84113 w_k1_MSFT_s110
                                 + 0.977672 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.952918 w_k1_ADM_s110
                                 + 0.9011 w_k1_AXP_s110 + 1.15869 w_k1_DIS_s110
                                 + 1.0273 w_k1_FDX_s110
                                 + 0.764137 w_k1_IBM_s110
                                 + 0.966813 w_k1_INTC_s110
                                 + 0.938597 w_k1_KO_s110
                                 + 0.933529 w_k1_MAT_s110
                                 + 1.06229 w_k1_MSFT_s110
                                 + 1.01375 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.17862 w_k1_ADM_s110
                                 + 1.27321 w_k1_AXP_s110 + 1.55 w_k1_DIS_s110
                                 + 1.37157 w_k1_FDX_s110 + 0.9062 w_k1_IBM_s110
                                 + 1.21497 w_k1_INTC_s110
                                 + 1.06511 w_k1_KO_s110 + 1.2861 w_k1_MAT_s110
                                 + 1.37944 w_k1_MSFT_s110
                                 + 1.30272 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.905866 w_k1_ADM_s110
                                 + 1.12892 w_k1_AXP_s110
                                 + 1.41429 w_k1_DIS_s110 + 1.1067 w_k1_FDX_s110
                                 + 1.17094 w_k1_IBM_s110
                                 + 1.12269 w_k1_INTC_s110
                                 + 1.23393 w_k1_KO_s110 + 1.11695 w_k1_MAT_s110
                                 + 1.0908 w_k1_MSFT_s110 + 1.09757 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.927208 w_k1_ADM_s110
                                 + 0.851083 w_k1_AXP_s110
                                 + 1.10605 w_k1_DIS_s110
                                 + 1.05613 w_k1_FDX_s110
                                 + 0.962773 w_k1_IBM_s110
                                 + 1.14656 w_k1_INTC_s110
                                 + 0.950802 w_k1_KO_s110
                                 + 0.868991 w_k1_MAT_s110
                                 + 1.01459 w_k1_MSFT_s110
                                 + 1.05515 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.726256 w_k1_ADM_s110
                                 + 0.689014 w_k1_AXP_s110
                                 + 0.673789 w_k1_DIS_s110
                                 + 0.698644 w_k1_FDX_s110
                                 + 0.821788 w_k1_IBM_s110
                                 + 0.735344 w_k1_INTC_s110
                                 + 0.772388 w_k1_KO_s110
                                 + 0.746963 w_k1_MAT_s110
                                 + 0.70384 w_k1_MSFT_s110
                                 + 0.822697 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.0346 w_k1_ADM_s111
                                 + 1.25479 w_k1_AXP_s111
                                 + 1.22731 w_k1_DIS_s111
                                 + 1.30441 w_k1_FDX_s111
                                 + 0.897453 w_k1_IBM_s111
                                 + 0.85796 w_k1_INTC_s111 + 0.9736 w_k1_KO_s111
                                 + 0.969786 w_k1_MAT_s111
                                 + 1.02898 w_k1_MSFT_s111
                                 + 1.07939 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.935939 w_k1_ADM_s111
                                 + 1.33595 w_k1_AXP_s111
                                 + 1.48667 w_k1_DIS_s111
                                 + 1.41251 w_k1_FDX_s111 + 1.1165 w_k1_IBM_s111
                                 + 1.38187 w_k1_INTC_s111
                                 + 1.42257 w_k1_KO_s111 + 1.15136 w_k1_MAT_s111
                                 + 1.29811 w_k1_MSFT_s111
                                 + 1.22856 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.124 w_k1_ADM_s111
                                 + 1.1964 w_k1_AXP_s111 + 1.66857 w_k1_DIS_s111
                                 + 1.24007 w_k1_FDX_s111
                                 + 1.53364 w_k1_IBM_s111
                                 + 1.06884 w_k1_INTC_s111
                                 + 1.72267 w_k1_KO_s111 + 1.43346 w_k1_MAT_s111
                                 + 1.11823 w_k1_MSFT_s111
                                 + 1.21944 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.12363 w_k1_ADM_s111
                                 + 0.893958 w_k1_AXP_s111
                                 + 0.95614 w_k1_DIS_s111
                                 + 1.11626 w_k1_FDX_s111
                                 + 0.777408 w_k1_IBM_s111
                                 + 0.955908 w_k1_INTC_s111
                                 + 0.887153 w_k1_KO_s111
                                 + 0.715049 w_k1_MAT_s111
                                 + 0.954911 w_k1_MSFT_s111
                                 + 0.977998 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.976624 w_k1_ADM_s111
                                 + 1.22227 w_k1_AXP_s111
                                 + 0.971803 w_k1_DIS_s111
                                 + 1.04839 w_k1_FDX_s111
                                 + 1.09218 w_k1_IBM_s111
                                 + 0.865492 w_k1_INTC_s111
                                 + 1.12047 w_k1_KO_s111 + 1.19991 w_k1_MAT_s111
                                 + 0.92215 w_k1_MSFT_s111
                                 + 1.06772 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.901303 w_k1_ADM_s111
                                 + 1.08251 w_k1_AXP_s111
                                 + 1.23292 w_k1_DIS_s111
                                 + 0.992058 w_k1_FDX_s111
                                 + 1.19911 w_k1_IBM_s111
                                 + 0.840656 w_k1_INTC_s111
                                 + 1.2799 w_k1_KO_s111 + 1.39723 w_k1_MAT_s111
                                 + 1.00166 w_k1_MSFT_s111
                                 + 1.17829 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.832052 w_k1_ADM_s111
                                 + 0.992619 w_k1_AXP_s111
                                 + 0.88037 w_k1_DIS_s111
                                 + 0.897359 w_k1_FDX_s111
                                 + 1.34947 w_k1_IBM_s111
                                 + 0.790344 w_k1_INTC_s111
                                 + 0.968784 w_k1_KO_s111 + 1.1632 w_k1_MAT_s111
                                 + 0.783596 w_k1_MSFT_s111
                                 + 0.906695 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.840335 w_k1_ADM_s111
                                 + 0.836472 w_k1_AXP_s111
                                 + 0.673382 w_k1_DIS_s111
                                 + 0.821232 w_k1_FDX_s111
                                 + 0.833434 w_k1_IBM_s111
                                 + 0.778339 w_k1_INTC_s111
                                 + 0.771114 w_k1_KO_s111
                                 + 0.844792 w_k1_MAT_s111
                                 + 0.748355 w_k1_MSFT_s111
                                 + 0.821208 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.23824 w_k1_ADM_s112
                                 + 1.13273 w_k1_AXP_s112
                                 + 0.696355 w_k1_DIS_s112
                                 + 0.878745 w_k1_FDX_s112 + 1.01 w_k1_IBM_s112
                                 + 0.721554 w_k1_INTC_s112
                                 + 0.879342 w_k1_KO_s112
                                 + 1.09221 w_k1_MAT_s112
                                 + 0.767341 w_k1_MSFT_s112
                                 + 0.861279 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.74956 w_k1_ADM_s112
                                 + 0.691841 w_k1_AXP_s112
                                 + 0.571961 w_k1_DIS_s112
                                 + 0.618537 w_k1_FDX_s112
                                 + 0.888532 w_k1_IBM_s112
                                 + 0.6266 w_k1_INTC_s112
                                 + 0.672627 w_k1_KO_s112
                                 + 0.60314 w_k1_MAT_s112
                                 + 0.598314 w_k1_MSFT_s112
                                 + 0.703005 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.17158 w_k1_ADM_s112
                                 + 0.815997 w_k1_AXP_s112
                                 + 0.857918 w_k1_DIS_s112
                                 + 0.836561 w_k1_FDX_s112
                                 + 0.810569 w_k1_IBM_s112
                                 + 0.707449 w_k1_INTC_s112
                                 + 0.832686 w_k1_KO_s112
                                 + 0.722292 w_k1_MAT_s112
                                 + 0.896382 w_k1_MSFT_s112
                                 + 0.849169 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.01217 w_k1_ADM_s112
                                 + 0.963531 w_k1_AXP_s112
                                 + 0.998616 w_k1_DIS_s112
                                 + 0.861312 w_k1_FDX_s112
                                 + 1.00343 w_k1_IBM_s112
                                 + 0.928598 w_k1_INTC_s112
                                 + 1.00196 w_k1_KO_s112
                                 + 0.937211 w_k1_MAT_s112
                                 + 1.05978 w_k1_MSFT_s112
                                 + 0.972922 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.22333 w_k1_ADM_s112
                                 + 1.19101 w_k1_AXP_s112
                                 + 1.24376 w_k1_DIS_s112
                                 + 0.981487 w_k1_FDX_s112
                                 + 1.30354 w_k1_IBM_s112
                                 + 0.890052 w_k1_INTC_s112
                                 + 1.26008 w_k1_KO_s112 + 1.39084 w_k1_MAT_s112
                                 + 1.00317 w_k1_MSFT_s112
                                 + 0.999224 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.33817 w_k1_ADM_s112
                                 + 1.12744 w_k1_AXP_s112
                                 + 0.92997 w_k1_DIS_s112
                                 + 0.869194 w_k1_FDX_s112
                                 + 1.52042 w_k1_IBM_s112
                                 + 0.710859 w_k1_INTC_s112
                                 + 1.15483 w_k1_KO_s112 + 1.4236 w_k1_MAT_s112
                                 + 0.924424 w_k1_MSFT_s112
                                 + 0.923927 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.61744 w_k1_ADM_s112
                                 + 1.62339 w_k1_AXP_s112
                                 + 1.65449 w_k1_DIS_s112
                                 + 1.38832 w_k1_FDX_s112
                                 + 1.33219 w_k1_IBM_s112
                                 + 1.10489 w_k1_INTC_s112
                                 + 1.28372 w_k1_KO_s112 + 1.6868 w_k1_MAT_s112
                                 + 1.21065 w_k1_MSFT_s112
                                 + 1.22075 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.27046 w_k1_ADM_s112
                                 + 1.10577 w_k1_AXP_s112
                                 + 1.05724 w_k1_DIS_s112
                                 + 1.14108 w_k1_FDX_s112
                                 + 1.01077 w_k1_IBM_s112
                                 + 0.915591 w_k1_INTC_s112
                                 + 1.09202 w_k1_KO_s112 + 1.10191 w_k1_MAT_s112
                                 + 1.07886 w_k1_MSFT_s112
                                 + 1.00921 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.47155 w_k1_ADM_s113
                                 + 1.62492 w_k1_AXP_s113
                                 + 1.90096 w_k1_DIS_s113
                                 + 1.49101 w_k1_FDX_s113
                                 + 1.33066 w_k1_IBM_s113
                                 + 1.77217 w_k1_INTC_s113
                                 + 1.53052 w_k1_KO_s113 + 1.18038 w_k1_MAT_s113
                                 + 1.50239 w_k1_MSFT_s113
                                 + 1.42783 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.724415 w_k1_ADM_s113
                                 + 0.812855 w_k1_AXP_s113
                                 + 0.754121 w_k1_DIS_s113
                                 + 0.790978 w_k1_FDX_s113
                                 + 0.983112 w_k1_IBM_s113
                                 + 0.932832 w_k1_INTC_s113
                                 + 0.940534 w_k1_KO_s113 + 0.8393 w_k1_MAT_s113
                                 + 0.85126 w_k1_MSFT_s113
                                 + 0.883015 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.812917 w_k1_ADM_s113
                                 + 0.731219 w_k1_AXP_s113
                                 + 0.593371 w_k1_DIS_s113
                                 + 0.819453 w_k1_FDX_s113
                                 + 0.82067 w_k1_IBM_s113
                                 + 1.02437 w_k1_INTC_s113
                                 + 0.769589 w_k1_KO_s113
                                 + 0.625907 w_k1_MAT_s113
                                 + 0.925463 w_k1_MSFT_s113
                                 + 0.881558 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.01418 w_k1_ADM_s113
                                 + 0.870208 w_k1_AXP_s113
                                 + 0.92929 w_k1_DIS_s113
                                 + 1.08174 w_k1_FDX_s113
                                 + 0.661968 w_k1_IBM_s113
                                 + 1.44707 w_k1_INTC_s113
                                 + 0.800795 w_k1_KO_s113
                                 + 0.442064 w_k1_MAT_s113
                                 + 1.1422 w_k1_MSFT_s113 + 1.05539 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.686058 w_k1_ADM_s113
                                 + 0.596723 w_k1_AXP_s113
                                 + 0.445167 w_k1_DIS_s113
                                 + 0.673524 w_k1_FDX_s113
                                 + 0.63807 w_k1_IBM_s113
                                 + 0.711999 w_k1_INTC_s113
                                 + 0.622213 w_k1_KO_s113
                                 + 0.603405 w_k1_MAT_s113
                                 + 0.668199 w_k1_MSFT_s113
                                 + 0.787999 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.685611 w_k1_ADM_s113
                                 + 0.824438 w_k1_AXP_s113
                                 + 0.87582 w_k1_DIS_s113
                                 + 0.816137 w_k1_FDX_s113
                                 + 1.41946 w_k1_IBM_s113
                                 + 1.08433 w_k1_INTC_s113
                                 + 1.02908 w_k1_KO_s113 + 1.28364 w_k1_MAT_s113
                                 + 0.854758 w_k1_MSFT_s113
                                 + 0.976179 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.09553 w_k1_ADM_s113
                                 + 0.949436 w_k1_AXP_s113
                                 + 1.03443 w_k1_DIS_s113
                                 + 1.08548 w_k1_FDX_s113
                                 + 0.972219 w_k1_IBM_s113
                                 + 1.12035 w_k1_INTC_s113
                                 + 1.05662 w_k1_KO_s113
                                 + 0.777536 w_k1_MAT_s113
                                 + 1.1787 w_k1_MSFT_s113 + 1.10328 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.18915 w_k1_ADM_s113
                                 + 1.43852 w_k1_AXP_s113
                                 + 1.21816 w_k1_DIS_s113
                                 + 1.34068 w_k1_FDX_s113
                                 + 0.985929 w_k1_IBM_s113
                                 + 1.42172 w_k1_INTC_s113
                                 + 1.18891 w_k1_KO_s113
                                 + 0.865824 w_k1_MAT_s113
                                 + 1.27563 w_k1_MSFT_s113
                                 + 1.25498 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.32228 w_k1_ADM_s114
                                 + 1.53861 w_k1_AXP_s114
                                 + 1.81293 w_k1_DIS_s114 + 1.3756 w_k1_FDX_s114
                                 + 1.66025 w_k1_IBM_s114
                                 + 1.39441 w_k1_INTC_s114
                                 + 1.51582 w_k1_KO_s114 + 1.84589 w_k1_MAT_s114
                                 + 1.18897 w_k1_MSFT_s114
                                 + 1.38267 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.932926 w_k1_ADM_s114
                                 + 0.974371 w_k1_AXP_s114
                                 + 0.746079 w_k1_DIS_s114
                                 + 0.869857 w_k1_FDX_s114
                                 + 1.13429 w_k1_IBM_s114
                                 + 0.972785 w_k1_INTC_s114
                                 + 0.935466 w_k1_KO_s114
                                 + 1.28483 w_k1_MAT_s114
                                 + 0.868243 w_k1_MSFT_s114
                                 + 0.914912 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.02385 w_k1_ADM_s114
                                 + 1.01172 w_k1_AXP_s114
                                 + 1.02841 w_k1_DIS_s114
                                 + 0.995422 w_k1_FDX_s114
                                 + 1.1104 w_k1_IBM_s114
                                 + 0.953405 w_k1_INTC_s114
                                 + 1.1623 w_k1_KO_s114 + 1.31421 w_k1_MAT_s114
                                 + 1.05812 w_k1_MSFT_s114 + 1.1025 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.13299 w_k1_ADM_s114
                                 + 1.19433 w_k1_AXP_s114
                                 + 1.11445 w_k1_DIS_s114
                                 + 1.19014 w_k1_FDX_s114
                                 + 1.15873 w_k1_IBM_s114
                                 + 1.21209 w_k1_INTC_s114
                                 + 1.06776 w_k1_KO_s114 + 1.23162 w_k1_MAT_s114
                                 + 1.02932 w_k1_MSFT_s114
                                 + 1.06723 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.07436 w_k1_ADM_s114
                                 + 0.781102 w_k1_AXP_s114
                                 + 0.918497 w_k1_DIS_s114
                                 + 1.06932 w_k1_FDX_s114
                                 + 0.616333 w_k1_IBM_s114
                                 + 1.02228 w_k1_INTC_s114
                                 + 0.782869 w_k1_KO_s114
                                 + 0.552225 w_k1_MAT_s114
                                 + 0.888066 w_k1_MSFT_s114
                                 + 0.876017 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.52625 w_k1_ADM_s114
                                 + 1.23743 w_k1_AXP_s114
                                 + 1.58207 w_k1_DIS_s114
                                 + 1.32568 w_k1_FDX_s114
                                 + 1.16822 w_k1_IBM_s114
                                 + 1.18259 w_k1_INTC_s114
                                 + 1.27708 w_k1_KO_s114 + 1.67841 w_k1_MAT_s114
                                 + 1.31233 w_k1_MSFT_s114
                                 + 1.22365 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.754249 w_k1_ADM_s114
                                 + 0.622565 w_k1_AXP_s114
                                 + 0.714178 w_k1_DIS_s114
                                 + 0.776868 w_k1_FDX_s114
                                 + 0.900111 w_k1_IBM_s114
                                 + 0.908966 w_k1_INTC_s114
                                 + 0.906692 w_k1_KO_s114
                                 + 0.797168 w_k1_MAT_s114
                                 + 0.841284 w_k1_MSFT_s114
                                 + 0.895318 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.31546 w_k1_ADM_s114
                                 + 0.820635 w_k1_AXP_s114
                                 + 1.28824 w_k1_DIS_s114
                                 + 1.07755 w_k1_FDX_s114
                                 + 0.904301 w_k1_IBM_s114
                                 + 1.07949 w_k1_INTC_s114
                                 + 1.15617 w_k1_KO_s114
                                 + 0.751153 w_k1_MAT_s114
                                 + 1.18765 w_k1_MSFT_s114
                                 + 1.05291 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.3524 w_k1_ADM_s115
                                 + 1.32088 w_k1_AXP_s115
                                 + 1.71907 w_k1_DIS_s115
                                 + 1.40118 w_k1_FDX_s115
                                 + 0.806615 w_k1_IBM_s115
                                 + 1.07182 w_k1_INTC_s115
                                 + 1.18074 w_k1_KO_s115
                                 + 0.998758 w_k1_MAT_s115
                                 + 1.28299 w_k1_MSFT_s115
                                 + 1.17918 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.07907 w_k1_ADM_s115
                                 + 1.09193 w_k1_AXP_s115
                                 + 1.35733 w_k1_DIS_s115
                                 + 1.01442 w_k1_FDX_s115
                                 + 1.43973 w_k1_IBM_s115
                                 + 1.05119 w_k1_INTC_s115
                                 + 1.46001 w_k1_KO_s115 + 1.20609 w_k1_MAT_s115
                                 + 1.05299 w_k1_MSFT_s115 + 1.0953 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.35551 w_k1_ADM_s115
                                 + 1.45782 w_k1_AXP_s115
                                 + 1.45157 w_k1_DIS_s115 + 1.3142 w_k1_FDX_s115
                                 + 1.6311 w_k1_IBM_s115
                                 + 1.16209 w_k1_INTC_s115
                                 + 1.36071 w_k1_KO_s115 + 1.6532 w_k1_MAT_s115
                                 + 1.07814 w_k1_MSFT_s115 + 1.1761 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.19706 w_k1_ADM_s115
                                 + 1.40048 w_k1_AXP_s115
                                 + 1.28944 w_k1_DIS_s115
                                 + 1.33074 w_k1_FDX_s115
                                 + 0.943768 w_k1_IBM_s115
                                 + 1.4023 w_k1_INTC_s115 + 1.1254 w_k1_KO_s115
                                 + 0.919658 w_k1_MAT_s115
                                 + 1.18248 w_k1_MSFT_s115
                                 + 1.16039 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.941946 w_k1_ADM_s115
                                 + 0.778531 w_k1_AXP_s115
                                 + 0.940033 w_k1_DIS_s115
                                 + 0.917557 w_k1_FDX_s115
                                 + 0.799491 w_k1_IBM_s115
                                 + 1.01619 w_k1_INTC_s115
                                 + 1.02103 w_k1_KO_s115
                                 + 0.735078 w_k1_MAT_s115
                                 + 0.958242 w_k1_MSFT_s115
                                 + 0.929319 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.817129 w_k1_ADM_s115
                                 + 0.80409 w_k1_AXP_s115
                                 + 0.687237 w_k1_DIS_s115
                                 + 0.752029 w_k1_FDX_s115
                                 + 0.798351 w_k1_IBM_s115
                                 + 0.754092 w_k1_INTC_s115
                                 + 0.786617 w_k1_KO_s115
                                 + 0.964983 w_k1_MAT_s115
                                 + 0.728531 w_k1_MSFT_s115
                                 + 0.858455 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.909857 w_k1_ADM_s115
                                 + 1.29054 w_k1_AXP_s115
                                 + 1.01411 w_k1_DIS_s115 + 1.0263 w_k1_FDX_s115
                                 + 1.12196 w_k1_IBM_s115
                                 + 1.11176 w_k1_INTC_s115
                                 + 1.14358 w_k1_KO_s115 + 1.41086 w_k1_MAT_s115
                                 + 0.932131 w_k1_MSFT_s115
                                 + 0.991032 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.16047 w_k1_ADM_s115
                                 + 0.905037 w_k1_AXP_s115
                                 + 0.923585 w_k1_DIS_s115
                                 + 1.1331 w_k1_FDX_s115
                                 + 0.648745 w_k1_IBM_s115
                                 + 1.02695 w_k1_INTC_s115
                                 + 0.921451 w_k1_KO_s115
                                 + 0.593508 w_k1_MAT_s115
                                 + 0.990995 w_k1_MSFT_s115
                                 + 0.920605 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.09378 w_k1_ADM_s116
                                 + 1.01704 w_k1_AXP_s116
                                 + 1.03555 w_k1_DIS_s116
                                 + 1.02866 w_k1_FDX_s116
                                 + 0.773186 w_k1_IBM_s116
                                 + 0.98214 w_k1_INTC_s116
                                 + 0.89273 w_k1_KO_s116
                                 + 0.693125 w_k1_MAT_s116
                                 + 0.887867 w_k1_MSFT_s116
                                 + 0.943466 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.909289 w_k1_ADM_s116
                                 + 0.713999 w_k1_AXP_s116
                                 + 0.702144 w_k1_DIS_s116
                                 + 0.745297 w_k1_FDX_s116
                                 + 0.868055 w_k1_IBM_s116
                                 + 0.789929 w_k1_INTC_s116
                                 + 0.711547 w_k1_KO_s116
                                 + 0.678876 w_k1_MAT_s116
                                 + 0.69382 w_k1_MSFT_s116
                                 + 0.824673 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.10945 w_k1_ADM_s116
                                 + 1.18696 w_k1_AXP_s116
                                 + 1.34627 w_k1_DIS_s116
                                 + 1.17434 w_k1_FDX_s116
                                 + 1.13797 w_k1_IBM_s116
                                 + 1.52841 w_k1_INTC_s116
                                 + 1.09925 w_k1_KO_s116 + 1.08633 w_k1_MAT_s116
                                 + 1.18501 w_k1_MSFT_s116
                                 + 1.05448 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.148 w_k1_ADM_s116
                                 + 1.21696 w_k1_AXP_s116
                                 + 0.90032 w_k1_DIS_s116
                                 + 0.99573 w_k1_FDX_s116
                                 + 1.27271 w_k1_IBM_s116
                                 + 0.982688 w_k1_INTC_s116
                                 + 1.01781 w_k1_KO_s116 + 1.29027 w_k1_MAT_s116
                                 + 0.919904 w_k1_MSFT_s116
                                 + 1.03043 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.29946 w_k1_ADM_s116
                                 + 1.05697 w_k1_AXP_s116
                                 + 1.08612 w_k1_DIS_s116 + 1.2164 w_k1_FDX_s116
                                 + 0.658846 w_k1_IBM_s116
                                 + 1.1421 w_k1_INTC_s116
                                 + 0.869396 w_k1_KO_s116
                                 + 0.41613 w_k1_MAT_s116
                                 + 1.03633 w_k1_MSFT_s116
                                 + 1.05454 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.09276 w_k1_ADM_s116
                                 + 1.08077 w_k1_AXP_s116 + 1.2231 w_k1_DIS_s116
                                 + 1.0371 w_k1_FDX_s116 + 1.22808 w_k1_IBM_s116
                                 + 1.31902 w_k1_INTC_s116
                                 + 1.12472 w_k1_KO_s116
                                 + 0.838644 w_k1_MAT_s116
                                 + 1.13572 w_k1_MSFT_s116
                                 + 1.11187 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.988889 w_k1_ADM_s116
                                 + 0.922599 w_k1_AXP_s116
                                 + 0.873064 w_k1_DIS_s116
                                 + 0.915458 w_k1_FDX_s116
                                 + 0.776411 w_k1_IBM_s116
                                 + 1.06962 w_k1_INTC_s116
                                 + 0.85722 w_k1_KO_s116 + 0.69375 w_k1_MAT_s116
                                 + 0.862998 w_k1_MSFT_s116
                                 + 0.927673 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.51882 w_k1_ADM_s116
                                 + 1.42658 w_k1_AXP_s116 + 1.6904 w_k1_DIS_s116
                                 + 1.40921 w_k1_FDX_s116
                                 + 1.30249 w_k1_IBM_s116
                                 + 1.31056 w_k1_INTC_s116
                                 + 1.40334 w_k1_KO_s116
                                 + 0.918681 w_k1_MAT_s116
                                 + 1.23283 w_k1_MSFT_s116
                                 + 1.16761 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.958121 w_k1_ADM_s117
                                 + 1.07952 w_k1_AXP_s117
                                 + 0.97221 w_k1_DIS_s117
                                 + 1.02633 w_k1_FDX_s117
                                 + 1.27121 w_k1_IBM_s117
                                 + 1.03636 w_k1_INTC_s117 + 1.1205 w_k1_KO_s117
                                 + 1.28514 w_k1_MAT_s117
                                 + 1.1364 w_k1_MSFT_s117 + 1.03659 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.780971 w_k1_ADM_s117
                                 + 0.622216 w_k1_AXP_s117
                                 + 0.546549 w_k1_DIS_s117
                                 + 0.726123 w_k1_FDX_s117
                                 + 0.567394 w_k1_IBM_s117
                                 + 0.797829 w_k1_INTC_s117
                                 + 0.60648 w_k1_KO_s117 + 0.50162 w_k1_MAT_s117
                                 + 0.738871 w_k1_MSFT_s117
                                 + 0.676723 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.15072 w_k1_ADM_s117
                                 + 1.10321 w_k1_AXP_s117
                                 + 1.09336 w_k1_DIS_s117
                                 + 1.11989 w_k1_FDX_s117
                                 + 0.914039 w_k1_IBM_s117
                                 + 1.23531 w_k1_INTC_s117
                                 + 0.943649 w_k1_KO_s117
                                 + 1.02118 w_k1_MAT_s117
                                 + 1.07152 w_k1_MSFT_s117
                                 + 0.871866 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.19625 w_k1_ADM_s117
                                 + 1.36488 w_k1_AXP_s117
                                 + 1.18945 w_k1_DIS_s117
                                 + 1.33653 w_k1_FDX_s117 + 1.0804 w_k1_IBM_s117
                                 + 1.14201 w_k1_INTC_s117
                                 + 1.24171 w_k1_KO_s117 + 1.81764 w_k1_MAT_s117
                                 + 1.08955 w_k1_MSFT_s117
                                 + 1.01445 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.06182 w_k1_ADM_s117
                                 + 0.802707 w_k1_AXP_s117
                                 + 1.14455 w_k1_DIS_s117
                                 + 1.12295 w_k1_FDX_s117
                                 + 0.912392 w_k1_IBM_s117
                                 + 0.979941 w_k1_INTC_s117
                                 + 1.01907 w_k1_KO_s117
                                 + 0.747104 w_k1_MAT_s117
                                 + 1.18078 w_k1_MSFT_s117
                                 + 1.04613 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.10568 w_k1_ADM_s117
                                 + 1.23055 w_k1_AXP_s117
                                 + 1.72742 w_k1_DIS_s117
                                 + 1.49829 w_k1_FDX_s117
                                 + 1.09177 w_k1_IBM_s117
                                 + 1.38391 w_k1_INTC_s117
                                 + 1.38023 w_k1_KO_s117 + 1.36248 w_k1_MAT_s117
                                 + 1.45632 w_k1_MSFT_s117 + 1.1937 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.825902 w_k1_ADM_s117
                                 + 0.769492 w_k1_AXP_s117
                                 + 0.81744 w_k1_DIS_s117
                                 + 0.904302 w_k1_FDX_s117
                                 + 0.813173 w_k1_IBM_s117
                                 + 1.07734 w_k1_INTC_s117
                                 + 0.817397 w_k1_KO_s117
                                 + 0.606948 w_k1_MAT_s117
                                 + 0.912566 w_k1_MSFT_s117
                                 + 0.782258 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.833129 w_k1_ADM_s117
                                 + 0.704913 w_k1_AXP_s117
                                 + 0.603991 w_k1_DIS_s117
                                 + 0.744203 w_k1_FDX_s117
                                 + 0.86637 w_k1_IBM_s117
                                 + 0.874376 w_k1_INTC_s117
                                 + 0.761715 w_k1_KO_s117
                                 + 1.03482 w_k1_MAT_s117
                                 + 0.770414 w_k1_MSFT_s117
                                 + 0.782606 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0149720744226094 w_k1_ADM_s000
                                 + 0.0217236470788794 w_k1_ADM_s001
                                 + 0.013249101801277 w_k1_ADM_s002
                                 + 0.022831630790267 w_k1_ADM_s003
                                 + 0.0155953975539247 w_k1_ADM_s004
                                 + 0.00802172988372477 w_k1_ADM_s005
                                 + 0.0120438018192699 w_k1_ADM_s006
                                 + 0.0317341704279184 w_k1_ADM_s007
                                 + 0.0176843878165115 w_k1_ADM_s010
                                 + 0.0206544127017012 w_k1_ADM_s011
                                 + 0.0052344747719605 w_k1_ADM_s012
                                 + 0.0146675622653546 w_k1_ADM_s013
                                 + 0.0107320422039087 w_k1_ADM_s014
                                 + 0.00485370053117556 w_k1_ADM_s015
                                 + 0.0158500973474659 w_k1_ADM_s016
                                 + 0.00499431932225454 w_k1_ADM_s017
                                 + 0.0453245153505046 w_k1_ADM_s100
                                 + 0.036713652746332 w_k1_ADM_s101
                                 + 0.0373361693734831 w_k1_ADM_s102
                                 + 0.0194028190390943 w_k1_ADM_s103
                                 + 0.0111785136591012 w_k1_ADM_s104
                                 + 0.0103843773241336 w_k1_ADM_s105
                                 + 0.010413239798836 w_k1_ADM_s106
                                 + 0.0253782293351625 w_k1_ADM_s107
                                 + 0.0514160129331695 w_k1_ADM_s110
                                 + 0.0859358378566129 w_k1_ADM_s111
                                 + 0.051828869970665 w_k1_ADM_s112
                                 + 0.0866034705162086 w_k1_ADM_s113
                                 + 0.116696404520701 w_k1_ADM_s114
                                 + 0.102466054323048 w_k1_ADM_s115
                                 + 0.0564867065231121 w_k1_ADM_s116
                                 + 0.0437609432411312 w_k1_ADM_s117
                                 + 0.0148295635043078 w_k1_AXP_s000
                                 + 0.0219195405511163 w_k1_AXP_s001
                                 + 0.012837275372103 w_k1_AXP_s002
                                 + 0.0210239813766559 w_k1_AXP_s003
                                 + 0.014583222228185 w_k1_AXP_s004
                                 + 0.00894425263771308 w_k1_AXP_s005
                                 + 0.0137463179595088 w_k1_AXP_s006
                                 + 0.0259595039134249 w_k1_AXP_s007
                                 + 0.0180909165610378 w_k1_AXP_s010
                                 + 0.0253569240717864 w_k1_AXP_s011
                                 + 0.00552147567313047 w_k1_AXP_s012
                                 + 0.0162272683389286 w_k1_AXP_s013
                                 + 0.00990894003547955 w_k1_AXP_s014
                                 + 0.00473704832714037 w_k1_AXP_s015
                                 + 0.0164379146453382 w_k1_AXP_s016
                                 + 0.00474285123625159 w_k1_AXP_s017
                                 + 0.0437359973989544 w_k1_AXP_s100
                                 + 0.0322632747551341 w_k1_AXP_s101
                                 + 0.0333985951439003 w_k1_AXP_s102
                                 + 0.0180302440501152 w_k1_AXP_s103
                                 + 0.0119105378317306 w_k1_AXP_s104
                                 + 0.0154275765461454 w_k1_AXP_s105
                                 + 0.0125657519167557 w_k1_AXP_s106
                                 + 0.0231942219150634 w_k1_AXP_s107
                                 + 0.0548741216448386 w_k1_AXP_s110
                                 + 0.0945128603783884 w_k1_AXP_s111
                                 + 0.0458510830469128 w_k1_AXP_s112
                                 + 0.0890077474226451 w_k1_AXP_s113
                                 + 0.104107611882056 w_k1_AXP_s114
                                 + 0.103441183098442 w_k1_AXP_s115
                                 + 0.0534176919138271 w_k1_AXP_s116
                                 + 0.0434101462143869 w_k1_AXP_s117
                                 + 0.0153464374193876 w_k1_DIS_s000
                                 + 0.0222781245867205 w_k1_DIS_s001
                                 + 0.0134609739563807 w_k1_DIS_s002
                                 + 0.0209366159758917 w_k1_DIS_s003
                                 + 0.0143827588201202 w_k1_DIS_s004
                                 + 0.00859873686138959 w_k1_DIS_s005
                                 + 0.0122667651964632 w_k1_DIS_s006
                                 + 0.0301225696270967 w_k1_DIS_s007
                                 + 0.0190572788024888 w_k1_DIS_s010
                                 + 0.0225637308304704 w_k1_DIS_s011
                                 + 0.00531370507030147 w_k1_DIS_s012
                                 + 0.0162127023574149 w_k1_DIS_s013
                                 + 0.0117710256893549 w_k1_DIS_s014
                                 + 0.00508378450780141 w_k1_DIS_s015
                                 + 0.0158270348937854 w_k1_DIS_s016
                                 + 0.00525719471679699 w_k1_DIS_s017
                                 + 0.0456650381685539 w_k1_DIS_s100
                                 + 0.0323514739528063 w_k1_DIS_s101
                                 + 0.0362563099718918 w_k1_DIS_s102
                                 + 0.0193055879324413 w_k1_DIS_s103
                                 + 0.0112284899210798 w_k1_DIS_s104
                                 + 0.0162391931900862 w_k1_DIS_s105
                                 + 0.0129712059364784 w_k1_DIS_s106
                                 + 0.0223397851110978 w_k1_DIS_s107
                                 + 0.0612251759357522 w_k1_DIS_s110
                                 + 0.0917630201525391 w_k1_DIS_s111
                                 + 0.0440199321797082 w_k1_DIS_s112
                                 + 0.0899703347299251 w_k1_DIS_s113
                                 + 0.114289063355571 w_k1_DIS_s114
                                 + 0.105290764462772 w_k1_DIS_s115
                                 + 0.0538015584978999 w_k1_DIS_s116
                                 + 0.0454153729140923 w_k1_DIS_s117
                                 + 0.0143632222084406 w_k1_FDX_s000
                                 + 0.0214316206485358 w_k1_FDX_s001
                                 + 0.0128294909065661 w_k1_FDX_s002
                                 + 0.021349784703933 w_k1_FDX_s003
                                 + 0.013938059276935 w_k1_FDX_s004
                                 + 0.00860170974772792 w_k1_FDX_s005
                                 + 0.012413697984847 w_k1_FDX_s006
                                 + 0.0313089691086983 w_k1_FDX_s007
                                 + 0.0180230613854273 w_k1_FDX_s010
                                 + 0.0228200709914091 w_k1_FDX_s011
                                 + 0.00543856694233544 w_k1_FDX_s012
                                 + 0.0164322066395805 w_k1_FDX_s013
                                 + 0.0109860102623659 w_k1_FDX_s014
                                 + 0.00497606751161303 w_k1_FDX_s015
                                 + 0.0165810588053366 w_k1_FDX_s016
                                 + 0.00466817422264689 w_k1_FDX_s017
                                 + 0.0427555059880747 w_k1_FDX_s100
                                 + 0.0309343903183157 w_k1_FDX_s101
                                 + 0.0341671301146796 w_k1_FDX_s102
                                 + 0.0192947131598682 w_k1_FDX_s103
                                 + 0.0114615598359003 w_k1_FDX_s104
                                 + 0.0148314618164994 w_k1_FDX_s105
                                 + 0.0123519175469576 w_k1_FDX_s106
                                 + 0.0241407944226806 w_k1_FDX_s107
                                 + 0.055340046094354 w_k1_FDX_s110
                                 + 0.0944900883517162 w_k1_FDX_s111
                                 + 0.0416170732871341 w_k1_FDX_s112
                                 + 0.0914686232201814 w_k1_FDX_s113
                                 + 0.112009068418745 w_k1_FDX_s114
                                 + 0.101888103352143 w_k1_FDX_s115
                                 + 0.052177415143241 w_k1_FDX_s116
                                 + 0.0472412965705436 w_k1_FDX_s117
                                 + 0.0147827763711465 w_k1_IBM_s000
                                 + 0.0216411995854924 w_k1_IBM_s001
                                 + 0.0140057399345747 w_k1_IBM_s002
                                 + 0.0215953445183125 w_k1_IBM_s003
                                 + 0.0150638930506162 w_k1_IBM_s004
                                 + 0.00854875794451772 w_k1_IBM_s005
                                 + 0.0129628023329517 w_k1_IBM_s006
                                 + 0.0319874287532813 w_k1_IBM_s007
                                 + 0.0191026885967475 w_k1_IBM_s010
                                 + 0.0225685945864528 w_k1_IBM_s011
                                 + 0.00548651828459053 w_k1_IBM_s012
                                 + 0.0154558191533138 w_k1_IBM_s013
                                 + 0.01091074124407 w_k1_IBM_s014
                                 + 0.00512210739327428 w_k1_IBM_s015
                                 + 0.0165006128647342 w_k1_IBM_s016
                                 + 0.0051065548850579 w_k1_IBM_s017
                                 + 0.0439606927083376 w_k1_IBM_s100
                                 + 0.0296122025814867 w_k1_IBM_s101
                                 + 0.0339162422164999 w_k1_IBM_s102
                                 + 0.0188608872064035 w_k1_IBM_s103
                                 + 0.0112983889412502 w_k1_IBM_s104
                                 + 0.0150009442467945 w_k1_IBM_s105
                                 + 0.0126828491185373 w_k1_IBM_s106
                                 + 0.0240459881106348 w_k1_IBM_s107
                                 + 0.0563598617501474 w_k1_IBM_s110
                                 + 0.0938788874308621 w_k1_IBM_s111
                                 + 0.0453218293334284 w_k1_IBM_s112
                                 + 0.0905423643965984 w_k1_IBM_s113
                                 + 0.111838987301942 w_k1_IBM_s114
                                 + 0.0970358765646251 w_k1_IBM_s115
                                 + 0.0509596355211064 w_k1_IBM_s116
                                 + 0.0442599985657698 w_k1_IBM_s117
                                 + 0.0137389474747895 w_k1_INTC_s000
                                 + 0.0252283393798983 w_k1_INTC_s001
                                 + 0.0133192873168594 w_k1_INTC_s002
                                 + 0.021769689488431 w_k1_INTC_s003
                                 + 0.0128698769197572 w_k1_INTC_s004
                                 + 0.00948460417509993 w_k1_INTC_s005
                                 + 0.0132086422120927 w_k1_INTC_s006
                                 + 0.032079132005083 w_k1_INTC_s007
                                 + 0.0211457473789835 w_k1_INTC_s010
                                 + 0.0242875771155185 w_k1_INTC_s011
                                 + 0.00566034425565858 w_k1_INTC_s012
                                 + 0.0150599117464572 w_k1_INTC_s013
                                 + 0.0115053131593572 w_k1_INTC_s014
                                 + 0.00441026246943669 w_k1_INTC_s015
                                 + 0.0161673339775986 w_k1_INTC_s016
                                 + 0.00508509932247782 w_k1_INTC_s017
                                 + 0.0427844546510562 w_k1_INTC_s100
                                 + 0.0337813524213244 w_k1_INTC_s101
                                 + 0.035579600155173 w_k1_INTC_s102
                                 + 0.0158803439690314 w_k1_INTC_s103
                                 + 0.010655716669756 w_k1_INTC_s104
                                 + 0.0161000599434945 w_k1_INTC_s105
                                 + 0.0124943686197449 w_k1_INTC_s106
                                 + 0.0230642754139521 w_k1_INTC_s107
                                 + 0.0564088944442015 w_k1_INTC_s110
                                 + 0.0828256021511348 w_k1_INTC_s111
                                 + 0.0359023155943394 w_k1_INTC_s112
                                 + 0.106307464557412 w_k1_INTC_s113
                                 + 0.113693347560579 w_k1_INTC_s114
                                 + 0.10073139321483 w_k1_INTC_s115
                                 + 0.056188155106703 w_k1_INTC_s116
                                 + 0.0479057355158403 w_k1_INTC_s117
                                 + 0.0138187876579156 w_k1_KO_s000
                                 + 0.0250850410822351 w_k1_KO_s001
                                 + 0.0137604177228964 w_k1_KO_s002
                                 + 0.0200708370282284 w_k1_KO_s003
                                 + 0.0144278360278934 w_k1_KO_s004
                                 + 0.00906365629036225 w_k1_KO_s005
                                 + 0.0126988878375225 w_k1_KO_s006
                                 + 0.0326753257605173 w_k1_KO_s007
                                 + 0.0191755187389987 w_k1_KO_s010
                                 + 0.0216705073907878 w_k1_KO_s011
                                 + 0.00559831960111252 w_k1_KO_s012
                                 + 0.0145363893314807 w_k1_KO_s013
                                 + 0.0122652200643109 w_k1_KO_s014
                                 + 0.00509499217979468 w_k1_KO_s015
                                 + 0.0157172104131346 w_k1_KO_s016
                                 + 0.00496849055675336 w_k1_KO_s017
                                 + 0.0410869378937718 w_k1_KO_s100
                                 + 0.0309499688114518 w_k1_KO_s101
                                 + 0.038345855589365 w_k1_KO_s102
                                 + 0.0185350796561598 w_k1_KO_s103
                                 + 0.0114403415851907 w_k1_KO_s104
                                 + 0.0137063286075874 w_k1_KO_s105
                                 + 0.0122696317732536 w_k1_KO_s106
                                 + 0.0242145425562419 w_k1_KO_s107
                                 + 0.0552164703105384 w_k1_KO_s110
                                 + 0.0946310361278123 w_k1_KO_s111
                                 + 0.0436971348800616 w_k1_KO_s112
                                 + 0.0911924651820426 w_k1_KO_s113
                                 + 0.113645538275491 w_k1_KO_s114
                                 + 0.10541374459316 w_k1_KO_s115
                                 + 0.0493953595180872 w_k1_KO_s116
                                 + 0.045087585866312 w_k1_KO_s117
                                 + 0.0170959394679842 w_k1_MAT_s000
                                 + 0.0237373813768712 w_k1_MAT_s001
                                 + 0.0145037272959753 w_k1_MAT_s002
                                 + 0.0190407216942664 w_k1_MAT_s003
                                 + 0.0118087631567393 w_k1_MAT_s004
                                 + 0.00811224538630099 w_k1_MAT_s005
                                 + 0.0145542500396286 w_k1_MAT_s006
                                 + 0.0291509252789135 w_k1_MAT_s007
                                 + 0.0205489337581016 w_k1_MAT_s010
                                 + 0.0220023441274429 w_k1_MAT_s011
                                 + 0.00578469195475911 w_k1_MAT_s012
                                 + 0.0183770410275703 w_k1_MAT_s013
                                 + 0.0120893825034104 w_k1_MAT_s014
                                 + 0.00608183584384637 w_k1_MAT_s015
                                 + 0.0173625725194584 w_k1_MAT_s016
                                 + 0.00395692051425148 w_k1_MAT_s017
                                 + 0.0437269206628446 w_k1_MAT_s100
                                 + 0.0310409408984258 w_k1_MAT_s101
                                 + 0.0332749311786349 w_k1_MAT_s102
                                 + 0.0219064524074419 w_k1_MAT_s103
                                 + 0.0130200780374621 w_k1_MAT_s104
                                 + 0.0136523060636706 w_k1_MAT_s105
                                 + 0.0151057054910843 w_k1_MAT_s106
                                 + 0.0215703051811906 w_k1_MAT_s107
                                 + 0.0602341646825425 w_k1_MAT_s110
                                 + 0.0941991008477896 w_k1_MAT_s111
                                 + 0.0467534058477578 w_k1_MAT_s112
                                 + 0.076791596491262 w_k1_MAT_s113
                                 + 0.121368250752723 w_k1_MAT_s114
                                 + 0.101888384692871 w_k1_MAT_s115
                                 + 0.0428981114914579 w_k1_MAT_s116
                                 + 0.0479020439937784 w_k1_MAT_s117
                                 + 0.0139942991860629 w_k1_MSFT_s000
                                 + 0.0245718271412822 w_k1_MSFT_s001
                                 + 0.012141202772259 w_k1_MSFT_s002
                                 + 0.0197850529618305 w_k1_MSFT_s003
                                 + 0.0140009586865338 w_k1_MSFT_s004
                                 + 0.00854115634072975 w_k1_MSFT_s005
                                 + 0.0135224388707285 w_k1_MSFT_s006
                                 + 0.0301209468422111 w_k1_MSFT_s007
                                 + 0.0192962867797684 w_k1_MSFT_s010
                                 + 0.0223520175022628 w_k1_MSFT_s011
                                 + 0.00579142911152811 w_k1_MSFT_s012
                                 + 0.016468204552744 w_k1_MSFT_s013
                                 + 0.0122709023105838 w_k1_MSFT_s014
                                 + 0.00442950643261849 w_k1_MSFT_s015
                                 + 0.0166893939490827 w_k1_MSFT_s016
                                 + 0.00470898745400131 w_k1_MSFT_s017
                                 + 0.0412762893031735 w_k1_MSFT_s100
                                 + 0.0311571196425094 w_k1_MSFT_s101
                                 + 0.0338811874019508 w_k1_MSFT_s102
                                 + 0.016754382171603 w_k1_MSFT_s103
                                 + 0.0108728038633633 w_k1_MSFT_s104
                                 + 0.015989986363682 w_k1_MSFT_s105
                                 + 0.0113776997960654 w_k1_MSFT_s106
                                 + 0.0213731703858528 w_k1_MSFT_s107
                                 + 0.0544475803509357 w_k1_MSFT_s110
                                 + 0.0842850405639999 w_k1_MSFT_s111
                                 + 0.0415813478406039 w_k1_MSFT_s112
                                 + 0.0951204908149501 w_k1_MSFT_s113
                                 + 0.108915781359855 w_k1_MSFT_s114
                                 + 0.0955891049874389 w_k1_MSFT_s115
                                 + 0.0489767227180212 w_k1_MSFT_s116
                                 + 0.0474582009508702 w_k1_MSFT_s117
                                 + 0.014501845107687 w_k1_PG_s000
                                 + 0.0233636294306268 w_k1_PG_s001
                                 + 0.0140508355257521 w_k1_PG_s002
                                 + 0.0213954606585414 w_k1_PG_s003
                                 + 0.0141843171101973 w_k1_PG_s004
                                 + 0.00831266217752407 w_k1_PG_s005
                                 + 0.0120623724703337 w_k1_PG_s006
                                 + 0.0315451656783864 w_k1_PG_s007
                                 + 0.0175871645129551 w_k1_PG_s010
                                 + 0.0213586958976216 w_k1_PG_s011
                                 + 0.00551258113519028 w_k1_PG_s012
                                 + 0.0153594658734357 w_k1_PG_s013
                                 + 0.0108341085906409 w_k1_PG_s014
                                 + 0.00498102191747492 w_k1_PG_s015
                                 + 0.0169658040143488 w_k1_PG_s016
                                 + 0.00512845933059802 w_k1_PG_s017
                                 + 0.0411212169433782 w_k1_PG_s100
                                 + 0.0309765331734411 w_k1_PG_s101
                                 + 0.0344337654380625 w_k1_PG_s102
                                 + 0.0181466732630561 w_k1_PG_s103
                                 + 0.0115033936951441 w_k1_PG_s104
                                 + 0.0139184570887904 w_k1_PG_s105
                                 + 0.0120082164459762 w_k1_PG_s106
                                 + 0.0239276559595306 w_k1_PG_s107
                                 + 0.0575600647058989 w_k1_PG_s110
                                 + 0.0910898200541496 w_k1_PG_s111
                                 + 0.0406422116762981 w_k1_PG_s112
                                 + 0.0941658089620966 w_k1_PG_s113
                                 + 0.111123473282393 w_k1_PG_s114
                                 + 0.0986092699620295 w_k1_PG_s115
                                 + 0.0506372434309136 w_k1_PG_s116
                                 + 0.0421302803471755 w_k1_PG_s117 >= 1.07
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
