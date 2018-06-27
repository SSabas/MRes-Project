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
 w_k1_ADM_s0_return_constraint:  1.19142 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.970029 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.02903 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.761341 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.839751 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.962693 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.967238 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.17702 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.24482 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.936159 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.860616 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.786799 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.746962 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.892992 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.993911 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.27608 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.41746 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.931654 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.656689 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.865138 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.15151 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.802144 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.05989 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.840847 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.44334 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.775647 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.907418 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.27489 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.50801 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.02266 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.03362 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.762715 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0274 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.01775 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.07274 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.702 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.09503 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.27271 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.17269 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.845311 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.11492 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.725807 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.861015 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.12369 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.870073 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.5743 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.30849 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.968906 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.813663 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.849145 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.629487 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.25646 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.85018 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.20204 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.33225 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.01486 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.663696 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.16868 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.807523 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.798809 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.2592 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.940348 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.53125 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.674606 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.30929 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.39091 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.92449 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.04601 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.8828 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.614644 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.663615 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.1266 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.04497 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.71784 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.94715 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.21509 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.21985 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.827574 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.10945 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.627575 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.798165 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.18939 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.765802 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.86368 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.34718 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.907825 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.730316 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.01402 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.634479 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.15592 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.879785 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.13137 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.49509 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.914232 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.614865 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.25189 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.21193 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.931848 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.45205 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.895552 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.6848 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.61142 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.08414 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.47352 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 2.06298 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.881224 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.08786 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.626921 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.708016 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.1771 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.882486 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.70218 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.786225 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.26822 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14368 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.877415 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.09637 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.736192 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.837992 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.11759 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.961604 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.32853 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.27663 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.820548 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.797661 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.943781 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.745625 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.927644 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.944042 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.22759 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.2966 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.909877 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.733884 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.06745 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.12345 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.823688 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.17101 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.91257 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.24987 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.715925 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.990907 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.29151 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.34039 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.01398 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.939383 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.708565 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.950233 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.03191 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.963984 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.35229 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.954524 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.23061 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.0594 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.998954 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.0855 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.874396 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.10537 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.977297 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.551575 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.73967 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.950303 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.28633 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.01682 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.904086 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.660824 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.89418 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.666204 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.800844 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.21924 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.07071 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.665292 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.71691 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.700129 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.10623 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.18387 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.10257 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.57612 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.610142 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.4166 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.993347 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 2.14755 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.711841 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.970825 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.699394 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.471013 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.70884 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.02229 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.50686 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.738028 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.923311 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10103 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.876826 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.11469 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.855446 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.904261 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.999692 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.951662 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.24081 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.08946 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.915791 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.800434 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.00338 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.726703 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09717 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.870408 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.00446 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01183 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.912706 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.898605 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.01878 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.10206 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.969621 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0483 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.868674 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.23304 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.745543 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.19046 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.30097 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.2368 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.06689 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.11187 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.867379 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.967901 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.06979 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.21843 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.38067 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.887001 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.36447 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.09121 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.937967 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.07915 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.7474 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.946499 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.00179 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.678358 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.67464 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.21293 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.04343 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.867725 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.947519 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.722177 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.57483 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.892996 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.0447 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.40456 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.10914 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.777555 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.3482 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.16777 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11675 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.29748 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.940661 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.66504 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.635014 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.23124 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.25161 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.80348 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.886712 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.12822 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.745839 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.64034 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.28965 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.10278 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.47064 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.761485 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.968147 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.16668 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.04887 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.04906 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.766481 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.99741 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.944222 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.533146 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 2.12241 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.21441 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.30951 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.01813 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.880962 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.567019 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.96949 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.719468 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.921699 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.51583 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.14213 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.57786 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 2.00553 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.575134 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.01214 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.24598 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.05706 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.7055 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.499148 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.54987 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.03589 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.57427 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.717424 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.891867 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.556963 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.464804 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.80972 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.960708 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 2.05713 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.650287 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.885723 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11674 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.905985 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.09522 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.79484 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.840345 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03985 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.837997 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.27996 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.16923 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.845181 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.78552 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.966023 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.705941 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.07328 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.963071 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.16575 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.18226 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.908418 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.74244 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.936341 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.11849 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.925085 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.13377 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.930084 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.32355 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.785408 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.0406 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.21547 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.29406 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.986222 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.1386 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.806478 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.948633 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.09655 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.0586 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.43601 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.936885 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.23226 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.0943 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.935693 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.06789 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.844047 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.964137 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06561 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.820882 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.15921 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.06991 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.9585 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.851547 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.954052 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.815726 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.17725 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.878416 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.15241 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.14631 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.988154 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.748407 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.14162 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.985879 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.893038 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.01067 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.918959 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.25434 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.754191 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.078 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.11461 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.30954 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.863037 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.992994 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.774005 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.815964 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.0269 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.942639 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.15681 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.877025 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.09655 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.619998362989118 z_k1_s0000
                                 - 0.196999458015971 z_k1_s0001
                                 - 0.947997584305647 z_k1_s0002
                                 - 0.344998732411276 z_k1_s0003
                                 - 0.561000112871877 z_k1_s0004
                                 - 0.644995785722951 z_k1_s0005
                                 - 0.379998734626472 z_k1_s0006
                                 - 0.816999174900525 z_k1_s0007
                                 - 0.31199985253272 z_k1_s0010
                                 - 0.38199937096471 z_k1_s0011
                                 - 0.279999905226776 z_k1_s0012
                                 - 0.67099889507152 z_k1_s0013
                                 - 0.948998437291911 z_k1_s0014
                                 - 0.332999817952246 z_k1_s0015
                                 - 0.230999985914549 z_k1_s0016
                                 - 0.504999168421933 z_k1_s0017
                                 - 1.81899446347397 z_k1_s0020
                                 - 1.45000400117469 z_k1_s0021
                                 - 2.11700028428103 z_k1_s0022
                                 - 0.700999674657232 z_k1_s0023
                                 - 0.870999265430384 z_k1_s0024
                                 - 0.868998589186728 z_k1_s0025
                                 - 2.7519950098789 z_k1_s0026
                                 - 0.998998548671785 z_k1_s0027
                                 - 0.293000610991025 z_k1_s0030
                                 - 0.922000270256793 z_k1_s0031
                                 - 1.03700444738658 z_k1_s0032
                                 - 0.757002088482602 z_k1_s0033
                                 - 0.356000905343794 z_k1_s0034
                                 - 0.277000543739548 z_k1_s0035
                                 - 0.654002335203022 z_k1_s0036
                                 - 0.432001342988623 z_k1_s0037
                                 - 0.670999573349125 z_k1_s0040
                                 - 0.604998736102239 z_k1_s0041
                                 - 0.668998430645397 z_k1_s0042
                                 - 0.175999997993142 z_k1_s0043
                                 - 0.183999876011091 z_k1_s0044
                                 - 0.397000566836433 z_k1_s0045
                                 - 0.395998319485655 z_k1_s0046
                                 - 0.253999981867385 z_k1_s0047
                                 - 0.844998837028345 z_k1_s0050
                                 - 0.392000853685612 z_k1_s0051
                                 - 1.0880039042113 z_k1_s0052
                                 - 0.592998081623006 z_k1_s0053
                                 - 1.08299987185608 z_k1_s0054
                                 - 0.446000437057334 z_k1_s0055
                                 - 0.577000797609269 z_k1_s0056
                                 - 0.761000513562733 z_k1_s0057
                                 - 1.2589960067036 z_k1_s0060
                                 - 0.734998022003928 z_k1_s0061
                                 - 1.21699474590708 z_k1_s0062
                                 - 0.772997903385128 z_k1_s0063
                                 - 1.84199690737625 z_k1_s0064
                                 - 0.801998888420513 z_k1_s0065
                                 - 0.770997614376387 z_k1_s0066
                                 - 0.614998312231947 z_k1_s0067
                                 - 0.279999979859418 z_k1_s0070
                                 - 0.350999022134428 z_k1_s0071
                                 - 0.471000043359955 z_k1_s0072
                                 - 0.109999974923342 z_k1_s0073
                                 - 0.423999763529964 z_k1_s0074
                                 - 0.215999915805833 z_k1_s0075
                                 - 0.219999949846684 z_k1_s0076
                                 - 0.330998851930173 z_k1_s0077
                                 - 0.280998952450822 z_k1_s0100
                                 - 0.430999649806879 z_k1_s0101
                                 - 0.140999824716038 z_k1_s0102
                                 - 0.247999629831145 z_k1_s0103
                                 - 0.143999838663159 z_k1_s0104
                                 - 0.122999741033311 z_k1_s0105
                                 - 0.453998233603938 z_k1_s0106
                                 - 0.254999108508657 z_k1_s0107
                                 - 0.0859999862500785 z_k1_s0110
                                 - 0.06400004394426 z_k1_s0111
                                 - 0.077000022070429 z_k1_s0112
                                 - 0.0740000340105458 z_k1_s0113
                                 - 0.0870000121367148 z_k1_s0114
                                 - 0.248000327084053 z_k1_s0115
                                 - 0.0870000121367148 z_k1_s0116
                                 - 0.172999729586716 z_k1_s0117
                                 - 0.305000770232724 z_k1_s0120
                                 - 0.272999946050554 z_k1_s0121
                                 - 0.118000084470257 z_k1_s0122
                                 - 0.338000162919849 z_k1_s0123
                                 - 0.260999435982139 z_k1_s0124
                                 - 0.0970000762509776 z_k1_s0125
                                 - 0.118000084470257 z_k1_s0126
                                 - 0.0980000919566654 z_k1_s0127
                                 - 0.151000204707101 z_k1_s0130
                                 - 0.0720000608099987 z_k1_s0131
                                 - 0.138999709471971 z_k1_s0132
                                 - 0.0450000049312403 z_k1_s0133
                                 - 0.114999601001948 z_k1_s0134
                                 - 0.198000233377514 z_k1_s0135
                                 - 0.0640000540533322 z_k1_s0136
                                 - 0.0979999284191233 z_k1_s0137
                                 - 0.0500000651224443 z_k1_s0140
                                 - 0.0579999989979945 z_k1_s0141
                                 - 0.174000412994206 z_k1_s0142
                                 - 0.0899999841003287 z_k1_s0143
                                 - 0.171000219390758 z_k1_s0144
                                 - 0.0979996683757454 z_k1_s0145
                                 - 0.0570000176635563 z_k1_s0146
                                 - 0.134000327616233 z_k1_s0147
                                 - 0.314000148013252 z_k1_s0150
                                 - 0.225000092531758 z_k1_s0151
                                 - 0.23499990784384 z_k1_s0152
                                 - 0.108000270975772 z_k1_s0153
                                 - 0.143000757370701 z_k1_s0154
                                 - 0.204000763577045 z_k1_s0155
                                 - 0.107000289444564 z_k1_s0156
                                 - 0.0800002217006206 z_k1_s0157
                                 - 0.244999793692926 z_k1_s0160
                                 - 0.0560000579870032 z_k1_s0161
                                 - 0.0650000541634798 z_k1_s0162
                                 - 0.103999608261346 z_k1_s0163
                                 - 0.21400037930755 z_k1_s0164
                                 - 0.086000075908606 z_k1_s0165
                                 - 0.0930000831569814 z_k1_s0166
                                 - 0.0570000064510325 z_k1_s0167
                                 - 0.0759999334746765 z_k1_s0170
                                 - 0.139999460190494 z_k1_s0171
                                 - 0.0589999353158766 z_k1_s0172
                                 - 0.245999716862817 z_k1_s0173
                                 - 0.121000021871432 z_k1_s0174
                                 - 0.0689999109151875 z_k1_s0175
                                 - 0.148999715709674 z_k1_s0176
                                 - 0.1310001956706 z_k1_s0177
                                 - 0.109999870847817 z_k1_s1000
                                 - 0.154999760453833 z_k1_s1001
                                 - 0.2789998220765 z_k1_s1002
                                 - 0.368999239489101 z_k1_s1003
                                 - 0.23099989159016 z_k1_s1004
                                 - 0.123999880722952 z_k1_s1005
                                 - 0.382999068464521 z_k1_s1006
                                 - 0.157999801334158 z_k1_s1007
                                 - 0.178000214434685 z_k1_s1010
                                 - 0.486000947448168 z_k1_s1011
                                 - 0.224000485864266 z_k1_s1012
                                 - 0.707000218269203 z_k1_s1013
                                 - 0.194000286253629 z_k1_s1014
                                 - 0.384000684452757 z_k1_s1015
                                 - 0.169000206511589 z_k1_s1016
                                 - 0.256000369701681 z_k1_s1017
                                 - 0.344000912320119 z_k1_s1020
                                 - 0.217001040868225 z_k1_s1021
                                 - 0.215000768325326 z_k1_s1022
                                 - 0.100000154627801 z_k1_s1023
                                 - 0.284000661043237 z_k1_s1024
                                 - 0.133000213579985 z_k1_s1025
                                 - 0.147000219377857 z_k1_s1026
                                 - 0.14500010533516 z_k1_s1027
                                 - 0.918998742873556 z_k1_s1030
                                 - 0.220000023755477 z_k1_s1031
                                 - 0.427999077895522 z_k1_s1032
                                 - 1.0649989223985 z_k1_s1033
                                 - 0.213999930547631 z_k1_s1034
                                 - 0.421999767887416 z_k1_s1035
                                 - 0.367999712216543 z_k1_s1036
                                 - 0.279999781034326 z_k1_s1037
                                 - 0.082000059975178 z_k1_s1040
                                 - 0.130999876855309 z_k1_s1041
                                 - 0.0650000618657003 z_k1_s1042
                                 - 0.230000270540208 z_k1_s1043
                                 - 0.0810000551510875 z_k1_s1044
                                 - 0.0690000811620623 z_k1_s1045
                                 - 0.0700000859861528 z_k1_s1046
                                 - 0.110999780373499 z_k1_s1047
                                 - 0.229000097683548 z_k1_s1050
                                 - 0.0610001123647225 z_k1_s1051
                                 - 0.101000182164536 z_k1_s1052
                                 - 0.168000340519275 z_k1_s1053
                                 - 0.222000280828828 z_k1_s1054
                                 - 0.183999752758421 z_k1_s1055
                                 - 0.0850000595244905 z_k1_s1056
                                 - 0.134000553509778 z_k1_s1057
                                 - 0.0830001868921327 z_k1_s1060
                                 - 0.139000170735991 z_k1_s1061
                                 - 0.19500003767966 z_k1_s1062
                                 - 0.19500003767966 z_k1_s1063
                                 - 0.0780001319720538 z_k1_s1064
                                 - 0.0950001082799804 z_k1_s1065
                                 - 0.308999933815257 z_k1_s1066
                                 - 0.0750001224000444 z_k1_s1067
                                 - 0.294000193284451 z_k1_s1070
                                 - 0.273000446907305 z_k1_s1071
                                 - 0.286000379045643 z_k1_s1072
                                 - 0.429998907346383 z_k1_s1073
                                 - 0.501000207119277 z_k1_s1074
                                 - 0.475001840444368 z_k1_s1075
                                 - 0.614001748447344 z_k1_s1076
                                 - 0.87100144367571 z_k1_s1077
                                 - 0.779003494855415 z_k1_s1100
                                 - 0.415000933380137 z_k1_s1101
                                 - 0.491999728533851 z_k1_s1102
                                 - 0.391002095308286 z_k1_s1103
                                 - 0.418001629141294 z_k1_s1104
                                 - 0.383002482617669 z_k1_s1105
                                 - 0.219000666834788 z_k1_s1106
                                 - 0.267000697784581 z_k1_s1107
                                 - 0.81199947870903 z_k1_s1110
                                 - 1.18199762892288 z_k1_s1111
                                 - 0.940000233665976 z_k1_s1112
                                 - 1.33599883226096 z_k1_s1113
                                 - 0.28899898043623 z_k1_s1114
                                 - 0.649998621560147 z_k1_s1115
                                 - 0.679001300364376 z_k1_s1116
                                 - 0.405999109956103 z_k1_s1117
                                 - 0.0930000776043517 z_k1_s1120
                                 - 0.173999980137176 z_k1_s1121
                                 - 0.370000601038816 z_k1_s1122
                                 - 0.18100040206223 z_k1_s1123
                                 - 0.24899947533424 z_k1_s1124
                                 - 0.104999932875883 z_k1_s1125
                                 - 0.347000585285051 z_k1_s1126
                                 - 0.0800000478435286 z_k1_s1127
                                 - 0.356000456382291 z_k1_s1130
                                 - 0.681000754843247 z_k1_s1131
                                 - 0.328001094114075 z_k1_s1132
                                 - 0.257000845520439 z_k1_s1133
                                 - 0.508000099734643 z_k1_s1134
                                 - 0.293000927125511 z_k1_s1135
                                 - 0.76500129585628 z_k1_s1136
                                 - 0.318001246746598 z_k1_s1137
                                 - 0.280999048831146 z_k1_s1140
                                 - 0.650999008403926 z_k1_s1141
                                 - 0.680997907108203 z_k1_s1142
                                 - 0.821998349598956 z_k1_s1143
                                 - 0.719999500813345 z_k1_s1144
                                 - 0.225999195013088 z_k1_s1145
                                 - 0.613998493808434 z_k1_s1146
                                 - 0.326998800839409 z_k1_s1147
                                 - 0.336999337692684 z_k1_s1150
                                 - 0.244999328057172 z_k1_s1151
                                 - 0.906997666630044 z_k1_s1152
                                 - 0.463997046256122 z_k1_s1153
                                 - 0.317999379854805 z_k1_s1154
                                 - 0.519999701245076 z_k1_s1155
                                 - 0.619997769027629 z_k1_s1156
                                 - 0.650998876075985 z_k1_s1157
                                 - 0.446000684052351 z_k1_s1160
                                 - 0.188999624223183 z_k1_s1161
                                 - 0.481999148058963 z_k1_s1162
                                 - 0.590000454069929 z_k1_s1163
                                 - 0.447000148553273 z_k1_s1164
                                 - 0.303000378535695 z_k1_s1165
                                 - 0.213999543226296 z_k1_s1166
                                 - 0.285999428235086 z_k1_s1167
                                 - 0.368002121364959 z_k1_s1170
                                 - 0.188000559883015 z_k1_s1171
                                 - 0.25800062031357 z_k1_s1172
                                 - 0.420001720186586 z_k1_s1173
                                 - 0.481002228614306 z_k1_s1174
                                 - 0.802000731150161 z_k1_s1175
                                 - 0.544001807840665 z_k1_s1176
                                 - 0.333000733546423 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.02196 w_k1_ADM_s000
                                 + 1.47346 w_k1_AXP_s000
                                 + 1.26417 w_k1_DIS_s000
                                 + 1.14716 w_k1_FDX_s000
                                 + 1.44894 w_k1_IBM_s000
                                 + 1.07105 w_k1_INTC_s000
                                 + 1.31898 w_k1_KO_s000 + 1.65442 w_k1_MAT_s000
                                 + 1.09177 w_k1_MSFT_s000
                                 + 1.21627 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.887552 w_k1_ADM_s000
                                 + 0.727291 w_k1_AXP_s000
                                 + 0.798501 w_k1_DIS_s000
                                 + 0.963184 w_k1_FDX_s000
                                 + 0.500568 w_k1_IBM_s000
                                 + 0.824172 w_k1_INTC_s000
                                 + 0.778397 w_k1_KO_s000
                                 + 0.681165 w_k1_MAT_s000
                                 + 1.01731 w_k1_MSFT_s000
                                 + 0.900987 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.847584 w_k1_ADM_s000
                                 + 0.936418 w_k1_AXP_s000
                                 + 0.892398 w_k1_DIS_s000
                                 + 0.870106 w_k1_FDX_s000
                                 + 1.27741 w_k1_IBM_s000
                                 + 0.728069 w_k1_INTC_s000
                                 + 1.02241 w_k1_KO_s000 + 1.48887 w_k1_MAT_s000
                                 + 0.932786 w_k1_MSFT_s000
                                 + 1.04528 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.14958 w_k1_ADM_s000
                                 + 1.36423 w_k1_AXP_s000
                                 + 1.14673 w_k1_DIS_s000
                                 + 1.05605 w_k1_FDX_s000
                                 + 1.82473 w_k1_IBM_s000
                                 + 0.989656 w_k1_INTC_s000
                                 + 1.38353 w_k1_KO_s000 + 1.99129 w_k1_MAT_s000
                                 + 1.0665 w_k1_MSFT_s000 + 1.16725 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.786615 w_k1_ADM_s000
                                 + 0.98988 w_k1_AXP_s000
                                 + 0.793524 w_k1_DIS_s000
                                 + 0.891927 w_k1_FDX_s000
                                 + 0.911492 w_k1_IBM_s000
                                 + 0.790588 w_k1_INTC_s000
                                 + 0.825107 w_k1_KO_s000
                                 + 1.01684 w_k1_MAT_s000
                                 + 0.883537 w_k1_MSFT_s000
                                 + 0.845514 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.733297 w_k1_ADM_s000
                                 + 0.73977 w_k1_AXP_s000
                                 + 0.829017 w_k1_DIS_s000
                                 + 0.884972 w_k1_FDX_s000
                                 + 0.733711 w_k1_IBM_s000
                                 + 0.962609 w_k1_INTC_s000
                                 + 0.839927 w_k1_KO_s000
                                 + 0.678902 w_k1_MAT_s000
                                 + 1.0274 w_k1_MSFT_s000
                                 + 0.880347 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.17736 w_k1_ADM_s000
                                 + 1.45601 w_k1_AXP_s000
                                 + 1.51965 w_k1_DIS_s000 + 1.3704 w_k1_FDX_s000
                                 + 1.11137 w_k1_IBM_s000
                                 + 1.2623 w_k1_INTC_s000 + 1.21101 w_k1_KO_s000
                                 + 1.43517 w_k1_MAT_s000
                                 + 1.3822 w_k1_MSFT_s000 + 1.28219 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.972382 w_k1_ADM_s000
                                 + 0.963702 w_k1_AXP_s000
                                 + 1.29149 w_k1_DIS_s000
                                 + 1.12622 w_k1_FDX_s000
                                 + 0.973953 w_k1_IBM_s000
                                 + 0.960559 w_k1_INTC_s000
                                 + 1.16616 w_k1_KO_s000 + 1.25396 w_k1_MAT_s000
                                 + 1.09797 w_k1_MSFT_s000 + 1.1785 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.67121 w_k1_ADM_s001
                                 + 1.31672 w_k1_AXP_s001
                                 + 1.83338 w_k1_DIS_s001
                                 + 1.29496 w_k1_FDX_s001
                                 + 1.41251 w_k1_IBM_s001
                                 + 1.03063 w_k1_INTC_s001
                                 + 1.39687 w_k1_KO_s001 + 1.21497 w_k1_MAT_s001
                                 + 1.17092 w_k1_MSFT_s001
                                 + 1.25555 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.17682 w_k1_ADM_s001
                                 + 1.13455 w_k1_AXP_s001
                                 + 1.24395 w_k1_DIS_s001
                                 + 1.15116 w_k1_FDX_s001
                                 + 1.03328 w_k1_IBM_s001
                                 + 1.33922 w_k1_INTC_s001
                                 + 0.946627 w_k1_KO_s001
                                 + 0.876014 w_k1_MAT_s001
                                 + 1.18642 w_k1_MSFT_s001
                                 + 1.18848 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.0135 w_k1_ADM_s001
                                 + 1.37869 w_k1_AXP_s001
                                 + 1.22967 w_k1_DIS_s001
                                 + 1.06062 w_k1_FDX_s001
                                 + 1.67594 w_k1_IBM_s001
                                 + 1.16434 w_k1_INTC_s001
                                 + 1.43884 w_k1_KO_s001 + 1.82864 w_k1_MAT_s001
                                 + 1.06584 w_k1_MSFT_s001 + 1.2792 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.932817 w_k1_ADM_s001
                                 + 0.870268 w_k1_AXP_s001
                                 + 0.946738 w_k1_DIS_s001
                                 + 1.04949 w_k1_FDX_s001
                                 + 0.732265 w_k1_IBM_s001
                                 + 0.975067 w_k1_INTC_s001
                                 + 0.889376 w_k1_KO_s001
                                 + 0.753031 w_k1_MAT_s001
                                 + 1.1487 w_k1_MSFT_s001 + 1.0832 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.825992 w_k1_ADM_s001
                                 + 0.716594 w_k1_AXP_s001
                                 + 0.77605 w_k1_DIS_s001
                                 + 0.785799 w_k1_FDX_s001
                                 + 0.847611 w_k1_IBM_s001
                                 + 0.768996 w_k1_INTC_s001
                                 + 0.849934 w_k1_KO_s001
                                 + 0.73821 w_k1_MAT_s001
                                 + 0.878096 w_k1_MSFT_s001
                                 + 0.913336 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.979165 w_k1_ADM_s001
                                 + 0.685577 w_k1_AXP_s001
                                 + 0.655542 w_k1_DIS_s001
                                 + 0.708041 w_k1_FDX_s001
                                 + 1.1292 w_k1_IBM_s001
                                 + 0.720646 w_k1_INTC_s001
                                 + 0.930306 w_k1_KO_s001
                                 + 0.927656 w_k1_MAT_s001
                                 + 0.835072 w_k1_MSFT_s001
                                 + 1.00628 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.7365 w_k1_ADM_s001
                                 + 1.69547 w_k1_AXP_s001
                                 + 1.93937 w_k1_DIS_s001 + 1.6104 w_k1_FDX_s001
                                 + 0.908477 w_k1_IBM_s001
                                 + 1.20049 w_k1_INTC_s001
                                 + 1.24465 w_k1_KO_s001
                                 + 0.871513 w_k1_MAT_s001
                                 + 1.49807 w_k1_MSFT_s001
                                 + 1.51191 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.2291 w_k1_ADM_s001
                                 + 0.990919 w_k1_AXP_s001
                                 + 1.00443 w_k1_DIS_s001
                                 + 0.965247 w_k1_FDX_s001
                                 + 1.2404 w_k1_IBM_s001
                                 + 1.00067 w_k1_INTC_s001
                                 + 1.26425 w_k1_KO_s001 + 1.26771 w_k1_MAT_s001
                                 + 1.04482 w_k1_MSFT_s001
                                 + 1.24562 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.19206 w_k1_ADM_s002
                                 + 1.30359 w_k1_AXP_s002
                                 + 1.67515 w_k1_DIS_s002
                                 + 1.31744 w_k1_FDX_s002
                                 + 1.25248 w_k1_IBM_s002
                                 + 1.25982 w_k1_INTC_s002
                                 + 1.25843 w_k1_KO_s002 + 1.14507 w_k1_MAT_s002
                                 + 1.24027 w_k1_MSFT_s002
                                 + 1.14996 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.748486 w_k1_ADM_s002
                                 + 0.813206 w_k1_AXP_s002
                                 + 0.620142 w_k1_DIS_s002
                                 + 0.702967 w_k1_FDX_s002
                                 + 1.10694 w_k1_IBM_s002
                                 + 0.775101 w_k1_INTC_s002
                                 + 0.815709 w_k1_KO_s002
                                 + 1.11587 w_k1_MAT_s002
                                 + 0.698055 w_k1_MSFT_s002
                                 + 0.83587 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.05034 w_k1_ADM_s002
                                 + 1.17201 w_k1_AXP_s002
                                 + 0.973191 w_k1_DIS_s002
                                 + 1.00447 w_k1_FDX_s002 + 0.9445 w_k1_IBM_s002
                                 + 0.937802 w_k1_INTC_s002
                                 + 0.890658 w_k1_KO_s002
                                 + 0.710122 w_k1_MAT_s002
                                 + 0.909514 w_k1_MSFT_s002
                                 + 1.01568 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.842566 w_k1_ADM_s002
                                 + 0.796485 w_k1_AXP_s002
                                 + 0.832491 w_k1_DIS_s002
                                 + 0.897085 w_k1_FDX_s002
                                 + 0.580374 w_k1_IBM_s002
                                 + 0.833383 w_k1_INTC_s002
                                 + 0.681802 w_k1_KO_s002
                                 + 0.402269 w_k1_MAT_s002
                                 + 0.796734 w_k1_MSFT_s002
                                 + 0.840241 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.03797 w_k1_ADM_s002
                                 + 0.994716 w_k1_AXP_s002
                                 + 1.14284 w_k1_DIS_s002 + 1.1349 w_k1_FDX_s002
                                 + 0.648273 w_k1_IBM_s002
                                 + 1.09608 w_k1_INTC_s002
                                 + 0.868459 w_k1_KO_s002
                                 + 0.559842 w_k1_MAT_s002
                                 + 1.17389 w_k1_MSFT_s002
                                 + 1.11835 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.838085 w_k1_ADM_s002
                                 + 0.575532 w_k1_AXP_s002
                                 + 0.681659 w_k1_DIS_s002
                                 + 0.784172 w_k1_FDX_s002
                                 + 0.659725 w_k1_IBM_s002
                                 + 0.807327 w_k1_INTC_s002
                                 + 0.67217 w_k1_KO_s002
                                 + 0.579822 w_k1_MAT_s002
                                 + 0.809248 w_k1_MSFT_s002
                                 + 0.840424 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.10913 w_k1_ADM_s002
                                 + 1.22523 w_k1_AXP_s002
                                 + 1.13046 w_k1_DIS_s002
                                 + 1.02305 w_k1_FDX_s002
                                 + 1.46697 w_k1_IBM_s002
                                 + 1.11692 w_k1_INTC_s002
                                 + 1.11786 w_k1_KO_s002 + 1.20605 w_k1_MAT_s002
                                 + 1.15306 w_k1_MSFT_s002
                                 + 1.09108 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.806561 w_k1_ADM_s002
                                 + 0.668968 w_k1_AXP_s002
                                 + 0.832817 w_k1_DIS_s002
                                 + 0.835732 w_k1_FDX_s002
                                 + 0.952945 w_k1_IBM_s002
                                 + 1.00867 w_k1_INTC_s002
                                 + 0.79969 w_k1_KO_s002
                                 + 0.654279 w_k1_MAT_s002
                                 + 1.01099 w_k1_MSFT_s002
                                 + 0.87615 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.875957 w_k1_ADM_s003
                                 + 0.687826 w_k1_AXP_s003
                                 + 0.779405 w_k1_DIS_s003
                                 + 0.901596 w_k1_FDX_s003
                                 + 0.706547 w_k1_IBM_s003
                                 + 0.726803 w_k1_INTC_s003
                                 + 0.869489 w_k1_KO_s003
                                 + 0.524129 w_k1_MAT_s003
                                 + 0.851568 w_k1_MSFT_s003
                                 + 0.829763 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.902981 w_k1_ADM_s003
                                 + 0.905469 w_k1_AXP_s003
                                 + 1.12468 w_k1_DIS_s003
                                 + 1.06314 w_k1_FDX_s003
                                 + 1.15228 w_k1_IBM_s003
                                 + 0.898483 w_k1_INTC_s003
                                 + 1.18631 w_k1_KO_s003 + 0.97126 w_k1_MAT_s003
                                 + 1.01546 w_k1_MSFT_s003
                                 + 1.08534 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.24547 w_k1_ADM_s003
                                 + 1.4943 w_k1_AXP_s003 + 1.47279 w_k1_DIS_s003
                                 + 1.35694 w_k1_FDX_s003
                                 + 1.40179 w_k1_IBM_s003
                                 + 1.27485 w_k1_INTC_s003
                                 + 1.42455 w_k1_KO_s003 + 1.19613 w_k1_MAT_s003
                                 + 1.23763 w_k1_MSFT_s003
                                 + 1.26105 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.02598 w_k1_ADM_s003
                                 + 0.858961 w_k1_AXP_s003
                                 + 1.11322 w_k1_DIS_s003
                                 + 1.16643 w_k1_FDX_s003
                                 + 0.85099 w_k1_IBM_s003
                                 + 1.0071 w_k1_INTC_s003
                                 + 0.982468 w_k1_KO_s003
                                 + 0.721013 w_k1_MAT_s003
                                 + 1.17129 w_k1_MSFT_s003
                                 + 1.03469 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.564361 w_k1_ADM_s003
                                 + 0.543227 w_k1_AXP_s003
                                 + 0.601358 w_k1_DIS_s003
                                 + 0.672712 w_k1_FDX_s003
                                 + 0.830869 w_k1_IBM_s003
                                 + 0.677354 w_k1_INTC_s003
                                 + 0.769364 w_k1_KO_s003
                                 + 0.710257 w_k1_MAT_s003
                                 + 0.814388 w_k1_MSFT_s003
                                 + 0.768957 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.904623 w_k1_ADM_s003
                                 + 0.911288 w_k1_AXP_s003
                                 + 0.88378 w_k1_DIS_s003
                                 + 0.906202 w_k1_FDX_s003
                                 + 0.882974 w_k1_IBM_s003
                                 + 0.879093 w_k1_INTC_s003
                                 + 0.906842 w_k1_KO_s003
                                 + 1.17761 w_k1_MAT_s003
                                 + 0.989325 w_k1_MSFT_s003
                                 + 0.966483 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.81347 w_k1_ADM_s003
                                 + 0.786352 w_k1_AXP_s003
                                 + 0.679965 w_k1_DIS_s003
                                 + 0.835697 w_k1_FDX_s003
                                 + 1.15045 w_k1_IBM_s003
                                 + 0.727309 w_k1_INTC_s003
                                 + 0.855911 w_k1_KO_s003
                                 + 0.970593 w_k1_MAT_s003
                                 + 0.770286 w_k1_MSFT_s003
                                 + 0.916969 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.11905 w_k1_ADM_s003
                                 + 1.1776 w_k1_AXP_s003 + 1.03478 w_k1_DIS_s003
                                 + 1.25546 w_k1_FDX_s003
                                 + 0.961641 w_k1_IBM_s003
                                 + 1.12809 w_k1_INTC_s003
                                 + 0.996935 w_k1_KO_s003
                                 + 0.699502 w_k1_MAT_s003
                                 + 1.19804 w_k1_MSFT_s003
                                 + 1.01712 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.919117 w_k1_ADM_s004
                                 + 0.743461 w_k1_AXP_s004
                                 + 0.650815 w_k1_DIS_s004
                                 + 0.782976 w_k1_FDX_s004
                                 + 0.950728 w_k1_IBM_s004
                                 + 0.892315 w_k1_INTC_s004
                                 + 0.852404 w_k1_KO_s004
                                 + 0.815955 w_k1_MAT_s004
                                 + 0.856131 w_k1_MSFT_s004
                                 + 0.906085 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.22676 w_k1_ADM_s004
                                 + 1.19716 w_k1_AXP_s004
                                 + 0.988254 w_k1_DIS_s004
                                 + 1.07962 w_k1_FDX_s004
                                 + 1.24336 w_k1_IBM_s004
                                 + 1.16731 w_k1_INTC_s004
                                 + 1.14985 w_k1_KO_s004 + 1.16357 w_k1_MAT_s004
                                 + 0.992005 w_k1_MSFT_s004
                                 + 1.1618 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.984915 w_k1_ADM_s004
                                 + 1.04567 w_k1_AXP_s004
                                 + 0.892346 w_k1_DIS_s004
                                 + 1.01844 w_k1_FDX_s004
                                 + 0.838832 w_k1_IBM_s004
                                 + 1.02337 w_k1_INTC_s004
                                 + 0.854473 w_k1_KO_s004
                                 + 0.800545 w_k1_MAT_s004
                                 + 1.06784 w_k1_MSFT_s004
                                 + 0.96144 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.14209 w_k1_ADM_s004
                                 + 2.14723 w_k1_AXP_s004
                                 + 1.45023 w_k1_DIS_s004
                                 + 1.35842 w_k1_FDX_s004
                                 + 1.41288 w_k1_IBM_s004
                                 + 1.06688 w_k1_INTC_s004
                                 + 1.30793 w_k1_KO_s004 + 2.06295 w_k1_MAT_s004
                                 + 1.21534 w_k1_MSFT_s004
                                 + 1.30431 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.14409 w_k1_ADM_s004
                                 + 1.21817 w_k1_AXP_s004
                                 + 1.67396 w_k1_DIS_s004
                                 + 1.31957 w_k1_FDX_s004 + 1.2727 w_k1_IBM_s004
                                 + 1.33319 w_k1_INTC_s004
                                 + 1.36796 w_k1_KO_s004 + 1.26802 w_k1_MAT_s004
                                 + 1.3621 w_k1_MSFT_s004 + 1.24705 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.776806 w_k1_ADM_s004
                                 + 1.11412 w_k1_AXP_s004
                                 + 1.33839 w_k1_DIS_s004
                                 + 1.09847 w_k1_FDX_s004
                                 + 1.24384 w_k1_IBM_s004
                                 + 1.34907 w_k1_INTC_s004
                                 + 1.36272 w_k1_KO_s004 + 1.32457 w_k1_MAT_s004
                                 + 1.37598 w_k1_MSFT_s004
                                 + 1.14198 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.06061 w_k1_ADM_s004
                                 + 0.85102 w_k1_AXP_s004
                                 + 0.839319 w_k1_DIS_s004
                                 + 0.893266 w_k1_FDX_s004
                                 + 0.773657 w_k1_IBM_s004
                                 + 0.889549 w_k1_INTC_s004
                                 + 0.788735 w_k1_KO_s004
                                 + 0.585578 w_k1_MAT_s004
                                 + 0.929338 w_k1_MSFT_s004
                                 + 0.85839 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.0802 w_k1_ADM_s004
                                 + 1.36033 w_k1_AXP_s004
                                 + 1.38861 w_k1_DIS_s004
                                 + 1.32686 w_k1_FDX_s004
                                 + 0.984161 w_k1_IBM_s004
                                 + 1.21601 w_k1_INTC_s004
                                 + 1.16938 w_k1_KO_s004
                                 + 0.940319 w_k1_MAT_s004
                                 + 1.25361 w_k1_MSFT_s004
                                 + 1.03078 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.909389 w_k1_ADM_s005
                                 + 1.02636 w_k1_AXP_s005
                                 + 1.05253 w_k1_DIS_s005
                                 + 0.973203 w_k1_FDX_s005
                                 + 1.34943 w_k1_IBM_s005
                                 + 0.915792 w_k1_INTC_s005
                                 + 1.13694 w_k1_KO_s005 + 1.24494 w_k1_MAT_s005
                                 + 1.05705 w_k1_MSFT_s005 + 1.0399 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.19601 w_k1_ADM_s005
                                 + 0.962244 w_k1_AXP_s005
                                 + 0.928535 w_k1_DIS_s005
                                 + 0.984709 w_k1_FDX_s005
                                 + 0.562758 w_k1_IBM_s005
                                 + 0.887208 w_k1_INTC_s005
                                 + 0.857071 w_k1_KO_s005
                                 + 0.733511 w_k1_MAT_s005
                                 + 1.02082 w_k1_MSFT_s005
                                 + 0.942535 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.905038 w_k1_ADM_s005
                                 + 0.790327 w_k1_AXP_s005
                                 + 0.712498 w_k1_DIS_s005
                                 + 0.884013 w_k1_FDX_s005
                                 + 0.820372 w_k1_IBM_s005
                                 + 0.754674 w_k1_INTC_s005
                                 + 0.829195 w_k1_KO_s005
                                 + 0.856163 w_k1_MAT_s005
                                 + 0.868463 w_k1_MSFT_s005
                                 + 0.832328 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.65618 w_k1_ADM_s005
                                 + 1.42171 w_k1_AXP_s005
                                 + 1.29979 w_k1_DIS_s005
                                 + 1.30919 w_k1_FDX_s005
                                 + 0.948356 w_k1_IBM_s005
                                 + 1.00973 w_k1_INTC_s005
                                 + 0.950492 w_k1_KO_s005
                                 + 1.00527 w_k1_MAT_s005
                                 + 1.09335 w_k1_MSFT_s005
                                 + 1.08941 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.894374 w_k1_ADM_s005
                                 + 0.783614 w_k1_AXP_s005
                                 + 0.887191 w_k1_DIS_s005
                                 + 0.865753 w_k1_FDX_s005
                                 + 0.869653 w_k1_IBM_s005
                                 + 0.827828 w_k1_INTC_s005
                                 + 0.949887 w_k1_KO_s005
                                 + 0.889545 w_k1_MAT_s005
                                 + 0.88994 w_k1_MSFT_s005
                                 + 0.985754 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.40772 w_k1_ADM_s005
                                 + 1.60836 w_k1_AXP_s005
                                 + 1.22733 w_k1_DIS_s005
                                 + 1.23685 w_k1_FDX_s005 + 1.3561 w_k1_IBM_s005
                                 + 1.01998 w_k1_INTC_s005
                                 + 1.12632 w_k1_KO_s005 + 1.67373 w_k1_MAT_s005
                                 + 1.23391 w_k1_MSFT_s005
                                 + 1.20052 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.13535 w_k1_ADM_s005
                                 + 1.20049 w_k1_AXP_s005 + 1.5422 w_k1_DIS_s005
                                 + 1.27054 w_k1_FDX_s005
                                 + 1.05519 w_k1_IBM_s005
                                 + 0.927003 w_k1_INTC_s005
                                 + 1.23965 w_k1_KO_s005 + 1.18941 w_k1_MAT_s005
                                 + 1.29764 w_k1_MSFT_s005
                                 + 1.08733 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.24799 w_k1_ADM_s005
                                 + 1.21923 w_k1_AXP_s005
                                 + 1.00992 w_k1_DIS_s005
                                 + 1.12647 w_k1_FDX_s005
                                 + 0.993367 w_k1_IBM_s005
                                 + 0.89316 w_k1_INTC_s005 + 1.0212 w_k1_KO_s005
                                 + 1.33408 w_k1_MAT_s005
                                 + 1.03546 w_k1_MSFT_s005
                                 + 1.06332 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.24901 w_k1_ADM_s006
                                 + 1.44541 w_k1_AXP_s006
                                 + 1.47482 w_k1_DIS_s006
                                 + 1.17222 w_k1_FDX_s006
                                 + 1.39465 w_k1_IBM_s006
                                 + 1.1893 w_k1_INTC_s006 + 1.19692 w_k1_KO_s006
                                 + 1.31162 w_k1_MAT_s006
                                 + 1.23563 w_k1_MSFT_s006
                                 + 1.05995 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.996639 w_k1_ADM_s006
                                 + 0.624727 w_k1_AXP_s006
                                 + 0.85306 w_k1_DIS_s006
                                 + 0.948358 w_k1_FDX_s006
                                 + 0.541672 w_k1_IBM_s006
                                 + 0.971049 w_k1_INTC_s006
                                 + 0.850258 w_k1_KO_s006
                                 + 0.651767 w_k1_MAT_s006
                                 + 1.13542 w_k1_MSFT_s006
                                 + 0.796785 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.939906 w_k1_ADM_s006
                                 + 0.84429 w_k1_AXP_s006
                                 + 0.837527 w_k1_DIS_s006
                                 + 0.829662 w_k1_FDX_s006
                                 + 1.12143 w_k1_IBM_s006
                                 + 0.80045 w_k1_INTC_s006
                                 + 1.03955 w_k1_KO_s006 + 1.2597 w_k1_MAT_s006
                                 + 0.849281 w_k1_MSFT_s006
                                 + 0.883856 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.11135 w_k1_ADM_s006
                                 + 1.27781 w_k1_AXP_s006
                                 + 1.32446 w_k1_DIS_s006
                                 + 1.28514 w_k1_FDX_s006
                                 + 0.970556 w_k1_IBM_s006
                                 + 1.27425 w_k1_INTC_s006
                                 + 1.06146 w_k1_KO_s006
                                 + 0.941597 w_k1_MAT_s006
                                 + 1.34697 w_k1_MSFT_s006
                                 + 0.970535 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.06401 w_k1_ADM_s006
                                 + 1.02766 w_k1_AXP_s006
                                 + 0.761332 w_k1_DIS_s006
                                 + 0.972108 w_k1_FDX_s006
                                 + 0.826147 w_k1_IBM_s006
                                 + 0.980057 w_k1_INTC_s006
                                 + 0.823716 w_k1_KO_s006
                                 + 0.954978 w_k1_MAT_s006
                                 + 0.966158 w_k1_MSFT_s006
                                 + 0.882058 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.785002 w_k1_ADM_s006
                                 + 1.25379 w_k1_AXP_s006
                                 + 0.975063 w_k1_DIS_s006
                                 + 0.953173 w_k1_FDX_s006
                                 + 1.62292 w_k1_IBM_s006
                                 + 1.10501 w_k1_INTC_s006
                                 + 1.10523 w_k1_KO_s006 + 1.76085 w_k1_MAT_s006
                                 + 0.974419 w_k1_MSFT_s006
                                 + 1.06111 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.701507 w_k1_ADM_s006
                                 + 0.697336 w_k1_AXP_s006
                                 + 0.764864 w_k1_DIS_s006
                                 + 0.770506 w_k1_FDX_s006
                                 + 0.8941 w_k1_IBM_s006
                                 + 0.85756 w_k1_INTC_s006
                                 + 0.817783 w_k1_KO_s006
                                 + 0.895146 w_k1_MAT_s006
                                 + 0.864539 w_k1_MSFT_s006
                                 + 0.816128 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.07673 w_k1_ADM_s006
                                 + 0.838723 w_k1_AXP_s006
                                 + 1.35196 w_k1_DIS_s006 + 1.1313 w_k1_FDX_s006
                                 + 0.852263 w_k1_IBM_s006
                                 + 1.16932 w_k1_INTC_s006
                                 + 1.06094 w_k1_KO_s006
                                 + 0.824868 w_k1_MAT_s006
                                 + 1.28366 w_k1_MSFT_s006
                                 + 0.962746 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.17509 w_k1_ADM_s007
                                 + 0.724533 w_k1_AXP_s007
                                 + 0.948366 w_k1_DIS_s007
                                 + 0.916717 w_k1_FDX_s007
                                 + 0.8188 w_k1_IBM_s007
                                 + 0.807116 w_k1_INTC_s007
                                 + 0.821913 w_k1_KO_s007
                                 + 0.805411 w_k1_MAT_s007
                                 + 0.829163 w_k1_MSFT_s007
                                 + 0.968674 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.976455 w_k1_ADM_s007
                                 + 0.719359 w_k1_AXP_s007
                                 + 0.719172 w_k1_DIS_s007
                                 + 0.814203 w_k1_FDX_s007
                                 + 0.645603 w_k1_IBM_s007
                                 + 0.647457 w_k1_INTC_s007
                                 + 0.647269 w_k1_KO_s007
                                 + 0.596805 w_k1_MAT_s007
                                 + 0.734188 w_k1_MSFT_s007
                                 + 0.870143 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.24194 w_k1_ADM_s007
                                 + 0.816853 w_k1_AXP_s007
                                 + 1.20466 w_k1_DIS_s007
                                 + 0.944052 w_k1_FDX_s007
                                 + 1.24658 w_k1_IBM_s007
                                 + 0.855069 w_k1_INTC_s007
                                 + 1.05053 w_k1_KO_s007
                                 + 0.917547 w_k1_MAT_s007
                                 + 0.987585 w_k1_MSFT_s007
                                 + 0.968453 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.971229 w_k1_ADM_s007
                                 + 0.947065 w_k1_AXP_s007
                                 + 1.13394 w_k1_DIS_s007
                                 + 1.16616 w_k1_FDX_s007
                                 + 0.678686 w_k1_IBM_s007
                                 + 0.956514 w_k1_INTC_s007
                                 + 0.696529 w_k1_KO_s007
                                 + 0.667873 w_k1_MAT_s007 + 1.04 w_k1_MSFT_s007
                                 + 1.00732 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.36903 w_k1_ADM_s007
                                 + 1.68809 w_k1_AXP_s007
                                 + 1.43453 w_k1_DIS_s007
                                 + 1.29747 w_k1_FDX_s007
                                 + 1.52863 w_k1_IBM_s007
                                 + 0.861975 w_k1_INTC_s007
                                 + 1.12511 w_k1_KO_s007 + 1.83083 w_k1_MAT_s007
                                 + 1.01104 w_k1_MSFT_s007
                                 + 1.14567 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.3651 w_k1_ADM_s007
                                 + 1.28033 w_k1_AXP_s007
                                 + 1.24413 w_k1_DIS_s007
                                 + 1.37342 w_k1_FDX_s007
                                 + 0.933044 w_k1_IBM_s007
                                 + 1.24034 w_k1_INTC_s007
                                 + 1.07609 w_k1_KO_s007 + 1.07679 w_k1_MAT_s007
                                 + 1.12328 w_k1_MSFT_s007
                                 + 1.05059 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.01788 w_k1_ADM_s007
                                 + 1.15148 w_k1_AXP_s007
                                 + 0.87899 w_k1_DIS_s007
                                 + 1.07283 w_k1_FDX_s007
                                 + 0.876075 w_k1_IBM_s007
                                 + 0.842581 w_k1_INTC_s007
                                 + 0.75857 w_k1_KO_s007 + 1.20181 w_k1_MAT_s007
                                 + 0.759535 w_k1_MSFT_s007
                                 + 0.942812 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.893474 w_k1_ADM_s007
                                 + 0.696662 w_k1_AXP_s007
                                 + 0.713553 w_k1_DIS_s007
                                 + 0.736321 w_k1_FDX_s007
                                 + 1.0079 w_k1_IBM_s007
                                 + 0.921577 w_k1_INTC_s007
                                 + 0.825967 w_k1_KO_s007
                                 + 0.896712 w_k1_MAT_s007
                                 + 0.76866 w_k1_MSFT_s007
                                 + 0.742509 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.06131 w_k1_ADM_s010
                                 + 1.10155 w_k1_AXP_s010
                                 + 0.868223 w_k1_DIS_s010
                                 + 1.07457 w_k1_FDX_s010
                                 + 0.757807 w_k1_IBM_s010
                                 + 0.905069 w_k1_INTC_s010
                                 + 0.689331 w_k1_KO_s010
                                 + 0.787822 w_k1_MAT_s010
                                 + 0.900301 w_k1_MSFT_s010
                                 + 0.91535 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.916359 w_k1_ADM_s010
                                 + 0.893075 w_k1_AXP_s010
                                 + 0.884207 w_k1_DIS_s010
                                 + 0.937667 w_k1_FDX_s010
                                 + 1.09706 w_k1_IBM_s010
                                 + 0.906388 w_k1_INTC_s010
                                 + 0.96985 w_k1_KO_s010
                                 + 0.882219 w_k1_MAT_s010
                                 + 0.786243 w_k1_MSFT_s010
                                 + 0.91727 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.843177 w_k1_ADM_s010
                                 + 0.982899 w_k1_AXP_s010
                                 + 1.04652 w_k1_DIS_s010
                                 + 0.964265 w_k1_FDX_s010
                                 + 1.48096 w_k1_IBM_s010
                                 + 1.15824 w_k1_INTC_s010
                                 + 1.08973 w_k1_KO_s010 + 1.24358 w_k1_MAT_s010
                                 + 1.05991 w_k1_MSFT_s010 + 1.0629 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.15835 w_k1_ADM_s010
                                 + 1.24371 w_k1_AXP_s010
                                 + 1.51485 w_k1_DIS_s010
                                 + 1.27214 w_k1_FDX_s010 + 1.1362 w_k1_IBM_s010
                                 + 1.03841 w_k1_INTC_s010
                                 + 1.21696 w_k1_KO_s010 + 1.30013 w_k1_MAT_s010
                                 + 1.15787 w_k1_MSFT_s010
                                 + 1.12366 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.99377 w_k1_ADM_s010
                                 + 1.3186 w_k1_AXP_s010 + 1.28035 w_k1_DIS_s010
                                 + 1.12561 w_k1_FDX_s010
                                 + 1.76532 w_k1_IBM_s010
                                 + 1.1798 w_k1_INTC_s010 + 1.27865 w_k1_KO_s010
                                 + 1.36646 w_k1_MAT_s010
                                 + 1.09416 w_k1_MSFT_s010 + 1.1869 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.942145 w_k1_ADM_s010
                                 + 1.53027 w_k1_AXP_s010
                                 + 1.09884 w_k1_DIS_s010
                                 + 1.24702 w_k1_FDX_s010
                                 + 1.11607 w_k1_IBM_s010
                                 + 1.14222 w_k1_INTC_s010
                                 + 0.950928 w_k1_KO_s010
                                 + 1.12781 w_k1_MAT_s010
                                 + 0.973448 w_k1_MSFT_s010
                                 + 1.15621 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.754883 w_k1_ADM_s010
                                 + 0.626829 w_k1_AXP_s010
                                 + 0.58592 w_k1_DIS_s010
                                 + 0.877468 w_k1_FDX_s010
                                 + 0.707058 w_k1_IBM_s010
                                 + 0.894814 w_k1_INTC_s010
                                 + 0.682286 w_k1_KO_s010
                                 + 0.733989 w_k1_MAT_s010
                                 + 0.807377 w_k1_MSFT_s010
                                 + 0.841899 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.845957 w_k1_ADM_s010
                                 + 0.766161 w_k1_AXP_s010
                                 + 1.00913 w_k1_DIS_s010
                                 + 0.965303 w_k1_FDX_s010
                                 + 0.889315 w_k1_IBM_s010
                                 + 0.991953 w_k1_INTC_s010
                                 + 0.935315 w_k1_KO_s010
                                 + 0.741413 w_k1_MAT_s010
                                 + 0.97644 w_k1_MSFT_s010
                                 + 1.00407 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.644149 w_k1_ADM_s011
                                 + 0.701912 w_k1_AXP_s011
                                 + 0.677021 w_k1_DIS_s011
                                 + 0.804092 w_k1_FDX_s011
                                 + 0.666295 w_k1_IBM_s011
                                 + 0.904453 w_k1_INTC_s011
                                 + 0.677722 w_k1_KO_s011
                                 + 0.488494 w_k1_MAT_s011
                                 + 0.726343 w_k1_MSFT_s011
                                 + 0.762585 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.24604 w_k1_ADM_s011
                                 + 1.08688 w_k1_AXP_s011
                                 + 1.46353 w_k1_DIS_s011
                                 + 1.36261 w_k1_FDX_s011
                                 + 0.783084 w_k1_IBM_s011
                                 + 1.11966 w_k1_INTC_s011
                                 + 0.950901 w_k1_KO_s011
                                 + 0.654394 w_k1_MAT_s011
                                 + 1.09989 w_k1_MSFT_s011
                                 + 0.871461 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.13178 w_k1_ADM_s011
                                 + 1.20519 w_k1_AXP_s011
                                 + 1.80887 w_k1_DIS_s011
                                 + 1.22885 w_k1_FDX_s011
                                 + 1.23599 w_k1_IBM_s011
                                 + 1.26896 w_k1_INTC_s011
                                 + 1.35345 w_k1_KO_s011 + 1.3208 w_k1_MAT_s011
                                 + 1.31088 w_k1_MSFT_s011 + 1.0536 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.74441 w_k1_ADM_s011
                                 + 0.788401 w_k1_AXP_s011
                                 + 0.739369 w_k1_DIS_s011
                                 + 0.753887 w_k1_FDX_s011
                                 + 0.932448 w_k1_IBM_s011
                                 + 0.800616 w_k1_INTC_s011
                                 + 0.757105 w_k1_KO_s011
                                 + 0.942551 w_k1_MAT_s011
                                 + 0.730567 w_k1_MSFT_s011
                                 + 0.743835 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.06484 w_k1_ADM_s011
                                 + 1.17589 w_k1_AXP_s011 + 1.1211 w_k1_DIS_s011
                                 + 1.01104 w_k1_FDX_s011
                                 + 1.26628 w_k1_IBM_s011
                                 + 1.11681 w_k1_INTC_s011
                                 + 1.12372 w_k1_KO_s011
                                 + 0.952209 w_k1_MAT_s011
                                 + 0.977225 w_k1_MSFT_s011
                                 + 0.906781 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.980234 w_k1_ADM_s011
                                 + 0.855854 w_k1_AXP_s011
                                 + 0.835848 w_k1_DIS_s011
                                 + 0.899866 w_k1_FDX_s011
                                 + 1.05039 w_k1_IBM_s011
                                 + 1.00701 w_k1_INTC_s011
                                 + 0.958789 w_k1_KO_s011
                                 + 0.758017 w_k1_MAT_s011
                                 + 0.823621 w_k1_MSFT_s011
                                 + 0.892461 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.866653 w_k1_ADM_s011
                                 + 1.40231 w_k1_AXP_s011
                                 + 1.38199 w_k1_DIS_s011
                                 + 1.14071 w_k1_FDX_s011
                                 + 1.47729 w_k1_IBM_s011
                                 + 1.39217 w_k1_INTC_s011
                                 + 1.34926 w_k1_KO_s011 + 1.77293 w_k1_MAT_s011
                                 + 1.14731 w_k1_MSFT_s011
                                 + 0.997015 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.977174 w_k1_ADM_s011
                                 + 1.19015 w_k1_AXP_s011
                                 + 1.15741 w_k1_DIS_s011
                                 + 1.05347 w_k1_FDX_s011
                                 + 0.86747 w_k1_IBM_s011
                                 + 1.08627 w_k1_INTC_s011 + 1.0182 w_k1_KO_s011
                                 + 1.02368 w_k1_MAT_s011
                                 + 0.961947 w_k1_MSFT_s011
                                 + 0.951311 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.767359 w_k1_ADM_s012
                                 + 0.916182 w_k1_AXP_s012
                                 + 0.886595 w_k1_DIS_s012
                                 + 0.902438 w_k1_FDX_s012
                                 + 0.985418 w_k1_IBM_s012
                                 + 1.10349 w_k1_INTC_s012
                                 + 0.878988 w_k1_KO_s012
                                 + 0.800487 w_k1_MAT_s012
                                 + 0.899636 w_k1_MSFT_s012
                                 + 1.01483 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.949046 w_k1_ADM_s012
                                 + 1.20662 w_k1_AXP_s012
                                 + 1.19736 w_k1_DIS_s012
                                 + 1.01736 w_k1_FDX_s012
                                 + 1.44336 w_k1_IBM_s012
                                 + 0.976175 w_k1_INTC_s012
                                 + 1.1477 w_k1_KO_s012 + 1.42268 w_k1_MAT_s012
                                 + 1.01615 w_k1_MSFT_s012
                                 + 1.09838 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.57266 w_k1_ADM_s012
                                 + 0.656164 w_k1_AXP_s012
                                 + 0.509738 w_k1_DIS_s012
                                 + 0.70813 w_k1_FDX_s012
                                 + 0.985697 w_k1_IBM_s012
                                 + 0.878871 w_k1_INTC_s012
                                 + 0.819956 w_k1_KO_s012
                                 + 0.843388 w_k1_MAT_s012
                                 + 0.750028 w_k1_MSFT_s012
                                 + 0.899362 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08352 w_k1_ADM_s012
                                 + 1.34602 w_k1_AXP_s012
                                 + 1.52145 w_k1_DIS_s012
                                 + 1.24349 w_k1_FDX_s012
                                 + 1.22774 w_k1_IBM_s012
                                 + 1.20079 w_k1_INTC_s012
                                 + 1.36184 w_k1_KO_s012 + 1.37314 w_k1_MAT_s012
                                 + 1.20404 w_k1_MSFT_s012
                                 + 1.16116 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.876441 w_k1_ADM_s012
                                 + 1.17042 w_k1_AXP_s012
                                 + 1.00776 w_k1_DIS_s012
                                 + 1.11156 w_k1_FDX_s012
                                 + 0.95168 w_k1_IBM_s012
                                 + 1.44264 w_k1_INTC_s012
                                 + 1.04692 w_k1_KO_s012 + 1.02929 w_k1_MAT_s012
                                 + 1.11116 w_k1_MSFT_s012 + 1.0598 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.682024 w_k1_ADM_s012
                                 + 0.73196 w_k1_AXP_s012
                                 + 0.611146 w_k1_DIS_s012
                                 + 0.774992 w_k1_FDX_s012
                                 + 0.613667 w_k1_IBM_s012
                                 + 0.83412 w_k1_INTC_s012
                                 + 0.59754 w_k1_KO_s012
                                 + 0.538933 w_k1_MAT_s012
                                 + 0.753765 w_k1_MSFT_s012
                                 + 0.849583 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.573406 w_k1_ADM_s012
                                 + 0.915649 w_k1_AXP_s012
                                 + 1.08509 w_k1_DIS_s012
                                 + 1.04886 w_k1_FDX_s012
                                 + 0.738584 w_k1_IBM_s012
                                 + 1.31496 w_k1_INTC_s012
                                 + 0.891117 w_k1_KO_s012
                                 + 0.655814 w_k1_MAT_s012
                                 + 1.06491 w_k1_MSFT_s012
                                 + 0.97111 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.648655 w_k1_ADM_s012
                                 + 0.799216 w_k1_AXP_s012
                                 + 0.815864 w_k1_DIS_s012
                                 + 0.845487 w_k1_FDX_s012
                                 + 0.765767 w_k1_IBM_s012
                                 + 0.918477 w_k1_INTC_s012
                                 + 0.754736 w_k1_KO_s012
                                 + 0.749815 w_k1_MAT_s012
                                 + 0.965177 w_k1_MSFT_s012
                                 + 0.958814 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.12608 w_k1_ADM_s013
                                 + 1.19181 w_k1_AXP_s013
                                 + 1.17212 w_k1_DIS_s013
                                 + 1.17392 w_k1_FDX_s013
                                 + 0.945384 w_k1_IBM_s013
                                 + 1.15296 w_k1_INTC_s013
                                 + 0.897278 w_k1_KO_s013
                                 + 0.863324 w_k1_MAT_s013
                                 + 1.21549 w_k1_MSFT_s013
                                 + 1.03004 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.34323 w_k1_ADM_s013
                                 + 1.50384 w_k1_AXP_s013 + 1.614 w_k1_DIS_s013
                                 + 1.40148 w_k1_FDX_s013
                                 + 1.40838 w_k1_IBM_s013
                                 + 1.18415 w_k1_INTC_s013 + 1.3637 w_k1_KO_s013
                                 + 1.48425 w_k1_MAT_s013
                                 + 1.17108 w_k1_MSFT_s013 + 1.1344 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.99687 w_k1_ADM_s013
                                 + 0.833032 w_k1_AXP_s013
                                 + 0.880573 w_k1_DIS_s013
                                 + 0.99967 w_k1_FDX_s013 + 0.7158 w_k1_IBM_s013
                                 + 0.9388 w_k1_INTC_s013
                                 + 0.804864 w_k1_KO_s013
                                 + 0.731555 w_k1_MAT_s013
                                 + 0.931648 w_k1_MSFT_s013
                                 + 0.897622 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.3328 w_k1_ADM_s013
                                 + 1.02857 w_k1_AXP_s013
                                 + 0.979761 w_k1_DIS_s013
                                 + 1.11936 w_k1_FDX_s013
                                 + 0.680601 w_k1_IBM_s013
                                 + 0.844921 w_k1_INTC_s013
                                 + 0.773351 w_k1_KO_s013
                                 + 0.789518 w_k1_MAT_s013
                                 + 0.915858 w_k1_MSFT_s013
                                 + 0.974672 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.874562 w_k1_ADM_s013
                                 + 0.780982 w_k1_AXP_s013
                                 + 0.677317 w_k1_DIS_s013
                                 + 0.864001 w_k1_FDX_s013
                                 + 0.746715 w_k1_IBM_s013
                                 + 0.903972 w_k1_INTC_s013
                                 + 0.836189 w_k1_KO_s013
                                 + 0.778646 w_k1_MAT_s013
                                 + 0.838511 w_k1_MSFT_s013
                                 + 0.847906 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.00959 w_k1_ADM_s013
                                 + 1.09955 w_k1_AXP_s013
                                 + 1.08171 w_k1_DIS_s013
                                 + 0.965174 w_k1_FDX_s013
                                 + 1.30925 w_k1_IBM_s013
                                 + 1.18266 w_k1_INTC_s013
                                 + 1.18824 w_k1_KO_s013 + 1.48945 w_k1_MAT_s013
                                 + 1.13817 w_k1_MSFT_s013
                                 + 1.12396 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.834684 w_k1_ADM_s013
                                 + 0.917144 w_k1_AXP_s013
                                 + 0.722208 w_k1_DIS_s013
                                 + 0.75372 w_k1_FDX_s013
                                 + 1.03858 w_k1_IBM_s013
                                 + 0.674211 w_k1_INTC_s013
                                 + 0.761076 w_k1_KO_s013
                                 + 0.984587 w_k1_MAT_s013
                                 + 0.802192 w_k1_MSFT_s013
                                 + 0.829499 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.99582 w_k1_ADM_s013
                                 + 0.835623 w_k1_AXP_s013
                                 + 0.744614 w_k1_DIS_s013
                                 + 0.906768 w_k1_FDX_s013
                                 + 1.02068 w_k1_IBM_s013
                                 + 0.874975 w_k1_INTC_s013
                                 + 0.822323 w_k1_KO_s013
                                 + 0.686103 w_k1_MAT_s013
                                 + 0.809773 w_k1_MSFT_s013
                                 + 0.928951 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.839698 w_k1_ADM_s014
                                 + 0.863155 w_k1_AXP_s014
                                 + 0.652121 w_k1_DIS_s014
                                 + 0.737965 w_k1_FDX_s014
                                 + 1.32407 w_k1_IBM_s014
                                 + 0.613971 w_k1_INTC_s014
                                 + 0.818398 w_k1_KO_s014
                                 + 1.55226 w_k1_MAT_s014
                                 + 0.772957 w_k1_MSFT_s014
                                 + 0.917838 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.03313 w_k1_ADM_s014
                                 + 0.715171 w_k1_AXP_s014
                                 + 0.480671 w_k1_DIS_s014
                                 + 0.848923 w_k1_FDX_s014
                                 + 0.461837 w_k1_IBM_s014
                                 + 0.683597 w_k1_INTC_s014
                                 + 0.494785 w_k1_KO_s014
                                 + 0.47931 w_k1_MAT_s014
                                 + 0.757558 w_k1_MSFT_s014
                                 + 0.787592 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.20335 w_k1_ADM_s014
                                 + 1.38773 w_k1_AXP_s014 + 1.4414 w_k1_DIS_s014
                                 + 1.25798 w_k1_FDX_s014
                                 + 0.992899 w_k1_IBM_s014
                                 + 0.880182 w_k1_INTC_s014
                                 + 0.915153 w_k1_KO_s014 + 1.2462 w_k1_MAT_s014
                                 + 1.22724 w_k1_MSFT_s014 + 1.1489 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.069 w_k1_ADM_s014
                                 + 1.11677 w_k1_AXP_s014
                                 + 1.24631 w_k1_DIS_s014
                                 + 1.07065 w_k1_FDX_s014
                                 + 1.62223 w_k1_IBM_s014
                                 + 0.97713 w_k1_INTC_s014 + 1.2254 w_k1_KO_s014
                                 + 1.29448 w_k1_MAT_s014
                                 + 1.09898 w_k1_MSFT_s014
                                 + 1.20378 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.951555 w_k1_ADM_s014
                                 + 0.96044 w_k1_AXP_s014
                                 + 1.00131 w_k1_DIS_s014
                                 + 0.999867 w_k1_FDX_s014
                                 + 1.20822 w_k1_IBM_s014
                                 + 0.694264 w_k1_INTC_s014
                                 + 0.905466 w_k1_KO_s014 + 1.2804 w_k1_MAT_s014
                                 + 0.895438 w_k1_MSFT_s014
                                 + 1.03699 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.05973 w_k1_ADM_s014
                                 + 0.774687 w_k1_AXP_s014
                                 + 0.775321 w_k1_DIS_s014
                                 + 0.988058 w_k1_FDX_s014
                                 + 0.636971 w_k1_IBM_s014
                                 + 0.71999 w_k1_INTC_s014
                                 + 0.704363 w_k1_KO_s014
                                 + 0.587105 w_k1_MAT_s014
                                 + 0.987844 w_k1_MSFT_s014
                                 + 0.901508 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.00785 w_k1_ADM_s014
                                 + 0.80458 w_k1_AXP_s014
                                 + 0.658004 w_k1_DIS_s014
                                 + 0.724239 w_k1_FDX_s014
                                 + 0.840101 w_k1_IBM_s014
                                 + 0.678553 w_k1_INTC_s014
                                 + 0.721452 w_k1_KO_s014
                                 + 0.74404 w_k1_MAT_s014
                                 + 0.743006 w_k1_MSFT_s014
                                 + 0.823835 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.0045 w_k1_ADM_s014
                                 + 0.886666 w_k1_AXP_s014
                                 + 1.01566 w_k1_DIS_s014
                                 + 0.904098 w_k1_FDX_s014
                                 + 0.901248 w_k1_IBM_s014
                                 + 0.914878 w_k1_INTC_s014
                                 + 0.889601 w_k1_KO_s014
                                 + 0.781885 w_k1_MAT_s014
                                 + 1.03399 w_k1_MSFT_s014
                                 + 0.902436 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.895376 w_k1_ADM_s015
                                 + 0.70053 w_k1_AXP_s015
                                 + 0.849728 w_k1_DIS_s015
                                 + 0.80596 w_k1_FDX_s015
                                 + 0.904201 w_k1_IBM_s015
                                 + 0.886552 w_k1_INTC_s015
                                 + 0.962754 w_k1_KO_s015
                                 + 0.654039 w_k1_MAT_s015
                                 + 0.921568 w_k1_MSFT_s015
                                 + 0.889706 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.18199 w_k1_ADM_s015
                                 + 0.657618 w_k1_AXP_s015
                                 + 0.767691 w_k1_DIS_s015
                                 + 0.896578 w_k1_FDX_s015
                                 + 0.659903 w_k1_IBM_s015
                                 + 0.767585 w_k1_INTC_s015
                                 + 0.759644 w_k1_KO_s015
                                 + 0.501519 w_k1_MAT_s015
                                 + 1.01262 w_k1_MSFT_s015
                                 + 0.826998 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.2485 w_k1_ADM_s015
                                 + 1.18613 w_k1_AXP_s015
                                 + 1.24709 w_k1_DIS_s015
                                 + 1.08657 w_k1_FDX_s015
                                 + 1.06225 w_k1_IBM_s015
                                 + 1.09001 w_k1_INTC_s015
                                 + 1.31057 w_k1_KO_s015 + 1.04679 w_k1_MAT_s015
                                 + 1.3459 w_k1_MSFT_s015 + 1.09839 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.28498 w_k1_ADM_s015
                                 + 1.35301 w_k1_AXP_s015
                                 + 1.13719 w_k1_DIS_s015
                                 + 0.913274 w_k1_FDX_s015
                                 + 1.58202 w_k1_IBM_s015
                                 + 1.05881 w_k1_INTC_s015 + 1.4551 w_k1_KO_s015
                                 + 1.48584 w_k1_MAT_s015
                                 + 1.17131 w_k1_MSFT_s015
                                 + 1.14796 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.922968 w_k1_ADM_s015
                                 + 0.765765 w_k1_AXP_s015
                                 + 0.712025 w_k1_DIS_s015
                                 + 0.809494 w_k1_FDX_s015
                                 + 1.23332 w_k1_IBM_s015
                                 + 0.913775 w_k1_INTC_s015
                                 + 1.11343 w_k1_KO_s015
                                 + 0.790196 w_k1_MAT_s015
                                 + 0.872704 w_k1_MSFT_s015
                                 + 0.906721 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.34981 w_k1_ADM_s015
                                 + 1.16703 w_k1_AXP_s015
                                 + 1.07084 w_k1_DIS_s015
                                 + 1.02302 w_k1_FDX_s015
                                 + 0.795893 w_k1_IBM_s015
                                 + 0.933847 w_k1_INTC_s015
                                 + 0.976965 w_k1_KO_s015
                                 + 0.637239 w_k1_MAT_s015
                                 + 1.01625 w_k1_MSFT_s015
                                 + 0.929233 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.46841 w_k1_ADM_s015
                                 + 1.83212 w_k1_AXP_s015
                                 + 1.78154 w_k1_DIS_s015
                                 + 1.30204 w_k1_FDX_s015 + 1.6479 w_k1_IBM_s015
                                 + 1.15702 w_k1_INTC_s015
                                 + 1.73155 w_k1_KO_s015 + 1.76648 w_k1_MAT_s015
                                 + 1.37484 w_k1_MSFT_s015
                                 + 1.29268 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.30202 w_k1_ADM_s015
                                 + 1.23537 w_k1_AXP_s015
                                 + 0.983896 w_k1_DIS_s015
                                 + 0.97868 w_k1_FDX_s015
                                 + 1.17669 w_k1_IBM_s015
                                 + 0.812609 w_k1_INTC_s015
                                 + 1.12702 w_k1_KO_s015 + 1.32128 w_k1_MAT_s015
                                 + 1.02532 w_k1_MSFT_s015
                                 + 0.893385 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.803219 w_k1_ADM_s016
                                 + 0.825371 w_k1_AXP_s016
                                 + 0.895163 w_k1_DIS_s016
                                 + 0.858405 w_k1_FDX_s016
                                 + 1.15968 w_k1_IBM_s016
                                 + 0.816056 w_k1_INTC_s016
                                 + 0.922703 w_k1_KO_s016
                                 + 0.962876 w_k1_MAT_s016
                                 + 0.947082 w_k1_MSFT_s016
                                 + 0.922093 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.29741 w_k1_ADM_s016
                                 + 1.87397 w_k1_AXP_s016
                                 + 1.69651 w_k1_DIS_s016 + 1.4817 w_k1_FDX_s016
                                 + 1.23428 w_k1_IBM_s016
                                 + 1.14973 w_k1_INTC_s016
                                 + 1.19838 w_k1_KO_s016 + 1.57267 w_k1_MAT_s016
                                 + 1.27265 w_k1_MSFT_s016
                                 + 1.29547 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.854748 w_k1_ADM_s016
                                 + 1.36151 w_k1_AXP_s016
                                 + 0.935842 w_k1_DIS_s016
                                 + 1.02527 w_k1_FDX_s016
                                 + 1.00228 w_k1_IBM_s016
                                 + 0.915331 w_k1_INTC_s016
                                 + 0.884114 w_k1_KO_s016
                                 + 1.33182 w_k1_MAT_s016
                                 + 0.95834 w_k1_MSFT_s016
                                 + 1.06204 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.03639 w_k1_ADM_s016
                                 + 1.08574 w_k1_AXP_s016
                                 + 1.21954 w_k1_DIS_s016
                                 + 1.06685 w_k1_FDX_s016
                                 + 1.65354 w_k1_IBM_s016
                                 + 0.860999 w_k1_INTC_s016
                                 + 1.22823 w_k1_KO_s016 + 1.75543 w_k1_MAT_s016
                                 + 1.03281 w_k1_MSFT_s016 + 1.2394 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.715776 w_k1_ADM_s016
                                 + 0.936259 w_k1_AXP_s016
                                 + 0.72925 w_k1_DIS_s016
                                 + 0.897738 w_k1_FDX_s016
                                 + 0.640334 w_k1_IBM_s016
                                 + 0.903439 w_k1_INTC_s016
                                 + 0.583304 w_k1_KO_s016
                                 + 0.692612 w_k1_MAT_s016
                                 + 0.867032 w_k1_MSFT_s016
                                 + 0.850457 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.833788 w_k1_ADM_s016
                                 + 1.46523 w_k1_AXP_s016
                                 + 1.27199 w_k1_DIS_s016
                                 + 1.21449 w_k1_FDX_s016
                                 + 1.15347 w_k1_IBM_s016
                                 + 1.18306 w_k1_INTC_s016
                                 + 1.03494 w_k1_KO_s016 + 1.64287 w_k1_MAT_s016
                                 + 1.30218 w_k1_MSFT_s016
                                 + 1.02118 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.15034 w_k1_ADM_s016
                                 + 1.11642 w_k1_AXP_s016
                                 + 1.01429 w_k1_DIS_s016
                                 + 1.13275 w_k1_FDX_s016
                                 + 0.883697 w_k1_IBM_s016
                                 + 1.10026 w_k1_INTC_s016
                                 + 0.939202 w_k1_KO_s016
                                 + 0.996413 w_k1_MAT_s016
                                 + 1.10749 w_k1_MSFT_s016
                                 + 1.02505 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.833785 w_k1_ADM_s016
                                 + 1.42819 w_k1_AXP_s016
                                 + 1.13132 w_k1_DIS_s016
                                 + 1.25859 w_k1_FDX_s016
                                 + 0.816945 w_k1_IBM_s016
                                 + 1.06104 w_k1_INTC_s016
                                 + 0.818841 w_k1_KO_s016
                                 + 0.999072 w_k1_MAT_s016
                                 + 1.0491 w_k1_MSFT_s016 + 1.02017 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.00061 w_k1_ADM_s017
                                 + 0.775029 w_k1_AXP_s017
                                 + 0.623504 w_k1_DIS_s017
                                 + 0.799442 w_k1_FDX_s017
                                 + 0.596152 w_k1_IBM_s017
                                 + 0.717883 w_k1_INTC_s017
                                 + 0.612233 w_k1_KO_s017
                                 + 0.536879 w_k1_MAT_s017
                                 + 0.697571 w_k1_MSFT_s017
                                 + 0.901824 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.888439 w_k1_ADM_s017
                                 + 0.831092 w_k1_AXP_s017
                                 + 0.853398 w_k1_DIS_s017
                                 + 0.943482 w_k1_FDX_s017
                                 + 1.13789 w_k1_IBM_s017
                                 + 0.956701 w_k1_INTC_s017
                                 + 1.07487 w_k1_KO_s017
                                 + 0.942853 w_k1_MAT_s017
                                 + 0.876131 w_k1_MSFT_s017
                                 + 1.01425 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.768216 w_k1_ADM_s017
                                 + 0.741604 w_k1_AXP_s017
                                 + 0.763222 w_k1_DIS_s017
                                 + 0.753294 w_k1_FDX_s017
                                 + 1.1447 w_k1_IBM_s017
                                 + 0.641143 w_k1_INTC_s017
                                 + 0.898642 w_k1_KO_s017
                                 + 0.825084 w_k1_MAT_s017
                                 + 0.714255 w_k1_MSFT_s017
                                 + 0.871369 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.19776 w_k1_ADM_s017
                                 + 1.11811 w_k1_AXP_s017
                                 + 1.18953 w_k1_DIS_s017
                                 + 1.15767 w_k1_FDX_s017
                                 + 1.05117 w_k1_IBM_s017
                                 + 0.965768 w_k1_INTC_s017
                                 + 1.05537 w_k1_KO_s017 + 1.01269 w_k1_MAT_s017
                                 + 1.08591 w_k1_MSFT_s017
                                 + 1.07793 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.977039 w_k1_ADM_s017
                                 + 1.02442 w_k1_AXP_s017 + 1.1573 w_k1_DIS_s017
                                 + 1.09634 w_k1_FDX_s017
                                 + 0.811142 w_k1_IBM_s017
                                 + 0.903584 w_k1_INTC_s017
                                 + 0.941583 w_k1_KO_s017
                                 + 1.16023 w_k1_MAT_s017
                                 + 1.0235 w_k1_MSFT_s017 + 1.07857 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.948055 w_k1_ADM_s017
                                 + 0.789313 w_k1_AXP_s017
                                 + 0.735915 w_k1_DIS_s017
                                 + 0.799881 w_k1_FDX_s017
                                 + 0.843522 w_k1_IBM_s017
                                 + 0.831152 w_k1_INTC_s017
                                 + 0.81583 w_k1_KO_s017
                                 + 0.834338 w_k1_MAT_s017
                                 + 0.810318 w_k1_MSFT_s017
                                 + 0.8803 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.21976 w_k1_ADM_s017
                                 + 1.24058 w_k1_AXP_s017
                                 + 1.43119 w_k1_DIS_s017
                                 + 1.27554 w_k1_FDX_s017
                                 + 1.44783 w_k1_IBM_s017
                                 + 1.05671 w_k1_INTC_s017
                                 + 1.47342 w_k1_KO_s017 + 1.45519 w_k1_MAT_s017
                                 + 1.05286 w_k1_MSFT_s017
                                 + 1.23147 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.01623 w_k1_ADM_s017
                                 + 0.787313 w_k1_AXP_s017
                                 + 0.799052 w_k1_DIS_s017
                                 + 0.953935 w_k1_FDX_s017
                                 + 0.853666 w_k1_IBM_s017
                                 + 0.771825 w_k1_INTC_s017
                                 + 0.813246 w_k1_KO_s017
                                 + 0.689083 w_k1_MAT_s017
                                 + 0.895253 w_k1_MSFT_s017
                                 + 1.02571 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.930096 w_k1_ADM_s100
                                 + 0.859548 w_k1_AXP_s100
                                 + 0.864307 w_k1_DIS_s100
                                 + 0.780989 w_k1_FDX_s100
                                 + 1.24564 w_k1_IBM_s100
                                 + 0.852478 w_k1_INTC_s100
                                 + 0.920684 w_k1_KO_s100
                                 + 0.915456 w_k1_MAT_s100
                                 + 0.819925 w_k1_MSFT_s100
                                 + 0.96634 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.15321 w_k1_ADM_s100
                                 + 0.599043 w_k1_AXP_s100
                                 + 0.856905 w_k1_DIS_s100
                                 + 0.846295 w_k1_FDX_s100
                                 + 0.902121 w_k1_IBM_s100
                                 + 0.976969 w_k1_INTC_s100
                                 + 0.888495 w_k1_KO_s100
                                 + 0.686206 w_k1_MAT_s100
                                 + 0.95393 w_k1_MSFT_s100
                                 + 0.999804 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.744189 w_k1_ADM_s100
                                 + 0.79185 w_k1_AXP_s100
                                 + 0.746163 w_k1_DIS_s100
                                 + 0.889632 w_k1_FDX_s100
                                 + 0.671997 w_k1_IBM_s100
                                 + 0.983803 w_k1_INTC_s100
                                 + 0.675767 w_k1_KO_s100
                                 + 0.687286 w_k1_MAT_s100
                                 + 0.949036 w_k1_MSFT_s100
                                 + 1.00504 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.10223 w_k1_ADM_s100
                                 + 1.03033 w_k1_AXP_s100
                                 + 0.74685 w_k1_DIS_s100
                                 + 0.934104 w_k1_FDX_s100
                                 + 0.876103 w_k1_IBM_s100
                                 + 1.06225 w_k1_INTC_s100
                                 + 0.77375 w_k1_KO_s100 + 0.90772 w_k1_MAT_s100
                                 + 0.989587 w_k1_MSFT_s100
                                 + 0.971536 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.22331 w_k1_ADM_s100
                                 + 0.98192 w_k1_AXP_s100
                                 + 1.11325 w_k1_DIS_s100
                                 + 1.02304 w_k1_FDX_s100 + 1.254 w_k1_IBM_s100
                                 + 1.10771 w_k1_INTC_s100
                                 + 0.919718 w_k1_KO_s100
                                 + 0.901648 w_k1_MAT_s100
                                 + 0.992235 w_k1_MSFT_s100
                                 + 1.05917 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.30345 w_k1_ADM_s100
                                 + 1.30654 w_k1_AXP_s100
                                 + 1.02809 w_k1_DIS_s100 + 1.082 w_k1_FDX_s100
                                 + 1.23966 w_k1_IBM_s100
                                 + 0.995323 w_k1_INTC_s100
                                 + 0.987982 w_k1_KO_s100
                                 + 1.28353 w_k1_MAT_s100
                                 + 0.994699 w_k1_MSFT_s100
                                 + 1.22966 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.965769 w_k1_ADM_s100
                                 + 1.22358 w_k1_AXP_s100
                                 + 1.16853 w_k1_DIS_s100
                                 + 1.16425 w_k1_FDX_s100
                                 + 0.992068 w_k1_IBM_s100
                                 + 1.22406 w_k1_INTC_s100
                                 + 1.04373 w_k1_KO_s100 + 1.14649 w_k1_MAT_s100
                                 + 1.24468 w_k1_MSFT_s100
                                 + 1.20459 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.12393 w_k1_ADM_s100
                                 + 1.18428 w_k1_AXP_s100
                                 + 1.46379 w_k1_DIS_s100
                                 + 1.19203 w_k1_FDX_s100
                                 + 1.54661 w_k1_IBM_s100
                                 + 1.48382 w_k1_INTC_s100
                                 + 1.29433 w_k1_KO_s100 + 1.0066 w_k1_MAT_s100
                                 + 1.44866 w_k1_MSFT_s100
                                 + 1.26577 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.33051 w_k1_ADM_s101
                                 + 1.13641 w_k1_AXP_s101 + 1.4019 w_k1_DIS_s101
                                 + 1.02153 w_k1_FDX_s101
                                 + 1.36384 w_k1_IBM_s101
                                 + 1.31019 w_k1_INTC_s101
                                 + 1.49052 w_k1_KO_s101 + 1.29127 w_k1_MAT_s101
                                 + 1.08312 w_k1_MSFT_s101
                                 + 1.20077 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.00406 w_k1_ADM_s101
                                 + 1.0252 w_k1_AXP_s101 + 1.1769 w_k1_DIS_s101
                                 + 0.973774 w_k1_FDX_s101
                                 + 1.19092 w_k1_IBM_s101
                                 + 1.44785 w_k1_INTC_s101
                                 + 1.20104 w_k1_KO_s101 + 1.03501 w_k1_MAT_s101
                                 + 1.09596 w_k1_MSFT_s101
                                 + 1.02748 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.30725 w_k1_ADM_s101
                                 + 1.13836 w_k1_AXP_s101
                                 + 1.50052 w_k1_DIS_s101
                                 + 1.17033 w_k1_FDX_s101
                                 + 0.953635 w_k1_IBM_s101
                                 + 1.1305 w_k1_INTC_s101 + 1.21536 w_k1_KO_s101
                                 + 1.19274 w_k1_MAT_s101
                                 + 1.17737 w_k1_MSFT_s101
                                 + 1.18057 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09706 w_k1_ADM_s101
                                 + 1.12931 w_k1_AXP_s101
                                 + 1.05101 w_k1_DIS_s101
                                 + 1.01342 w_k1_FDX_s101
                                 + 0.82045 w_k1_IBM_s101
                                 + 1.25373 w_k1_INTC_s101
                                 + 1.06911 w_k1_KO_s101 + 1.02397 w_k1_MAT_s101
                                 + 1.00602 w_k1_MSFT_s101
                                 + 0.954459 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.3213 w_k1_ADM_s101
                                 + 1.67598 w_k1_AXP_s101
                                 + 1.42474 w_k1_DIS_s101
                                 + 1.34022 w_k1_FDX_s101
                                 + 1.17593 w_k1_IBM_s101
                                 + 1.34861 w_k1_INTC_s101
                                 + 1.17323 w_k1_KO_s101 + 1.42817 w_k1_MAT_s101
                                 + 1.41723 w_k1_MSFT_s101
                                 + 1.08274 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.01412 w_k1_ADM_s101
                                 + 1.0447 w_k1_AXP_s101
                                 + 0.665251 w_k1_DIS_s101
                                 + 0.727344 w_k1_FDX_s101
                                 + 0.974677 w_k1_IBM_s101
                                 + 0.980787 w_k1_INTC_s101
                                 + 0.865274 w_k1_KO_s101
                                 + 0.844448 w_k1_MAT_s101
                                 + 0.80096 w_k1_MSFT_s101
                                 + 0.831628 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.53596 w_k1_ADM_s101
                                 + 1.36012 w_k1_AXP_s101
                                 + 1.36376 w_k1_DIS_s101
                                 + 1.13501 w_k1_FDX_s101
                                 + 1.72273 w_k1_IBM_s101
                                 + 1.16803 w_k1_INTC_s101
                                 + 1.61921 w_k1_KO_s101 + 1.7432 w_k1_MAT_s101
                                 + 1.16621 w_k1_MSFT_s101
                                 + 1.28858 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.970542 w_k1_ADM_s101
                                 + 0.711582 w_k1_AXP_s101
                                 + 0.700019 w_k1_DIS_s101
                                 + 0.825622 w_k1_FDX_s101
                                 + 0.653416 w_k1_IBM_s101
                                 + 0.823211 w_k1_INTC_s101
                                 + 0.678858 w_k1_KO_s101
                                 + 0.821345 w_k1_MAT_s101
                                 + 0.840577 w_k1_MSFT_s101
                                 + 0.833784 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.964231 w_k1_ADM_s102
                                 + 1.0253 w_k1_AXP_s102
                                 + 0.987977 w_k1_DIS_s102
                                 + 1.10972 w_k1_FDX_s102
                                 + 0.836809 w_k1_IBM_s102
                                 + 0.851192 w_k1_INTC_s102
                                 + 0.82293 w_k1_KO_s102
                                 + 0.844002 w_k1_MAT_s102
                                 + 0.844737 w_k1_MSFT_s102
                                 + 0.987286 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.897483 w_k1_ADM_s102
                                 + 0.90693 w_k1_AXP_s102
                                 + 0.999293 w_k1_DIS_s102
                                 + 0.887873 w_k1_FDX_s102
                                 + 1.10095 w_k1_IBM_s102
                                 + 0.996867 w_k1_INTC_s102
                                 + 1.06008 w_k1_KO_s102
                                 + 0.909421 w_k1_MAT_s102
                                 + 1.00135 w_k1_MSFT_s102
                                 + 1.04168 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.868125 w_k1_ADM_s102
                                 + 1.07965 w_k1_AXP_s102
                                 + 0.935295 w_k1_DIS_s102
                                 + 0.929819 w_k1_FDX_s102
                                 + 1.35893 w_k1_IBM_s102
                                 + 0.937014 w_k1_INTC_s102
                                 + 0.939924 w_k1_KO_s102
                                 + 1.32156 w_k1_MAT_s102
                                 + 0.864796 w_k1_MSFT_s102
                                 + 1.04741 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.738513 w_k1_ADM_s102
                                 + 0.739054 w_k1_AXP_s102
                                 + 0.7993 w_k1_DIS_s102
                                 + 0.791947 w_k1_FDX_s102
                                 + 0.870329 w_k1_IBM_s102
                                 + 0.974983 w_k1_INTC_s102
                                 + 0.787962 w_k1_KO_s102
                                 + 0.708968 w_k1_MAT_s102
                                 + 0.859381 w_k1_MSFT_s102
                                 + 0.772894 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.01619 w_k1_ADM_s102
                                 + 1.24168 w_k1_AXP_s102
                                 + 1.49081 w_k1_DIS_s102
                                 + 1.33657 w_k1_FDX_s102
                                 + 1.01822 w_k1_IBM_s102
                                 + 1.4155 w_k1_INTC_s102
                                 + 0.956605 w_k1_KO_s102
                                 + 0.937014 w_k1_MAT_s102
                                 + 1.38433 w_k1_MSFT_s102
                                 + 1.03532 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04184 w_k1_ADM_s102
                                 + 1.75116 w_k1_AXP_s102
                                 + 1.30526 w_k1_DIS_s102 + 1.272 w_k1_FDX_s102
                                 + 1.2644 w_k1_IBM_s102 + 1.1491 w_k1_INTC_s102
                                 + 0.982568 w_k1_KO_s102
                                 + 1.85413 w_k1_MAT_s102
                                 + 1.05262 w_k1_MSFT_s102 + 1.1205 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.613639 w_k1_ADM_s102
                                 + 0.572231 w_k1_AXP_s102
                                 + 0.574138 w_k1_DIS_s102
                                 + 0.750723 w_k1_FDX_s102
                                 + 0.687927 w_k1_IBM_s102
                                 + 0.897118 w_k1_INTC_s102
                                 + 0.624246 w_k1_KO_s102
                                 + 0.45975 w_k1_MAT_s102
                                 + 0.797669 w_k1_MSFT_s102
                                 + 0.736938 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.634036 w_k1_ADM_s102
                                 + 0.633688 w_k1_AXP_s102
                                 + 0.587741 w_k1_DIS_s102
                                 + 0.739775 w_k1_FDX_s102
                                 + 0.952468 w_k1_IBM_s102
                                 + 0.734498 w_k1_INTC_s102
                                 + 0.774724 w_k1_KO_s102 + 1.0529 w_k1_MAT_s102
                                 + 0.703054 w_k1_MSFT_s102
                                 + 0.835214 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.891699 w_k1_ADM_s103
                                 + 0.834287 w_k1_AXP_s103
                                 + 0.725692 w_k1_DIS_s103
                                 + 0.848737 w_k1_FDX_s103
                                 + 1.09167 w_k1_IBM_s103
                                 + 0.879721 w_k1_INTC_s103
                                 + 1.06132 w_k1_KO_s103 + 1.52446 w_k1_MAT_s103
                                 + 0.86791 w_k1_MSFT_s103
                                 + 0.970741 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.00819 w_k1_ADM_s103
                                 + 1.54439 w_k1_AXP_s103
                                 + 1.07798 w_k1_DIS_s103
                                 + 1.21368 w_k1_FDX_s103
                                 + 1.15756 w_k1_IBM_s103
                                 + 1.09563 w_k1_INTC_s103
                                 + 1.03715 w_k1_KO_s103 + 1.83303 w_k1_MAT_s103
                                 + 0.963281 w_k1_MSFT_s103
                                 + 1.14309 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.22737 w_k1_ADM_s103
                                 + 1.48847 w_k1_AXP_s103 + 1.414 w_k1_DIS_s103
                                 + 1.41792 w_k1_FDX_s103
                                 + 1.05001 w_k1_IBM_s103
                                 + 1.34578 w_k1_INTC_s103
                                 + 1.19511 w_k1_KO_s103 + 1.52021 w_k1_MAT_s103
                                 + 1.39463 w_k1_MSFT_s103
                                 + 1.19677 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.952254 w_k1_ADM_s103
                                 + 0.874395 w_k1_AXP_s103
                                 + 0.781851 w_k1_DIS_s103
                                 + 0.957781 w_k1_FDX_s103
                                 + 0.67613 w_k1_IBM_s103
                                 + 1.04974 w_k1_INTC_s103
                                 + 0.839813 w_k1_KO_s103
                                 + 0.614411 w_k1_MAT_s103
                                 + 0.920556 w_k1_MSFT_s103
                                 + 0.852747 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.02592 w_k1_ADM_s103
                                 + 0.911637 w_k1_AXP_s103
                                 + 1.29733 w_k1_DIS_s103
                                 + 1.14901 w_k1_FDX_s103
                                 + 0.90642 w_k1_IBM_s103
                                 + 1.17958 w_k1_INTC_s103
                                 + 1.08256 w_k1_KO_s103
                                 + 0.922336 w_k1_MAT_s103
                                 + 1.16778 w_k1_MSFT_s103
                                 + 1.01255 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.05566 w_k1_ADM_s103
                                 + 1.36546 w_k1_AXP_s103
                                 + 1.62353 w_k1_DIS_s103
                                 + 1.23414 w_k1_FDX_s103
                                 + 1.58429 w_k1_IBM_s103
                                 + 1.28183 w_k1_INTC_s103 + 1.547 w_k1_KO_s103
                                 + 1.85994 w_k1_MAT_s103
                                 + 1.1031 w_k1_MSFT_s103 + 1.26637 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.05832 w_k1_ADM_s103
                                 + 1.03267 w_k1_AXP_s103
                                 + 1.00472 w_k1_DIS_s103
                                 + 1.02297 w_k1_FDX_s103
                                 + 1.19655 w_k1_IBM_s103
                                 + 1.24825 w_k1_INTC_s103 + 1.174 w_k1_KO_s103
                                 + 1.14236 w_k1_MAT_s103
                                 + 1.08739 w_k1_MSFT_s103
                                 + 1.07908 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.701786 w_k1_ADM_s103
                                 + 1.05948 w_k1_AXP_s103
                                 + 1.17659 w_k1_DIS_s103
                                 + 0.919407 w_k1_FDX_s103
                                 + 1.79603 w_k1_IBM_s103
                                 + 1.24718 w_k1_INTC_s103
                                 + 1.32907 w_k1_KO_s103 + 1.67214 w_k1_MAT_s103
                                 + 0.978514 w_k1_MSFT_s103
                                 + 1.0748 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.867649 w_k1_ADM_s104
                                 + 0.827993 w_k1_AXP_s104
                                 + 1.00319 w_k1_DIS_s104
                                 + 0.94336 w_k1_FDX_s104
                                 + 1.08213 w_k1_IBM_s104
                                 + 0.866912 w_k1_INTC_s104
                                 + 0.914166 w_k1_KO_s104
                                 + 0.886436 w_k1_MAT_s104
                                 + 1.00013 w_k1_MSFT_s104
                                 + 1.00345 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.18967 w_k1_ADM_s104
                                 + 1.24675 w_k1_AXP_s104
                                 + 1.06013 w_k1_DIS_s104
                                 + 1.07694 w_k1_FDX_s104
                                 + 1.18361 w_k1_IBM_s104
                                 + 0.990692 w_k1_INTC_s104
                                 + 1.02263 w_k1_KO_s104 + 1.60308 w_k1_MAT_s104
                                 + 1.05813 w_k1_MSFT_s104
                                 + 0.98196 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.45413 w_k1_ADM_s104
                                 + 1.38039 w_k1_AXP_s104
                                 + 1.74983 w_k1_DIS_s104
                                 + 1.32536 w_k1_FDX_s104
                                 + 1.40452 w_k1_IBM_s104
                                 + 1.08336 w_k1_INTC_s104
                                 + 1.37582 w_k1_KO_s104 + 1.58103 w_k1_MAT_s104
                                 + 1.41065 w_k1_MSFT_s104 + 1.2798 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.05278 w_k1_ADM_s104
                                 + 1.23969 w_k1_AXP_s104
                                 + 0.851947 w_k1_DIS_s104
                                 + 1.08636 w_k1_FDX_s104
                                 + 0.802626 w_k1_IBM_s104
                                 + 0.792068 w_k1_INTC_s104
                                 + 0.779793 w_k1_KO_s104
                                 + 0.851511 w_k1_MAT_s104
                                 + 0.800459 w_k1_MSFT_s104
                                 + 0.88178 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.03002 w_k1_ADM_s104
                                 + 0.975715 w_k1_AXP_s104
                                 + 1.15678 w_k1_DIS_s104
                                 + 1.11007 w_k1_FDX_s104
                                 + 0.843739 w_k1_IBM_s104
                                 + 0.96297 w_k1_INTC_s104
                                 + 1.03127 w_k1_KO_s104
                                 + 0.982435 w_k1_MAT_s104
                                 + 1.22941 w_k1_MSFT_s104
                                 + 1.07441 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.729191 w_k1_ADM_s104
                                 + 0.991068 w_k1_AXP_s104
                                 + 1.00885 w_k1_DIS_s104
                                 + 0.872538 w_k1_FDX_s104
                                 + 1.53227 w_k1_IBM_s104
                                 + 0.819804 w_k1_INTC_s104
                                 + 1.20276 w_k1_KO_s104 + 1.24267 w_k1_MAT_s104
                                 + 0.917801 w_k1_MSFT_s104
                                 + 0.944831 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.764556 w_k1_ADM_s104
                                 + 0.621434 w_k1_AXP_s104
                                 + 0.764721 w_k1_DIS_s104
                                 + 0.741241 w_k1_FDX_s104
                                 + 1.03965 w_k1_IBM_s104
                                 + 0.820047 w_k1_INTC_s104
                                 + 0.867532 w_k1_KO_s104
                                 + 0.624545 w_k1_MAT_s104
                                 + 0.865028 w_k1_MSFT_s104
                                 + 0.81407 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.801479 w_k1_ADM_s104
                                 + 0.453491 w_k1_AXP_s104
                                 + 0.628421 w_k1_DIS_s104
                                 + 0.732586 w_k1_FDX_s104
                                 + 0.75683 w_k1_IBM_s104
                                 + 0.834688 w_k1_INTC_s104
                                 + 0.744584 w_k1_KO_s104
                                 + 0.622482 w_k1_MAT_s104
                                 + 0.864797 w_k1_MSFT_s104
                                 + 0.761095 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.05419 w_k1_ADM_s105
                                 + 0.926195 w_k1_AXP_s105
                                 + 0.858188 w_k1_DIS_s105
                                 + 1.00423 w_k1_FDX_s105
                                 + 0.852638 w_k1_IBM_s105
                                 + 0.919536 w_k1_INTC_s105
                                 + 0.875313 w_k1_KO_s105
                                 + 0.866235 w_k1_MAT_s105
                                 + 0.995842 w_k1_MSFT_s105
                                 + 0.905676 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.08839 w_k1_ADM_s105
                                 + 1.57102 w_k1_AXP_s105
                                 + 1.32485 w_k1_DIS_s105
                                 + 1.22672 w_k1_FDX_s105
                                 + 1.52247 w_k1_IBM_s105
                                 + 1.18381 w_k1_INTC_s105
                                 + 1.15886 w_k1_KO_s105 + 1.85527 w_k1_MAT_s105
                                 + 1.13318 w_k1_MSFT_s105
                                 + 1.15985 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.823793 w_k1_ADM_s105
                                 + 0.88387 w_k1_AXP_s105
                                 + 0.899847 w_k1_DIS_s105
                                 + 0.960102 w_k1_FDX_s105
                                 + 1.26449 w_k1_IBM_s105
                                 + 0.885034 w_k1_INTC_s105
                                 + 0.951688 w_k1_KO_s105
                                 + 1.30295 w_k1_MAT_s105
                                 + 0.937183 w_k1_MSFT_s105
                                 + 0.924261 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.07727 w_k1_ADM_s105
                                 + 1.18343 w_k1_AXP_s105
                                 + 1.29053 w_k1_DIS_s105
                                 + 1.11953 w_k1_FDX_s105
                                 + 1.10795 w_k1_IBM_s105
                                 + 0.941017 w_k1_INTC_s105
                                 + 1.16768 w_k1_KO_s105 + 1.16497 w_k1_MAT_s105
                                 + 1.19967 w_k1_MSFT_s105 + 1.211 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.02634 w_k1_ADM_s105
                                 + 1.27485 w_k1_AXP_s105
                                 + 1.02604 w_k1_DIS_s105
                                 + 1.23856 w_k1_FDX_s105
                                 + 0.900575 w_k1_IBM_s105
                                 + 1.02043 w_k1_INTC_s105
                                 + 0.889074 w_k1_KO_s105
                                 + 0.928027 w_k1_MAT_s105
                                 + 1.01612 w_k1_MSFT_s105
                                 + 0.959818 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.548324 w_k1_ADM_s105
                                 + 0.703071 w_k1_AXP_s105
                                 + 0.652988 w_k1_DIS_s105
                                 + 0.804385 w_k1_FDX_s105
                                 + 0.686062 w_k1_IBM_s105
                                 + 0.866857 w_k1_INTC_s105
                                 + 0.752931 w_k1_KO_s105
                                 + 0.70781 w_k1_MAT_s105
                                 + 0.905763 w_k1_MSFT_s105
                                 + 0.758824 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.65404 w_k1_ADM_s105
                                 + 0.908131 w_k1_AXP_s105
                                 + 0.777282 w_k1_DIS_s105
                                 + 0.750209 w_k1_FDX_s105
                                 + 1.02798 w_k1_IBM_s105
                                 + 0.729825 w_k1_INTC_s105
                                 + 0.824646 w_k1_KO_s105 + 1.2472 w_k1_MAT_s105
                                 + 0.769361 w_k1_MSFT_s105
                                 + 0.906103 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.24088 w_k1_ADM_s105
                                 + 1.47093 w_k1_AXP_s105
                                 + 1.70585 w_k1_DIS_s105
                                 + 1.36781 w_k1_FDX_s105
                                 + 1.23128 w_k1_IBM_s105
                                 + 1.5232 w_k1_INTC_s105 + 1.35973 w_k1_KO_s105
                                 + 1.16708 w_k1_MAT_s105
                                 + 1.47936 w_k1_MSFT_s105
                                 + 1.13474 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.12841 w_k1_ADM_s106
                                 + 1.26367 w_k1_AXP_s106 + 1.2677 w_k1_DIS_s106
                                 + 1.26457 w_k1_FDX_s106
                                 + 1.08786 w_k1_IBM_s106
                                 + 1.13332 w_k1_INTC_s106
                                 + 1.21257 w_k1_KO_s106
                                 + 0.927668 w_k1_MAT_s106
                                 + 1.18753 w_k1_MSFT_s106 + 1.2539 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02253 w_k1_ADM_s106
                                 + 1.47378 w_k1_AXP_s106
                                 + 1.58206 w_k1_DIS_s106
                                 + 1.26452 w_k1_FDX_s106
                                 + 1.60538 w_k1_IBM_s106
                                 + 1.27862 w_k1_INTC_s106 + 1.4632 w_k1_KO_s106
                                 + 1.32528 w_k1_MAT_s106
                                 + 1.26532 w_k1_MSFT_s106
                                 + 1.16803 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.743107 w_k1_ADM_s106
                                 + 0.838164 w_k1_AXP_s106
                                 + 0.752916 w_k1_DIS_s106
                                 + 0.920919 w_k1_FDX_s106
                                 + 0.921159 w_k1_IBM_s106
                                 + 1.16698 w_k1_INTC_s106
                                 + 0.918548 w_k1_KO_s106
                                 + 0.809601 w_k1_MAT_s106
                                 + 1.02863 w_k1_MSFT_s106
                                 + 0.93938 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.597244 w_k1_ADM_s106
                                 + 0.546594 w_k1_AXP_s106
                                 + 0.576589 w_k1_DIS_s106
                                 + 0.637184 w_k1_FDX_s106
                                 + 1.00316 w_k1_IBM_s106
                                 + 0.778925 w_k1_INTC_s106
                                 + 0.768789 w_k1_KO_s106
                                 + 0.739107 w_k1_MAT_s106
                                 + 0.757634 w_k1_MSFT_s106
                                 + 0.88999 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.25828 w_k1_ADM_s106
                                 + 1.54611 w_k1_AXP_s106
                                 + 1.42002 w_k1_DIS_s106 + 1.3895 w_k1_FDX_s106
                                 + 0.925943 w_k1_IBM_s106
                                 + 0.96029 w_k1_INTC_s106
                                 + 1.01425 w_k1_KO_s106 + 1.12142 w_k1_MAT_s106
                                 + 1.18338 w_k1_MSFT_s106
                                 + 1.27989 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.55864 w_k1_ADM_s106
                                 + 0.906508 w_k1_AXP_s106
                                 + 0.837363 w_k1_DIS_s106
                                 + 0.839529 w_k1_FDX_s106
                                 + 1.33755 w_k1_IBM_s106
                                 + 1.32874 w_k1_INTC_s106
                                 + 1.05323 w_k1_KO_s106
                                 + 0.967676 w_k1_MAT_s106
                                 + 0.987051 w_k1_MSFT_s106
                                 + 0.868043 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.658707 w_k1_ADM_s106
                                 + 0.829586 w_k1_AXP_s106
                                 + 0.771662 w_k1_DIS_s106
                                 + 0.903552 w_k1_FDX_s106
                                 + 0.762648 w_k1_IBM_s106
                                 + 0.979872 w_k1_INTC_s106
                                 + 0.760742 w_k1_KO_s106
                                 + 0.629823 w_k1_MAT_s106
                                 + 0.98735 w_k1_MSFT_s106
                                 + 0.889469 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.761649 w_k1_ADM_s106
                                 + 1.32459 w_k1_AXP_s106
                                 + 1.07729 w_k1_DIS_s106
                                 + 1.09427 w_k1_FDX_s106
                                 + 1.46348 w_k1_IBM_s106
                                 + 1.37394 w_k1_INTC_s106
                                 + 1.20715 w_k1_KO_s106 + 1.34153 w_k1_MAT_s106
                                 + 1.19508 w_k1_MSFT_s106
                                 + 1.12086 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.822483 w_k1_ADM_s107
                                 + 0.588152 w_k1_AXP_s107
                                 + 0.593425 w_k1_DIS_s107
                                 + 0.685466 w_k1_FDX_s107
                                 + 0.805948 w_k1_IBM_s107
                                 + 0.654965 w_k1_INTC_s107
                                 + 0.705281 w_k1_KO_s107
                                 + 0.692638 w_k1_MAT_s107
                                 + 0.708475 w_k1_MSFT_s107
                                 + 0.876087 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.779481 w_k1_ADM_s107
                                 + 1.02771 w_k1_AXP_s107
                                 + 0.954978 w_k1_DIS_s107
                                 + 0.909647 w_k1_FDX_s107
                                 + 1.06919 w_k1_IBM_s107
                                 + 0.907153 w_k1_INTC_s107
                                 + 0.787858 w_k1_KO_s107
                                 + 1.27279 w_k1_MAT_s107
                                 + 0.864369 w_k1_MSFT_s107
                                 + 1.07218 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.987949 w_k1_ADM_s107
                                 + 0.790727 w_k1_AXP_s107
                                 + 0.913699 w_k1_DIS_s107
                                 + 0.787308 w_k1_FDX_s107
                                 + 1.35625 w_k1_IBM_s107
                                 + 0.70026 w_k1_INTC_s107 + 1.0105 w_k1_KO_s107
                                 + 1.42986 w_k1_MAT_s107
                                 + 0.832394 w_k1_MSFT_s107
                                 + 0.962707 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.974762 w_k1_ADM_s107
                                 + 0.938259 w_k1_AXP_s107
                                 + 1.11491 w_k1_DIS_s107
                                 + 0.998481 w_k1_FDX_s107
                                 + 1.3181 w_k1_IBM_s107
                                 + 0.94909 w_k1_INTC_s107
                                 + 1.08788 w_k1_KO_s107
                                 + 0.989846 w_k1_MAT_s107
                                 + 1.03325 w_k1_MSFT_s107
                                 + 1.15177 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.15722 w_k1_ADM_s107
                                 + 1.03784 w_k1_AXP_s107
                                 + 1.10377 w_k1_DIS_s107
                                 + 1.09823 w_k1_FDX_s107
                                 + 0.787042 w_k1_IBM_s107
                                 + 0.855018 w_k1_INTC_s107
                                 + 0.871977 w_k1_KO_s107
                                 + 0.787725 w_k1_MAT_s107
                                 + 1.08303 w_k1_MSFT_s107 + 1.0846 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.819134 w_k1_ADM_s107
                                 + 0.689318 w_k1_AXP_s107
                                 + 0.838561 w_k1_DIS_s107
                                 + 0.895933 w_k1_FDX_s107
                                 + 0.599787 w_k1_IBM_s107
                                 + 0.901744 w_k1_INTC_s107
                                 + 0.693613 w_k1_KO_s107
                                 + 0.467909 w_k1_MAT_s107
                                 + 1.04638 w_k1_MSFT_s107
                                 + 0.986636 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.941399 w_k1_ADM_s107
                                 + 0.770085 w_k1_AXP_s107
                                 + 1.10117 w_k1_DIS_s107
                                 + 0.939637 w_k1_FDX_s107
                                 + 0.925139 w_k1_IBM_s107
                                 + 0.944751 w_k1_INTC_s107
                                 + 0.851149 w_k1_KO_s107
                                 + 0.883664 w_k1_MAT_s107
                                 + 1.06141 w_k1_MSFT_s107
                                 + 0.983196 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.34746 w_k1_ADM_s107
                                 + 1.31722 w_k1_AXP_s107
                                 + 1.61116 w_k1_DIS_s107
                                 + 1.26977 w_k1_FDX_s107
                                 + 1.26034 w_k1_IBM_s107
                                 + 1.06349 w_k1_INTC_s107
                                 + 1.19922 w_k1_KO_s107 + 1.03146 w_k1_MAT_s107
                                 + 1.21736 w_k1_MSFT_s107
                                 + 1.22373 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.12292 w_k1_ADM_s110
                                 + 1.1516 w_k1_AXP_s110 + 1.13829 w_k1_DIS_s110
                                 + 1.11146 w_k1_FDX_s110
                                 + 0.852509 w_k1_IBM_s110
                                 + 1.20905 w_k1_INTC_s110
                                 + 1.01575 w_k1_KO_s110
                                 + 0.704328 w_k1_MAT_s110
                                 + 1.15182 w_k1_MSFT_s110
                                 + 1.08935 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.07912 w_k1_ADM_s110
                                 + 1.1282 w_k1_AXP_s110
                                 + 0.909756 w_k1_DIS_s110
                                 + 0.954991 w_k1_FDX_s110
                                 + 1.46741 w_k1_IBM_s110
                                 + 0.962574 w_k1_INTC_s110
                                 + 1.07683 w_k1_KO_s110 + 1.55447 w_k1_MAT_s110
                                 + 0.913475 w_k1_MSFT_s110
                                 + 1.09736 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.816071 w_k1_ADM_s110
                                 + 0.743602 w_k1_AXP_s110
                                 + 0.790633 w_k1_DIS_s110
                                 + 0.939909 w_k1_FDX_s110
                                 + 0.636754 w_k1_IBM_s110
                                 + 0.873486 w_k1_INTC_s110
                                 + 0.740774 w_k1_KO_s110
                                 + 0.599255 w_k1_MAT_s110
                                 + 0.846765 w_k1_MSFT_s110
                                 + 0.899784 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.11219 w_k1_ADM_s110
                                 + 1.35809 w_k1_AXP_s110
                                 + 1.66418 w_k1_DIS_s110
                                 + 1.35831 w_k1_FDX_s110
                                 + 1.38417 w_k1_IBM_s110
                                 + 1.31569 w_k1_INTC_s110 + 1.6241 w_k1_KO_s110
                                 + 1.4959 w_k1_MAT_s110
                                 + 1.22021 w_k1_MSFT_s110
                                 + 1.32425 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.788673 w_k1_ADM_s110
                                 + 0.776075 w_k1_AXP_s110
                                 + 0.791878 w_k1_DIS_s110
                                 + 0.826943 w_k1_FDX_s110
                                 + 1.11222 w_k1_IBM_s110
                                 + 0.989939 w_k1_INTC_s110
                                 + 0.944633 w_k1_KO_s110
                                 + 1.10846 w_k1_MAT_s110
                                 + 0.87619 w_k1_MSFT_s110
                                 + 0.988074 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.677946 w_k1_ADM_s110
                                 + 0.739526 w_k1_AXP_s110
                                 + 0.780661 w_k1_DIS_s110
                                 + 0.804807 w_k1_FDX_s110
                                 + 0.895856 w_k1_IBM_s110
                                 + 0.88816 w_k1_INTC_s110
                                 + 0.788999 w_k1_KO_s110
                                 + 0.843065 w_k1_MAT_s110
                                 + 0.863395 w_k1_MSFT_s110
                                 + 0.889838 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.07422 w_k1_ADM_s110
                                 + 1.05905 w_k1_AXP_s110
                                 + 1.43474 w_k1_DIS_s110
                                 + 1.29721 w_k1_FDX_s110
                                 + 0.931879 w_k1_IBM_s110
                                 + 1.27832 w_k1_INTC_s110
                                 + 1.13319 w_k1_KO_s110
                                 + 0.985258 w_k1_MAT_s110
                                 + 1.34813 w_k1_MSFT_s110
                                 + 1.13172 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.939657 w_k1_ADM_s110
                                 + 1.11963 w_k1_AXP_s110
                                 + 1.26747 w_k1_DIS_s110
                                 + 1.07297 w_k1_FDX_s110 + 1.2258 w_k1_IBM_s110
                                 + 0.963634 w_k1_INTC_s110
                                 + 1.1004 w_k1_KO_s110 + 1.12289 w_k1_MAT_s110
                                 + 0.967785 w_k1_MSFT_s110
                                 + 0.985815 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.19158 w_k1_ADM_s111
                                 + 1.15658 w_k1_AXP_s111
                                 + 1.47008 w_k1_DIS_s111
                                 + 1.27472 w_k1_FDX_s111
                                 + 1.40287 w_k1_IBM_s111
                                 + 1.09408 w_k1_INTC_s111 + 1.3748 w_k1_KO_s111
                                 + 1.994 w_k1_MAT_s111 + 1.2791 w_k1_MSFT_s111
                                 + 1.26772 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.985247 w_k1_ADM_s111
                                 + 1.16025 w_k1_AXP_s111
                                 + 0.901295 w_k1_DIS_s111
                                 + 0.976111 w_k1_FDX_s111
                                 + 1.10715 w_k1_IBM_s111
                                 + 0.916818 w_k1_INTC_s111
                                 + 1.07395 w_k1_KO_s111 + 1.38491 w_k1_MAT_s111
                                 + 0.881473 w_k1_MSFT_s111
                                 + 1.04846 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.927882 w_k1_ADM_s111
                                 + 1.05083 w_k1_AXP_s111
                                 + 0.840308 w_k1_DIS_s111
                                 + 0.909959 w_k1_FDX_s111
                                 + 1.38225 w_k1_IBM_s111
                                 + 0.99124 w_k1_INTC_s111
                                 + 0.963591 w_k1_KO_s111
                                 + 1.51484 w_k1_MAT_s111
                                 + 0.860476 w_k1_MSFT_s111
                                 + 1.01695 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.28931 w_k1_ADM_s111
                                 + 1.17277 w_k1_AXP_s111
                                 + 0.904113 w_k1_DIS_s111
                                 + 1.06332 w_k1_FDX_s111
                                 + 0.761368 w_k1_IBM_s111
                                 + 1.03541 w_k1_INTC_s111
                                 + 0.841047 w_k1_KO_s111 + 1.0292 w_k1_MAT_s111
                                 + 1.01244 w_k1_MSFT_s111
                                 + 0.991521 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.23027 w_k1_ADM_s111
                                 + 1.04448 w_k1_AXP_s111
                                 + 1.17107 w_k1_DIS_s111
                                 + 1.36392 w_k1_FDX_s111
                                 + 0.707845 w_k1_IBM_s111
                                 + 1.11677 w_k1_INTC_s111
                                 + 0.925186 w_k1_KO_s111
                                 + 0.820853 w_k1_MAT_s111
                                 + 1.15955 w_k1_MSFT_s111
                                 + 1.08518 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.713739 w_k1_ADM_s111
                                 + 0.658185 w_k1_AXP_s111
                                 + 0.554052 w_k1_DIS_s111
                                 + 0.773646 w_k1_FDX_s111
                                 + 0.585739 w_k1_IBM_s111
                                 + 0.878927 w_k1_INTC_s111
                                 + 0.599639 w_k1_KO_s111
                                 + 0.535928 w_k1_MAT_s111
                                 + 0.73559 w_k1_MSFT_s111
                                 + 0.722392 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.904577 w_k1_ADM_s111
                                 + 0.836556 w_k1_AXP_s111
                                 + 0.888749 w_k1_DIS_s111
                                 + 0.947813 w_k1_FDX_s111
                                 + 0.897651 w_k1_IBM_s111
                                 + 0.89507 w_k1_INTC_s111
                                 + 0.941451 w_k1_KO_s111 + 1.2664 w_k1_MAT_s111
                                 + 1.00661 w_k1_MSFT_s111
                                 + 0.977201 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.45311 w_k1_ADM_s111
                                 + 1.78719 w_k1_AXP_s111
                                 + 1.59136 w_k1_DIS_s111
                                 + 1.45842 w_k1_FDX_s111 + 1.2291 w_k1_IBM_s111
                                 + 1.40681 w_k1_INTC_s111 + 1.3965 w_k1_KO_s111
                                 + 1.25878 w_k1_MAT_s111 + 1.3 w_k1_MSFT_s111
                                 + 1.24047 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.1411 w_k1_ADM_s112
                                 + 1.15031 w_k1_AXP_s112
                                 + 1.03893 w_k1_DIS_s112
                                 + 1.05388 w_k1_FDX_s112
                                 + 0.690119 w_k1_IBM_s112
                                 + 1.12663 w_k1_INTC_s112
                                 + 1.01926 w_k1_KO_s112
                                 + 0.690751 w_k1_MAT_s112
                                 + 1.07556 w_k1_MSFT_s112
                                 + 1.10312 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.11701 w_k1_ADM_s112
                                 + 1.20956 w_k1_AXP_s112
                                 + 1.26381 w_k1_DIS_s112
                                 + 1.11983 w_k1_FDX_s112
                                 + 1.18271 w_k1_IBM_s112
                                 + 1.21361 w_k1_INTC_s112
                                 + 1.20893 w_k1_KO_s112 + 1.14898 w_k1_MAT_s112
                                 + 1.16522 w_k1_MSFT_s112
                                 + 1.08625 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.862661 w_k1_ADM_s112
                                 + 0.868381 w_k1_AXP_s112
                                 + 0.943068 w_k1_DIS_s112
                                 + 0.905067 w_k1_FDX_s112
                                 + 0.896672 w_k1_IBM_s112
                                 + 0.966555 w_k1_INTC_s112
                                 + 0.992565 w_k1_KO_s112
                                 + 0.659084 w_k1_MAT_s112
                                 + 0.978886 w_k1_MSFT_s112
                                 + 1.00243 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.09777 w_k1_ADM_s112
                                 + 1.51284 w_k1_AXP_s112
                                 + 1.50425 w_k1_DIS_s112
                                 + 1.30322 w_k1_FDX_s112
                                 + 1.15124 w_k1_IBM_s112
                                 + 1.3279 w_k1_INTC_s112 + 1.30338 w_k1_KO_s112
                                 + 1.39614 w_k1_MAT_s112
                                 + 1.26741 w_k1_MSFT_s112
                                 + 1.15286 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.0497 w_k1_ADM_s112
                                 + 1.28746 w_k1_AXP_s112
                                 + 0.85893 w_k1_DIS_s112
                                 + 0.987533 w_k1_FDX_s112
                                 + 1.00493 w_k1_IBM_s112
                                 + 0.950866 w_k1_INTC_s112
                                 + 0.969243 w_k1_KO_s112
                                 + 0.956258 w_k1_MAT_s112
                                 + 0.924902 w_k1_MSFT_s112
                                 + 1.01752 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.47187 w_k1_ADM_s112
                                 + 1.811 w_k1_AXP_s112 + 1.9012 w_k1_DIS_s112
                                 + 1.37695 w_k1_FDX_s112
                                 + 1.35477 w_k1_IBM_s112
                                 + 1.26863 w_k1_INTC_s112
                                 + 1.42722 w_k1_KO_s112 + 1.60165 w_k1_MAT_s112
                                 + 1.43298 w_k1_MSFT_s112
                                 + 1.43238 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.828538 w_k1_ADM_s112
                                 + 0.796239 w_k1_AXP_s112
                                 + 0.645895 w_k1_DIS_s112
                                 + 0.683429 w_k1_FDX_s112
                                 + 0.744925 w_k1_IBM_s112
                                 + 0.897467 w_k1_INTC_s112
                                 + 0.812783 w_k1_KO_s112
                                 + 0.694509 w_k1_MAT_s112
                                 + 0.801205 w_k1_MSFT_s112
                                 + 0.776318 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.09178 w_k1_ADM_s112
                                 + 1.15733 w_k1_AXP_s112
                                 + 1.35045 w_k1_DIS_s112
                                 + 1.02462 w_k1_FDX_s112
                                 + 1.43667 w_k1_IBM_s112
                                 + 1.04522 w_k1_INTC_s112 + 1.3488 w_k1_KO_s112
                                 + 1.54747 w_k1_MAT_s112
                                 + 1.12672 w_k1_MSFT_s112 + 1.1219 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.17826 w_k1_ADM_s113
                                 + 1.6304 w_k1_AXP_s113 + 1.90204 w_k1_DIS_s113
                                 + 1.29257 w_k1_FDX_s113
                                 + 1.57241 w_k1_IBM_s113
                                 + 1.38207 w_k1_INTC_s113
                                 + 1.59866 w_k1_KO_s113 + 1.60183 w_k1_MAT_s113
                                 + 1.22033 w_k1_MSFT_s113
                                 + 1.33766 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.924035 w_k1_ADM_s113
                                 + 1.19684 w_k1_AXP_s113
                                 + 1.21905 w_k1_DIS_s113
                                 + 1.05062 w_k1_FDX_s113
                                 + 1.15973 w_k1_IBM_s113
                                 + 1.34881 w_k1_INTC_s113 + 1.1665 w_k1_KO_s113
                                 + 1.09118 w_k1_MAT_s113
                                 + 1.18131 w_k1_MSFT_s113
                                 + 1.19653 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.10608 w_k1_ADM_s113
                                 + 1.01161 w_k1_AXP_s113
                                 + 1.07938 w_k1_DIS_s113
                                 + 0.912663 w_k1_FDX_s113
                                 + 1.4785 w_k1_IBM_s113
                                 + 1.09821 w_k1_INTC_s113
                                 + 1.09094 w_k1_KO_s113 + 1.16041 w_k1_MAT_s113
                                 + 0.917343 w_k1_MSFT_s113
                                 + 1.12435 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.85365 w_k1_ADM_s113
                                 + 0.642114 w_k1_AXP_s113
                                 + 0.54819 w_k1_DIS_s113
                                 + 0.644131 w_k1_FDX_s113
                                 + 0.788641 w_k1_IBM_s113
                                 + 0.852707 w_k1_INTC_s113
                                 + 0.749869 w_k1_KO_s113
                                 + 0.653385 w_k1_MAT_s113
                                 + 0.734282 w_k1_MSFT_s113
                                 + 0.833062 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.31444 w_k1_ADM_s113
                                 + 1.23111 w_k1_AXP_s113
                                 + 0.938823 w_k1_DIS_s113
                                 + 1.05128 w_k1_FDX_s113
                                 + 0.828066 w_k1_IBM_s113
                                 + 1.20117 w_k1_INTC_s113
                                 + 1.00554 w_k1_KO_s113 + 0.82739 w_k1_MAT_s113
                                 + 1.00262 w_k1_MSFT_s113
                                 + 1.09363 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.896029 w_k1_ADM_s113
                                 + 0.783856 w_k1_AXP_s113
                                 + 0.551432 w_k1_DIS_s113
                                 + 0.784437 w_k1_FDX_s113
                                 + 0.595795 w_k1_IBM_s113
                                 + 0.768578 w_k1_INTC_s113
                                 + 0.649477 w_k1_KO_s113
                                 + 0.45021 w_k1_MAT_s113
                                 + 0.691163 w_k1_MSFT_s113
                                 + 0.841151 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.978545 w_k1_ADM_s113
                                 + 0.786081 w_k1_AXP_s113
                                 + 0.940332 w_k1_DIS_s113
                                 + 0.928614 w_k1_FDX_s113
                                 + 0.760207 w_k1_IBM_s113
                                 + 0.832874 w_k1_INTC_s113
                                 + 1.01557 w_k1_KO_s113
                                 + 0.800084 w_k1_MAT_s113
                                 + 0.961945 w_k1_MSFT_s113
                                 + 1.00426 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.911872 w_k1_ADM_s113
                                 + 0.876421 w_k1_AXP_s113
                                 + 0.784363 w_k1_DIS_s113
                                 + 0.778748 w_k1_FDX_s113
                                 + 1.03887 w_k1_IBM_s113
                                 + 0.810012 w_k1_INTC_s113
                                 + 1.0106 w_k1_KO_s113 + 1.1368 w_k1_MAT_s113
                                 + 0.821866 w_k1_MSFT_s113
                                 + 0.903206 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.935742 w_k1_ADM_s114
                                 + 0.829578 w_k1_AXP_s114
                                 + 0.90095 w_k1_DIS_s114
                                 + 0.776705 w_k1_FDX_s114
                                 + 1.46536 w_k1_IBM_s114
                                 + 0.725773 w_k1_INTC_s114
                                 + 1.01342 w_k1_KO_s114 + 1.35685 w_k1_MAT_s114
                                 + 0.789568 w_k1_MSFT_s114
                                 + 0.986319 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.887898 w_k1_ADM_s114
                                 + 0.910337 w_k1_AXP_s114
                                 + 0.874331 w_k1_DIS_s114
                                 + 0.885604 w_k1_FDX_s114
                                 + 0.916265 w_k1_IBM_s114
                                 + 0.819941 w_k1_INTC_s114
                                 + 0.714263 w_k1_KO_s114
                                 + 0.760892 w_k1_MAT_s114
                                 + 0.79607 w_k1_MSFT_s114
                                 + 0.855598 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.938362 w_k1_ADM_s114
                                 + 1.03028 w_k1_AXP_s114
                                 + 1.22358 w_k1_DIS_s114 + 1.0535 w_k1_FDX_s114
                                 + 1.42944 w_k1_IBM_s114
                                 + 0.948615 w_k1_INTC_s114
                                 + 1.13115 w_k1_KO_s114 + 1.24847 w_k1_MAT_s114
                                 + 0.999958 w_k1_MSFT_s114
                                 + 1.11794 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.14874 w_k1_ADM_s114
                                 + 0.967329 w_k1_AXP_s114
                                 + 1.01805 w_k1_DIS_s114
                                 + 1.10509 w_k1_FDX_s114
                                 + 0.741751 w_k1_IBM_s114
                                 + 0.947015 w_k1_INTC_s114
                                 + 0.772521 w_k1_KO_s114
                                 + 0.849624 w_k1_MAT_s114
                                 + 1.00736 w_k1_MSFT_s114
                                 + 1.01894 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.19003 w_k1_ADM_s114
                                 + 1.24986 w_k1_AXP_s114
                                 + 1.28059 w_k1_DIS_s114
                                 + 1.15609 w_k1_FDX_s114
                                 + 1.17108 w_k1_IBM_s114
                                 + 1.11773 w_k1_INTC_s114
                                 + 1.18573 w_k1_KO_s114 + 1.53825 w_k1_MAT_s114
                                 + 1.04174 w_k1_MSFT_s114
                                 + 1.08767 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.857359 w_k1_ADM_s114
                                 + 0.847093 w_k1_AXP_s114
                                 + 0.683494 w_k1_DIS_s114
                                 + 0.824442 w_k1_FDX_s114
                                 + 1.10052 w_k1_IBM_s114
                                 + 0.797139 w_k1_INTC_s114
                                 + 0.814751 w_k1_KO_s114
                                 + 1.34907 w_k1_MAT_s114
                                 + 0.759194 w_k1_MSFT_s114
                                 + 0.965974 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.778762 w_k1_ADM_s114
                                 + 0.7806 w_k1_AXP_s114
                                 + 0.772549 w_k1_DIS_s114
                                 + 0.820952 w_k1_FDX_s114
                                 + 0.873664 w_k1_IBM_s114
                                 + 0.886272 w_k1_INTC_s114
                                 + 0.776826 w_k1_KO_s114
                                 + 0.667045 w_k1_MAT_s114
                                 + 0.810592 w_k1_MSFT_s114
                                 + 0.844847 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.21122 w_k1_ADM_s114
                                 + 1.40501 w_k1_AXP_s114 + 1.621 w_k1_DIS_s114
                                 + 1.44504 w_k1_FDX_s114
                                 + 0.942571 w_k1_IBM_s114
                                 + 1.12808 w_k1_INTC_s114
                                 + 1.22099 w_k1_KO_s114 + 1.40264 w_k1_MAT_s114
                                 + 1.17548 w_k1_MSFT_s114
                                 + 1.28362 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.649296 w_k1_ADM_s115
                                 + 0.802397 w_k1_AXP_s115
                                 + 0.693867 w_k1_DIS_s115
                                 + 0.709773 w_k1_FDX_s115
                                 + 1.1263 w_k1_IBM_s115
                                 + 1.02731 w_k1_INTC_s115
                                 + 1.02808 w_k1_KO_s115
                                 + 0.900631 w_k1_MAT_s115
                                 + 0.786773 w_k1_MSFT_s115
                                 + 1.04077 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.615618 w_k1_ADM_s115
                                 + 0.646754 w_k1_AXP_s115
                                 + 0.65882 w_k1_DIS_s115
                                 + 0.776864 w_k1_FDX_s115
                                 + 0.546718 w_k1_IBM_s115
                                 + 0.899375 w_k1_INTC_s115
                                 + 0.730852 w_k1_KO_s115
                                 + 0.329278 w_k1_MAT_s115
                                 + 0.825434 w_k1_MSFT_s115
                                 + 0.802992 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.732892 w_k1_ADM_s115
                                 + 0.792617 w_k1_AXP_s115
                                 + 0.822458 w_k1_DIS_s115
                                 + 0.857449 w_k1_FDX_s115
                                 + 0.771159 w_k1_IBM_s115
                                 + 1.04558 w_k1_INTC_s115
                                 + 0.783507 w_k1_KO_s115
                                 + 0.643622 w_k1_MAT_s115
                                 + 0.843218 w_k1_MSFT_s115
                                 + 0.947307 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.06017 w_k1_ADM_s115
                                 + 1.599 w_k1_AXP_s115 + 1.78945 w_k1_DIS_s115
                                 + 1.38095 w_k1_FDX_s115 + 1.0547 w_k1_IBM_s115
                                 + 1.4193 w_k1_INTC_s115 + 1.35535 w_k1_KO_s115
                                 + 0.961018 w_k1_MAT_s115
                                 + 1.307 w_k1_MSFT_s115 + 1.23787 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.863081 w_k1_ADM_s115
                                 + 1.08405 w_k1_AXP_s115
                                 + 1.11967 w_k1_DIS_s115
                                 + 1.03474 w_k1_FDX_s115
                                 + 1.52737 w_k1_IBM_s115
                                 + 1.3563 w_k1_INTC_s115 + 1.21135 w_k1_KO_s115
                                 + 0.968209 w_k1_MAT_s115
                                 + 1.05748 w_k1_MSFT_s115
                                 + 1.17076 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.896287 w_k1_ADM_s115
                                 + 1.14643 w_k1_AXP_s115
                                 + 1.29407 w_k1_DIS_s115
                                 + 1.02703 w_k1_FDX_s115
                                 + 1.51371 w_k1_IBM_s115
                                 + 1.00273 w_k1_INTC_s115
                                 + 1.45188 w_k1_KO_s115 + 1.67254 w_k1_MAT_s115
                                 + 1.07454 w_k1_MSFT_s115
                                 + 1.21638 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.863631 w_k1_ADM_s115
                                 + 0.892408 w_k1_AXP_s115
                                 + 1.07485 w_k1_DIS_s115
                                 + 0.961718 w_k1_FDX_s115
                                 + 1.06489 w_k1_IBM_s115
                                 + 1.20104 w_k1_INTC_s115
                                 + 1.14365 w_k1_KO_s115
                                 + 0.801367 w_k1_MAT_s115
                                 + 0.986288 w_k1_MSFT_s115
                                 + 1.10443 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.831138 w_k1_ADM_s115
                                 + 1.03129 w_k1_AXP_s115
                                 + 0.934926 w_k1_DIS_s115
                                 + 1.13624 w_k1_FDX_s115
                                 + 0.902827 w_k1_IBM_s115
                                 + 1.14099 w_k1_INTC_s115 + 1.0781 w_k1_KO_s115
                                 + 0.686396 w_k1_MAT_s115
                                 + 0.958396 w_k1_MSFT_s115
                                 + 1.04862 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.916321 w_k1_ADM_s116
                                 + 1.19124 w_k1_AXP_s116
                                 + 1.03795 w_k1_DIS_s116
                                 + 1.10216 w_k1_FDX_s116
                                 + 0.810467 w_k1_IBM_s116
                                 + 1.32151 w_k1_INTC_s116
                                 + 1.02251 w_k1_KO_s116 + 1.21032 w_k1_MAT_s116
                                 + 0.965996 w_k1_MSFT_s116
                                 + 1.03623 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.12342 w_k1_ADM_s116
                                 + 1.23559 w_k1_AXP_s116
                                 + 1.60404 w_k1_DIS_s116
                                 + 1.20315 w_k1_FDX_s116
                                 + 1.49283 w_k1_IBM_s116
                                 + 1.2887 w_k1_INTC_s116 + 1.50976 w_k1_KO_s116
                                 + 1.83348 w_k1_MAT_s116
                                 + 1.23236 w_k1_MSFT_s116
                                 + 1.22996 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.738185 w_k1_ADM_s116
                                 + 0.850858 w_k1_AXP_s116
                                 + 0.763734 w_k1_DIS_s116
                                 + 0.802032 w_k1_FDX_s116
                                 + 0.974942 w_k1_IBM_s116
                                 + 1.23784 w_k1_INTC_s116
                                 + 0.930903 w_k1_KO_s116
                                 + 1.14832 w_k1_MAT_s116
                                 + 0.839108 w_k1_MSFT_s116
                                 + 0.926749 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.881097 w_k1_ADM_s116
                                 + 0.778348 w_k1_AXP_s116
                                 + 0.823607 w_k1_DIS_s116
                                 + 0.876631 w_k1_FDX_s116
                                 + 0.736517 w_k1_IBM_s116
                                 + 0.95214 w_k1_INTC_s116
                                 + 0.784392 w_k1_KO_s116
                                 + 0.740501 w_k1_MAT_s116
                                 + 0.907568 w_k1_MSFT_s116
                                 + 0.838642 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.03678 w_k1_ADM_s116
                                 + 1.17119 w_k1_AXP_s116
                                 + 1.43441 w_k1_DIS_s116
                                 + 1.25606 w_k1_FDX_s116
                                 + 1.06416 w_k1_IBM_s116
                                 + 1.02799 w_k1_INTC_s116
                                 + 1.22772 w_k1_KO_s116 + 1.39659 w_k1_MAT_s116
                                 + 1.17168 w_k1_MSFT_s116
                                 + 1.13461 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.05476 w_k1_ADM_s116
                                 + 0.951829 w_k1_AXP_s116
                                 + 1.02666 w_k1_DIS_s116
                                 + 0.955981 w_k1_FDX_s116
                                 + 1.13035 w_k1_IBM_s116
                                 + 1.25744 w_k1_INTC_s116
                                 + 1.16287 w_k1_KO_s116 + 1.02491 w_k1_MAT_s116
                                 + 0.90732 w_k1_MSFT_s116
                                 + 1.10328 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.17483 w_k1_ADM_s116
                                 + 1.57024 w_k1_AXP_s116
                                 + 1.08248 w_k1_DIS_s116
                                 + 1.22766 w_k1_FDX_s116
                                 + 1.26658 w_k1_IBM_s116
                                 + 1.32302 w_k1_INTC_s116 + 1.2752 w_k1_KO_s116
                                 + 1.99296 w_k1_MAT_s116
                                 + 1.11046 w_k1_MSFT_s116
                                 + 1.18502 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.934649 w_k1_ADM_s116
                                 + 1.15522 w_k1_AXP_s116
                                 + 1.02649 w_k1_DIS_s116
                                 + 0.878683 w_k1_FDX_s116
                                 + 1.47151 w_k1_IBM_s116
                                 + 1.02231 w_k1_INTC_s116
                                 + 1.15853 w_k1_KO_s116 + 1.87875 w_k1_MAT_s116
                                 + 0.865555 w_k1_MSFT_s116
                                 + 1.01518 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.07715 w_k1_ADM_s117
                                 + 0.949647 w_k1_AXP_s117
                                 + 0.757978 w_k1_DIS_s117
                                 + 0.899621 w_k1_FDX_s117
                                 + 1.05544 w_k1_IBM_s117
                                 + 0.823852 w_k1_INTC_s117
                                 + 1.12581 w_k1_KO_s117 + 1.01552 w_k1_MAT_s117
                                 + 0.770616 w_k1_MSFT_s117
                                 + 0.896812 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.64779 w_k1_ADM_s117
                                 + 1.82592 w_k1_AXP_s117
                                 + 1.63742 w_k1_DIS_s117 + 1.4811 w_k1_FDX_s117
                                 + 1.24881 w_k1_IBM_s117
                                 + 1.14655 w_k1_INTC_s117
                                 + 1.67403 w_k1_KO_s117 + 1.83856 w_k1_MAT_s117
                                 + 1.37893 w_k1_MSFT_s117 + 1.2737 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.14648 w_k1_ADM_s117
                                 + 1.56179 w_k1_AXP_s117
                                 + 1.02637 w_k1_DIS_s117 + 1.1967 w_k1_FDX_s117
                                 + 1.09908 w_k1_IBM_s117
                                 + 0.922393 w_k1_INTC_s117 + 1.166 w_k1_KO_s117
                                 + 1.43216 w_k1_MAT_s117
                                 + 0.974005 w_k1_MSFT_s117
                                 + 0.875128 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.989571 w_k1_ADM_s117
                                 + 1.29003 w_k1_AXP_s117
                                 + 1.32725 w_k1_DIS_s117
                                 + 1.17703 w_k1_FDX_s117
                                 + 1.28706 w_k1_IBM_s117
                                 + 1.0175 w_k1_INTC_s117 + 1.42052 w_k1_KO_s117
                                 + 1.34682 w_k1_MAT_s117
                                 + 1.16233 w_k1_MSFT_s117
                                 + 1.07965 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.736091 w_k1_ADM_s117
                                 + 1.18457 w_k1_AXP_s117
                                 + 0.938945 w_k1_DIS_s117
                                 + 0.952325 w_k1_FDX_s117
                                 + 1.46561 w_k1_IBM_s117
                                 + 0.995458 w_k1_INTC_s117
                                 + 1.17535 w_k1_KO_s117 + 1.36861 w_k1_MAT_s117
                                 + 0.896478 w_k1_MSFT_s117
                                 + 1.03771 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.05791 w_k1_ADM_s117
                                 + 0.933264 w_k1_AXP_s117
                                 + 1.04281 w_k1_DIS_s117
                                 + 1.15138 w_k1_FDX_s117
                                 + 0.738096 w_k1_IBM_s117
                                 + 0.961178 w_k1_INTC_s117
                                 + 1.00653 w_k1_KO_s117 + 1.16417 w_k1_MAT_s117
                                 + 1.01635 w_k1_MSFT_s117
                                 + 0.966193 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.908769 w_k1_ADM_s117
                                 + 0.822562 w_k1_AXP_s117
                                 + 0.701983 w_k1_DIS_s117
                                 + 0.813224 w_k1_FDX_s117
                                 + 0.89752 w_k1_IBM_s117
                                 + 0.711751 w_k1_INTC_s117
                                 + 0.854551 w_k1_KO_s117
                                 + 1.02903 w_k1_MAT_s117
                                 + 0.838994 w_k1_MSFT_s117
                                 + 0.732935 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.747281 w_k1_ADM_s117
                                 + 0.820608 w_k1_AXP_s117
                                 + 0.731615 w_k1_DIS_s117
                                 + 0.927635 w_k1_FDX_s117
                                 + 0.713307 w_k1_IBM_s117
                                 + 0.746819 w_k1_INTC_s117
                                 + 0.774514 w_k1_KO_s117
                                 + 0.870114 w_k1_MAT_s117
                                 + 0.89767 w_k1_MSFT_s117
                                 + 0.818486 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0416466719766076 w_k1_ADM_s000
                                 + 0.0401241369095001 w_k1_ADM_s001
                                 + 0.115583060893006 w_k1_ADM_s002
                                 + 0.0462436806637562 w_k1_ADM_s003
                                 + 0.0343210518989611 w_k1_ADM_s004
                                 + 0.0640533557783748 w_k1_ADM_s005
                                 + 0.0810047939658156 w_k1_ADM_s006
                                 + 0.0275855118449793 w_k1_ADM_s007
                                 + 0.0191675572684088 w_k1_ADM_s010
                                 + 0.00857565975016028 w_k1_ADM_s011
                                 + 0.0135307565050468 w_k1_ADM_s012
                                 + 0.00916775045654921 w_k1_ADM_s013
                                 + 0.00866119972850736 w_k1_ADM_s014
                                 + 0.0164790120643539 w_k1_ADM_s015
                                 + 0.0081217597765376 w_k1_ADM_s016
                                 + 0.0103890826201682 w_k1_ADM_s017
                                 + 0.018870898099373 w_k1_ADM_s100
                                 + 0.0294704349155147 w_k1_ADM_s101
                                 + 0.0139625397371273 w_k1_ADM_s102
                                 + 0.0383173126153935 w_k1_ADM_s103
                                 + 0.00839881012020519 w_k1_ADM_s104
                                 + 0.0112259778959871 w_k1_ADM_s105
                                 + 0.00909039635154437 w_k1_ADM_s106
                                 + 0.0387682706026445 w_k1_ADM_s107
                                 + 0.0323443531346339 w_k1_ADM_s110
                                 + 0.0675049437574334 w_k1_ADM_s111
                                 + 0.0160913062985784 w_k1_ADM_s112
                                 + 0.0359975145694835 w_k1_ADM_s113
                                 + 0.0434905790412144 w_k1_ADM_s114
                                 + 0.0334333279284904 w_k1_ADM_s115
                                 + 0.0279841357809632 w_k1_ADM_s116
                                 + 0.0336331610605062 w_k1_ADM_s117
                                 + 0.0478829536395496 w_k1_AXP_s000
                                 + 0.0361460618736514 w_k1_AXP_s001
                                 + 0.119965144838743 w_k1_AXP_s002
                                 + 0.0470502451872503 w_k1_AXP_s003
                                 + 0.0364958205104424 w_k1_AXP_s004
                                 + 0.0613392510825733 w_k1_AXP_s005
                                 + 0.0824612182228426 w_k1_AXP_s006
                                 + 0.0242049925309439 w_k1_AXP_s007
                                 + 0.0199952824637245 w_k1_AXP_s010
                                 + 0.00923517278918615 w_k1_AXP_s011
                                 + 0.0170407483380982 w_k1_AXP_s012
                                 + 0.00898429021233472 w_k1_AXP_s013
                                 + 0.00831441727366911 w_k1_AXP_s014
                                 + 0.0143524375212678 w_k1_AXP_s015
                                 + 0.0102017657620891 w_k1_AXP_s016
                                 + 0.00960465508719911 w_k1_AXP_s017
                                 + 0.0183309890864894 w_k1_AXP_s100
                                 + 0.0289227693534838 w_k1_AXP_s101
                                 + 0.016170854872985 w_k1_AXP_s102
                                 + 0.0412275854967837 w_k1_AXP_s103
                                 + 0.00847328618998112 w_k1_AXP_s104
                                 + 0.0128269738740816 w_k1_AXP_s105
                                 + 0.011421699533984 w_k1_AXP_s106
                                 + 0.0355060008920097 w_k1_AXP_s107
                                 + 0.034006880512375 w_k1_AXP_s110
                                 + 0.0688844806106464 w_k1_AXP_s111
                                 + 0.0179218207277988 w_k1_AXP_s112
                                 + 0.0362743560708964 w_k1_AXP_s113
                                 + 0.0435256520012133 w_k1_AXP_s114
                                 + 0.0405522468468054 w_k1_AXP_s115
                                 + 0.0311250795863478 w_k1_AXP_s116
                                 + 0.0367650120206217 w_k1_AXP_s117
                                 + 0.0479519000574896 w_k1_DIS_s000
                                 + 0.03936766403872 w_k1_DIS_s001
                                 + 0.121208952635875 w_k1_DIS_s002
                                 + 0.0498593763070492 w_k1_DIS_s003
                                 + 0.0341123340838848 w_k1_DIS_s004
                                 + 0.0596596444532567 w_k1_DIS_s005
                                 + 0.0813239273408409 w_k1_DIS_s006
                                 + 0.0251663474917961 w_k1_DIS_s007
                                 + 0.0199116516093402 w_k1_DIS_s010
                                 + 0.0097117719686644 w_k1_DIS_s011
                                 + 0.0170199230710748 w_k1_DIS_s012
                                 + 0.00870950586025178 w_k1_DIS_s013
                                 + 0.00844267528392313 w_k1_DIS_s014
                                 + 0.0144503761275385 w_k1_DIS_s015
                                 + 0.00926246186453068 w_k1_DIS_s016
                                 + 0.010132498799108 w_k1_DIS_s017
                                 + 0.0177512825171004 w_k1_DIS_s100
                                 + 0.0284223217572022 w_k1_DIS_s101
                                 + 0.016043436808187 w_k1_DIS_s102
                                 + 0.0400386625622278 w_k1_DIS_s103
                                 + 0.00817420742773028 w_k1_DIS_s104
                                 + 0.012276199842356 w_k1_DIS_s105
                                 + 0.0109393118399373 w_k1_DIS_s106
                                 + 0.0420665483587689 w_k1_DIS_s107
                                 + 0.0358659635196671 w_k1_DIS_s110
                                 + 0.0620493991781334 w_k1_DIS_s111
                                 + 0.0168339006652069 w_k1_DIS_s112
                                 + 0.0360950006803662 w_k1_DIS_s113
                                 + 0.0457334777914664 w_k1_DIS_s114
                                 + 0.0427552093436748 w_k1_DIS_s115
                                 + 0.0309762263892432 w_k1_DIS_s116
                                 + 0.0332250214124444 w_k1_DIS_s117
                                 + 0.046022237803539 w_k1_FDX_s000
                                 + 0.0368590071155257 w_k1_FDX_s001
                                 + 0.114913055752964 w_k1_FDX_s002
                                 + 0.0511393903045754 w_k1_FDX_s003
                                 + 0.0346861015622865 w_k1_FDX_s004
                                 + 0.0602611179621885 w_k1_FDX_s005
                                 + 0.0802083586816367 w_k1_FDX_s006
                                 + 0.0244192083144573 w_k1_FDX_s007
                                 + 0.0211752975998946 w_k1_FDX_s010
                                 + 0.00899387505158754 w_k1_FDX_s011
                                 + 0.0162875692110624 w_k1_FDX_s012
                                 + 0.00895059617970458 w_k1_FDX_s013
                                 + 0.00831621250410326 w_k1_FDX_s014
                                 + 0.013508477716797 w_k1_FDX_s015
                                 + 0.00944526908454447 w_k1_FDX_s016
                                 + 0.0102494494976536 w_k1_FDX_s017
                                 + 0.0181471221540294 w_k1_FDX_s100
                                 + 0.0257621041457376 w_k1_FDX_s101
                                 + 0.0161990761325981 w_k1_FDX_s102
                                 + 0.0407448332831189 w_k1_FDX_s103
                                 + 0.00837770902736839 w_k1_FDX_s104
                                 + 0.0125987313975284 w_k1_FDX_s105
                                 + 0.0113396260736274 w_k1_FDX_s106
                                 + 0.0376307084487097 w_k1_FDX_s107
                                 + 0.034801706036481 w_k1_FDX_s110
                                 + 0.0659751261842825 w_k1_FDX_s111
                                 + 0.0157321476596953 w_k1_FDX_s112
                                 + 0.0336245005851513 w_k1_FDX_s113
                                 + 0.0441589123324203 w_k1_FDX_s114
                                 + 0.040470411010526 w_k1_FDX_s115
                                 + 0.0298789356688891 w_k1_FDX_s116
                                 + 0.0354539054354721 w_k1_FDX_s117
                                 + 0.050400887928444 w_k1_IBM_s000
                                 + 0.0381269132037409 w_k1_IBM_s001
                                 + 0.124167005610108 w_k1_IBM_s002
                                 + 0.0507548643002244 w_k1_IBM_s003
                                 + 0.0348433905489688 w_k1_IBM_s004
                                 + 0.0572726317370918 w_k1_IBM_s005
                                 + 0.0830582832896977 w_k1_IBM_s006
                                 + 0.0249369106511872 w_k1_IBM_s007
                                 + 0.0211562746020545 w_k1_IBM_s010
                                 + 0.00920850974530772 w_k1_IBM_s011
                                 + 0.0169599100025196 w_k1_IBM_s012
                                 + 0.00885879139255869 w_k1_IBM_s013
                                 + 0.00849437772972437 w_k1_IBM_s014
                                 + 0.0146207443634658 w_k1_IBM_s015
                                 + 0.00955336870050983 w_k1_IBM_s016
                                 + 0.0101464406045684 w_k1_IBM_s017
                                 + 0.0185533391095927 w_k1_IBM_s100
                                 + 0.0267604894827301 w_k1_IBM_s101
                                 + 0.0160254838744125 w_k1_IBM_s102
                                 + 0.0423314951001067 w_k1_IBM_s103
                                 + 0.00850538786250766 w_k1_IBM_s104
                                 + 0.0118050842798407 w_k1_IBM_s105
                                 + 0.0123339338659915 w_k1_IBM_s106
                                 + 0.0382851084642521 w_k1_IBM_s107
                                 + 0.034669765972379 w_k1_IBM_s110
                                 + 0.0645809619368549 w_k1_IBM_s111
                                 + 0.015760176004078 w_k1_IBM_s112
                                 + 0.0354433409778445 w_k1_IBM_s113
                                 + 0.0452795773508951 w_k1_IBM_s114
                                 + 0.0422312524209853 w_k1_IBM_s115
                                 + 0.0305815443933212 w_k1_IBM_s116
                                 + 0.034700113386739 w_k1_IBM_s117
                                 + 0.0418689542141888 w_k1_INTC_s000
                                 + 0.0356582325491868 w_k1_INTC_s001
                                 + 0.117227025001405 w_k1_INTC_s002
                                 + 0.0457340327061494 w_k1_INTC_s003
                                 + 0.0361938263721299 w_k1_INTC_s004
                                 + 0.0510751151250902 w_k1_INTC_s005
                                 + 0.0824196436318098 w_k1_INTC_s006
                                 + 0.0208500252815425 w_k1_INTC_s007
                                 + 0.0203538816402136 w_k1_INTC_s010
                                 + 0.00962341310088201 w_k1_INTC_s011
                                 + 0.0181524924161382 w_k1_INTC_s012
                                 + 0.00894890847581112 w_k1_INTC_s013
                                 + 0.00661990493535846 w_k1_INTC_s014
                                 + 0.0133157456488777 w_k1_INTC_s015
                                 + 0.00871241718426967 w_k1_INTC_s016
                                 + 0.00889144547220391 w_k1_INTC_s017
                                 + 0.0199421066313153 w_k1_INTC_s100
                                 + 0.0312288374253431 w_k1_INTC_s101
                                 + 0.0160130149316559 w_k1_INTC_s102
                                 + 0.0434539345518671 w_k1_INTC_s103
                                 + 0.00738082398142716 w_k1_INTC_s104
                                 + 0.0118244818028988 w_k1_INTC_s105
                                 + 0.0125820496036121 w_k1_INTC_s106
                                 + 0.0341166964529417 w_k1_INTC_s107
                                 + 0.0357672318810608 w_k1_INTC_s110
                                 + 0.0636010550597843 w_k1_INTC_s111
                                 + 0.0167893158239971 w_k1_INTC_s112
                                 + 0.0372004174197185 w_k1_INTC_s113
                                 + 0.040601401920271 w_k1_INTC_s114
                                 + 0.0461358587575628 w_k1_INTC_s115
                                 + 0.0340738126734536 w_k1_INTC_s116
                                 + 0.0306962912290358 w_k1_INTC_s117
                                 + 0.0483523605522785 w_k1_KO_s000
                                 + 0.0383941662707233 w_k1_KO_s001
                                 + 0.110511009597222 w_k1_KO_s002
                                 + 0.050850690214733 w_k1_KO_s003
                                 + 0.0347152260657536 w_k1_KO_s004
                                 + 0.0578597234614146 w_k1_KO_s005
                                 + 0.0790414929863383 w_k1_KO_s006
                                 + 0.0217850547162054 w_k1_KO_s007
                                 + 0.0191651408919964 w_k1_KO_s010
                                 + 0.00908460864372433 w_k1_KO_s011
                                 + 0.0164879410799752 w_k1_KO_s012
                                 + 0.0084108200453061 w_k1_KO_s013
                                 + 0.00723332290527884 w_k1_KO_s014
                                 + 0.0157232103262737 w_k1_KO_s015
                                 + 0.00826226293067869 w_k1_KO_s016
                                 + 0.0100594973526859 w_k1_KO_s017
                                 + 0.016522595766152 w_k1_KO_s100
                                 + 0.0288442917883721 w_k1_KO_s101
                                 + 0.0140046540495298 w_k1_KO_s102
                                 + 0.0429810306528152 w_k1_KO_s103
                                 + 0.00787606496536746 w_k1_KO_s104
                                 + 0.0115164197893226 w_k1_KO_s105
                                 + 0.0113783358832106 w_k1_KO_s106
                                 + 0.0351268636540233 w_k1_KO_s107
                                 + 0.034766690071024 w_k1_KO_s110
                                 + 0.0627852335637654 w_k1_KO_s111
                                 + 0.016894462496078 w_k1_KO_s112
                                 + 0.0371345376019588 w_k1_KO_s113
                                 + 0.0406916666848571 w_k1_KO_s114
                                 + 0.0441612599731267 w_k1_KO_s115
                                 + 0.0315824252889691 w_k1_KO_s116
                                 + 0.0372184339672034 w_k1_KO_s117
                                 + 0.0583654786974796 w_k1_MAT_s000
                                 + 0.0358199065547654 w_k1_MAT_s001
                                 + 0.104503646929072 w_k1_MAT_s002
                                 + 0.0435127921349818 w_k1_MAT_s003
                                 + 0.0338000585723099 w_k1_MAT_s004
                                 + 0.0627852917485668 w_k1_MAT_s005
                                 + 0.0875998670999321 w_k1_MAT_s006
                                 + 0.0251069054386165 w_k1_MAT_s007
                                 + 0.0195716927851931 w_k1_MAT_s010
                                 + 0.0085751411044024 w_k1_MAT_s011
                                 + 0.0166797081166494 w_k1_MAT_s012
                                 + 0.00889149530376121 w_k1_MAT_s013
                                 + 0.00862423490208547 w_k1_MAT_s014
                                 + 0.0126238927365037 w_k1_MAT_s015
                                 + 0.0103222621473719 w_k1_MAT_s016
                                 + 0.00975653042408803 w_k1_MAT_s017
                                 + 0.0169934782795723 w_k1_MAT_s100
                                 + 0.028301833129695 w_k1_MAT_s101
                                 + 0.0157568232977979 w_k1_MAT_s102
                                 + 0.0498010007155539 w_k1_MAT_s103
                                 + 0.00859441084961554 w_k1_MAT_s104
                                 + 0.0123751368197564 w_k1_MAT_s105
                                 + 0.0103783929429633 w_k1_MAT_s106
                                 + 0.0344356355956594 w_k1_MAT_s107
                                 + 0.0337533245251583 w_k1_MAT_s110
                                 + 0.0801157619670244 w_k1_MAT_s111
                                 + 0.0153179935755051 w_k1_MAT_s112
                                 + 0.0338767812580402 w_k1_MAT_s113
                                 + 0.0470586266923033 w_k1_MAT_s114
                                 + 0.0353515704344594 w_k1_MAT_s115
                                 + 0.0377535377777925 w_k1_MAT_s116
                                 + 0.0409604114071524 w_k1_MAT_s117
                                 + 0.0471013301623009 w_k1_MSFT_s000
                                 + 0.0387282802660204 w_k1_MSFT_s001
                                 + 0.116610640844097 w_k1_MSFT_s002
                                 + 0.0494115128913775 w_k1_MSFT_s003
                                 + 0.0358623270325266 w_k1_MSFT_s004
                                 + 0.0593746813137811 w_k1_MSFT_s005
                                 + 0.0848210943353591 w_k1_MSFT_s006
                                 + 0.0216224979088208 w_k1_MSFT_s007
                                 + 0.0192128432790662 w_k1_MSFT_s010
                                 + 0.00843367687180283 w_k1_MSFT_s011
                                 + 0.0163064195209434 w_k1_MSFT_s012
                                 + 0.00891053869335186 w_k1_MSFT_s013
                                 + 0.00825869545681189 w_k1_MSFT_s014
                                 + 0.015212474696671 w_k1_MSFT_s015
                                 + 0.00933335480304629 w_k1_MSFT_s016
                                 + 0.00938857319042729 w_k1_MSFT_s017
                                 + 0.0192613629867148 w_k1_MSFT_s100
                                 + 0.0269521096975411 w_k1_MSFT_s101
                                 + 0.0153210407024059 w_k1_MSFT_s102
                                 + 0.0397637723948119 w_k1_MSFT_s103
                                 + 0.00815878676249089 w_k1_MSFT_s104
                                 + 0.0124924252788866 w_k1_MSFT_s105
                                 + 0.0120356243426279 w_k1_MSFT_s106
                                 + 0.0387828687417412 w_k1_MSFT_s107
                                 + 0.034206495151143 w_k1_MSFT_s110
                                 + 0.0626652342367549 w_k1_MSFT_s111
                                 + 0.0164328472263739 w_k1_MSFT_s112
                                 + 0.0343759768516111 w_k1_MSFT_s113
                                 + 0.0405283853059567 w_k1_MSFT_s114
                                 + 0.0396906319482642 w_k1_MSFT_s115
                                 + 0.0288751093231531 w_k1_MSFT_s116
                                 + 0.0328396280866851 w_k1_MSFT_s117
                                 + 0.0481741970454186 w_k1_PG_s000
                                 + 0.0411086565556439 w_k1_PG_s001
                                 + 0.116253178266608 w_k1_PG_s002
                                 + 0.0491534456658202 w_k1_PG_s003
                                 + 0.0346819344003183 w_k1_PG_s004
                                 + 0.0583936421266335 w_k1_PG_s005
                                 + 0.0744305489932042 w_k1_PG_s006
                                 + 0.0230947400223182 w_k1_PG_s007
                                 + 0.0203247597049359 w_k1_PG_s010
                                 + 0.00809064321130322 w_k1_PG_s011
                                 + 0.0167555041377829 w_k1_PG_s012
                                 + 0.00870020868197563 w_k1_PG_s013
                                 + 0.00833379882080662 w_k1_PG_s014
                                 + 0.0137655814554592 w_k1_PG_s015
                                 + 0.00919687871811333 w_k1_PG_s016
                                 + 0.0103621871171307 w_k1_PG_s017
                                 + 0.0195866278693366 w_k1_PG_s100
                                 + 0.0261296143540295 w_k1_PG_s101
                                 + 0.0154065087258796 w_k1_PG_s102
                                 + 0.0401311904819196 w_k1_PG_s103
                                 + 0.00790603349773028 w_k1_PG_s104
                                 + 0.0115672770548901 w_k1_PG_s105
                                 + 0.0116436378992427 w_k1_PG_s106
                                 + 0.0400246317514926 w_k1_PG_s107
                                 + 0.0352937832222235 w_k1_PG_s110
                                 + 0.0649958953241881 w_k1_PG_s111
                                 + 0.0163406215408955 w_k1_PG_s112
                                 + 0.0373143407958602 w_k1_PG_s113
                                 + 0.0437293813873466 w_k1_PG_s114
                                 + 0.0435325733139436 w_k1_PG_s115
                                 + 0.0302151239036234 w_k1_PG_s116
                                 + 0.0319402617415647 w_k1_PG_s117 >= 1.12
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
