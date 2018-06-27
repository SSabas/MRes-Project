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
 w_k1_ADM_s0_return_constraint:  1.15401 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.894579 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.775735 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.0173 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.00139 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.952344 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.99523 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.12752 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.18725 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.801786 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.15102 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.802176 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.34577 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.30219 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.08728 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.908002 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.740985 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.2207 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.63098 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.15681 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.818831 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.844954 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.961142 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.84693 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.905844 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.08093 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.5124 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.08143 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.05205 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.67025 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.43927 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.33715 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.837036 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.773145 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.11661 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.837975 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.12214 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.837968 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.17615 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.888022 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.70812 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.06839 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.27636 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.929875 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.844538 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.14026 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.2262 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.878589 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.01755 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.751634 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.50767 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.21989 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.917398 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.22202 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.964399 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.31548 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.83926 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.28417 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.723092 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.793538 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.12559 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.634127 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.730156 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.780757 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.16601 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.00482 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.819568 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.56391 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.61673 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.43192 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.562462 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.676295 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.997996 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.91746 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.957529 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.815945 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.1599 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.824744 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.687788 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.18142 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.35481 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.985072 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.04828 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.37435 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.54922 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.879874 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.898106 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.704928 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.27934 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.33249 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.05541 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.27299 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.897289 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.24331 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.93966 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.16331 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.639841 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.774222 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.924418 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.578334 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.616826 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.998021 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.22348 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.07625 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.824997 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.89645 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.81125 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.42426 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.574879 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.473229 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.936691 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.979226 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.10141 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.784462 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.10899 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.870174 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.76488 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.09323 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.14422 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.957536 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.03224 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.14439 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.25288 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.833647 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.934101 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.863007 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.24976 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.33845 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.09089 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.04052 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.884037 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.20669 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.59451 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.27982 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.830256 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.81887 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.930225 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.635605 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.771758 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.774232 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.20517 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.03225 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.867317 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.44555 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.64848 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.36266 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.774934 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.638678 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.07413 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.901985 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.08939 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.819688 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.02012 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.05794 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.787548 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.13083 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.48798 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.833019 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.854524 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.07126 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.81323 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.41883 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.24224 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.697247 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.27824 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.758054 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.760976 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.85411 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.23006 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.25817 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.37466 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.696642 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.566727 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.893117 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.24777 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.78776 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.61732 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.31569 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.729903 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.971483 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.764227 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.27019 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.04067 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.53028 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.564872 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.893438 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.77934 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.46739 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.09001 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.980677 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09722 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.899208 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.87091 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.12462 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.28149 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.95381 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.851098 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06043 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.14399 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.827286 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.883059 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.816372 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.0609 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.02073 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.01221 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.915441 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.938912 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.07731 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.29031 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.08736 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.806705 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.846806 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.930229 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.651938 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.721882 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.642301 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.0594 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.936743 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.821005 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.05247 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.2911 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.33017 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.896312 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.739439 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.05042 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.966154 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.16129 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.991417 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.07836 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.991756 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.747852 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.09844 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22181 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.880907 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.00925 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.16624 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.69121 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.03344 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.07498 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.790594 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.21349 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.03636 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.919733 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.37475 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.03798 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.19662 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.50527 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.95666 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.588607 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.884223 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.14459 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.738896 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.678514 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.07547 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.05347 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.992076 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.876089 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.6895 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.3186 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.26657 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.606674 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.763458 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.864471 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.13929 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.06997 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.932512 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.05395 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.03809 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.718885 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.20928 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.65233 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.868933 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.879655 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.11363 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.74272 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.30051 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.32474 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.850093 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.41153 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.836195 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.664137 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.91103 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.2205 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.4132 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.17785 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.697112 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.566979 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.787433 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.3948 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.637981 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.670372 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.10915 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.744288 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.976615 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.601629 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.48297 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.06659 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.49641 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.365387 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.776399 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.70968 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.37825 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.916404 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.868968 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.0852 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.915165 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.790517 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.05707 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.16036 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.944315 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.03499 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.24662 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.30298 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.948559 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.962455 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.826007 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.30108 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.14866 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.09996 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.01441 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.992791 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.09702 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.4383 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.15756 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.907708 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.875302 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.849755 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.680157 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.722772 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.815862 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.17853 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.01496 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.843565 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.43107 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.44673 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.23925 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.757223 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.702217 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.06111 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.92155 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.1256 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.839197 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.07655 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.928551 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.834678 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.07572 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.21185 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.930772 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.928897 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.11093 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.22207 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.996818 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.97324 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.84607 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.18725 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.09242 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.00942 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.11262 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.921018 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.13814 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.33293 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.009 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.826379 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.911609 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.974894 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.753267 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.776676 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.892058 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.01128 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.964464 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.81684 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.25755 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.33617 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.28633 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.821912 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.861351 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.06141 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.10417 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.06452 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.939394 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.276001570325309 z_k1_s0000
                                 - 0.444002413566471 z_k1_s0001
                                 - 0.157000562944949 z_k1_s0002
                                 - 0.572000589362019 z_k1_s0003
                                 - 0.395002059704149 z_k1_s0004
                                 - 0.136000522289994 z_k1_s0005
                                 - 0.335001943547134 z_k1_s0006
                                 - 0.275001827390119 z_k1_s0007
                                 - 0.145999552080373 z_k1_s0010
                                 - 0.300999319380732 z_k1_s0011
                                 - 0.114999598620302 z_k1_s0012
                                 - 0.2509984901575 z_k1_s0013
                                 - 0.101999745801182 z_k1_s0014
                                 - 0.45499897115692 z_k1_s0015
                                 - 0.129999682487762 z_k1_s0016
                                 - 0.148999568853865 z_k1_s0017
                                 - 0.128000141558735 z_k1_s0020
                                 - 0.131000141628077 z_k1_s0021
                                 - 0.0720001402643503 z_k1_s0022
                                 - 0.218000005038853 z_k1_s0023
                                 - 0.151000696490924 z_k1_s0024
                                 - 0.227000698247589 z_k1_s0025
                                 - 0.339000700836357 z_k1_s0026
                                 - 0.120000141373823 z_k1_s0027
                                 - 0.183000663130789 z_k1_s0030
                                 - 0.096000302502876 z_k1_s0031
                                 - 0.298001765428521 z_k1_s0032
                                 - 0.274000075422085 z_k1_s0033
                                 - 0.214000462922329 z_k1_s0034
                                 - 0.34000149417835 z_k1_s0035
                                 - 0.196000579172402 z_k1_s0036
                                 - 0.244000269172207 z_k1_s0037
                                 - 0.403000592541849 z_k1_s0040
                                 - 0.201999418400401 z_k1_s0041
                                 - 0.0950000052438254 z_k1_s0042
                                 - 0.091000060406205 z_k1_s0043
                                 - 0.194000666325456 z_k1_s0044
                                 - 0.209999592475716 z_k1_s0045
                                 - 0.132999978901348 z_k1_s0046
                                 - 0.0940000545844295 z_k1_s0047
                                 - 0.330000519037537 z_k1_s0050
                                 - 0.689000105119681 z_k1_s0051
                                 - 0.96600265576572 z_k1_s0052
                                 - 0.650002600681795 z_k1_s0053
                                 - 0.4610005988129 z_k1_s0054
                                 - 1.13100338878503 z_k1_s0055
                                 - 0.266000292108901 z_k1_s0056
                                 - 0.242000443760932 z_k1_s0057
                                 - 0.19899970008243 z_k1_s0060
                                 - 0.1759997281935 z_k1_s0061
                                 - 0.0680000076021315 z_k1_s0062
                                 - 0.0770000929848395 z_k1_s0063
                                 - 0.127000060221668 z_k1_s0064
                                 - 0.322999775309899 z_k1_s0065
                                 - 0.251000657049588 z_k1_s0066
                                 - 0.0830000629732364 z_k1_s0067
                                 - 0.157000481221252 z_k1_s0070
                                 - 0.0970000819644739 z_k1_s0071
                                 - 0.111000083124283 z_k1_s0072
                                 - 0.14199986390646 z_k1_s0073
                                 - 0.134999805826489 z_k1_s0074
                                 - 0.139000660444564 z_k1_s0075
                                 - 0.27900067204265 z_k1_s0076
                                 - 0.0900001388846359 z_k1_s0077
                                 - 0.370000715088751 z_k1_s0100
                                 - 0.415000827490153 z_k1_s0101
                                 - 0.474001058501103 z_k1_s0102
                                 - 0.620001757742096 z_k1_s0103
                                 - 0.650003714581366 z_k1_s0104
                                 - 1.30200224846457 z_k1_s0105
                                 - 1.17100284129404 z_k1_s0106
                                 - 1.27100100006938 z_k1_s0107
                                 - 0.324999810395491 z_k1_s0110
                                 - 0.276999959098225 z_k1_s0111
                                 - 0.425001049047028 z_k1_s0112
                                 - 0.52699909857988 z_k1_s0113
                                 - 1.04800053842095 z_k1_s0114
                                 - 0.262999916444917 z_k1_s0115
                                 - 0.31699983517928 z_k1_s0116
                                 - 0.258999756786934 z_k1_s0117
                                 - 0.76699901619055 z_k1_s0120
                                 - 0.548001514828143 z_k1_s0121
                                 - 0.996005764586592 z_k1_s0122
                                 - 0.789004126821882 z_k1_s0123
                                 - 1.51700706579325 z_k1_s0124
                                 - 0.55500164073391 z_k1_s0125
                                 - 0.413001081422501 z_k1_s0126
                                 - 0.762003913200398 z_k1_s0127
                                 - 0.705998311117683 z_k1_s0130
                                 - 0.371998347382769 z_k1_s0131
                                 - 0.248999227233967 z_k1_s0132
                                 - 0.26999907701547 z_k1_s0133
                                 - 0.775996129729112 z_k1_s0134
                                 - 0.802995936591044 z_k1_s0135
                                 - 0.920998453826648 z_k1_s0136
                                 - 0.944998282148366 z_k1_s0137
                                 - 1.09799802646216 z_k1_s0140
                                 - 0.392000767177618 z_k1_s0141
                                 - 0.494998730028058 z_k1_s0142
                                 - 0.226000154323856 z_k1_s0143
                                 - 0.571000417276643 z_k1_s0144
                                 - 0.362999970753829 z_k1_s0145
                                 - 0.381000065154837 z_k1_s0146
                                 - 0.339999978996974 z_k1_s0147
                                 - 0.653000091845075 z_k1_s0150
                                 - 0.294999757887009 z_k1_s0151
                                 - 0.316999661664396 z_k1_s0152
                                 - 0.418999805122808 z_k1_s0153
                                 - 0.369001006446293 z_k1_s0154
                                 - 0.212999854498737 z_k1_s0155
                                 - 0.858997880670544 z_k1_s0156
                                 - 0.477998891980769 z_k1_s0157
                                 - 0.644000707490075 z_k1_s0160
                                 - 0.477000544017365 z_k1_s0161
                                 - 0.347000777879809 z_k1_s0162
                                 - 0.413000461081689 z_k1_s0163
                                 - 1.24700249471349 z_k1_s0164
                                 - 0.907002689368559 z_k1_s0165
                                 - 1.30499835604333 z_k1_s0166
                                 - 1.43599893871535 z_k1_s0167
                                 - 1.56199387048332 z_k1_s0170
                                 - 1.12499711378774 z_k1_s0171
                                 - 1.55399783218277 z_k1_s0172
                                 - 1.42899815287302 z_k1_s0173
                                 - 0.994000405752579 z_k1_s0174
                                 - 1.15299272181868 z_k1_s0175
                                 - 0.85799764719845 z_k1_s0176
                                 - 0.798998442794823 z_k1_s0177
                                 - 0.15600015928564 z_k1_s1000
                                 - 0.237000117116585 z_k1_s1001
                                 - 0.517000692847359 z_k1_s1002
                                 - 0.264999862039512 z_k1_s1003
                                 - 0.124000175945019 z_k1_s1004
                                 - 0.456000243604071 z_k1_s1005
                                 - 0.508999254011512 z_k1_s1006
                                 - 0.141000167094724 z_k1_s1007
                                 - 0.0669999537974116 z_k1_s1010
                                 - 0.0809995871986003 z_k1_s1011
                                 - 0.107999709264628 z_k1_s1012
                                 - 0.0800001308624574 z_k1_s1013
                                 - 0.169999499083239 z_k1_s1014
                                 - 0.0629999472542887 z_k1_s1015
                                 - 0.156000099381895 z_k1_s1016
                                 - 0.0539999325322622 z_k1_s1017
                                 - 0.124999531919728 z_k1_s1020
                                 - 0.0680002267077849 z_k1_s1021
                                 - 0.157999680410227 z_k1_s1022
                                 - 0.0829996996586874 z_k1_s1023
                                 - 0.0489999133044233 z_k1_s1024
                                 - 0.0419999630608918 z_k1_s1025
                                 - 0.0569999592111997 z_k1_s1026
                                 - 0.0720000861613589 z_k1_s1027
                                 - 0.167998842454175 z_k1_s1030
                                 - 0.156999026816857 z_k1_s1031
                                 - 0.18199912599049 z_k1_s1032
                                 - 0.0699997076884733 z_k1_s1033
                                 - 0.224998612571899 z_k1_s1034
                                 - 0.0559997661507786 z_k1_s1035
                                 - 0.0609997859855054 z_k1_s1036
                                 - 0.220999508700439 z_k1_s1037
                                 - 0.130000377823736 z_k1_s1040
                                 - 0.127000184812541 z_k1_s1041
                                 - 0.175999707331129 z_k1_s1042
                                 - 0.11900039611554 z_k1_s1043
                                 - 0.060999967863584 z_k1_s1044
                                 - 0.200000162421409 z_k1_s1045
                                 - 0.2140003371408 z_k1_s1046
                                 - 0.061999995900673 z_k1_s1047
                                 - 0.145000025077191 z_k1_s1050
                                 - 0.383999676494218 z_k1_s1051
                                 - 0.189999859563278 z_k1_s1052
                                 - 0.374000830074687 z_k1_s1053
                                 - 0.185000017553662 z_k1_s1054
                                 - 0.486000473968928 z_k1_s1055
                                 - 0.121999956113246 z_k1_s1056
                                 - 0.207999877117682 z_k1_s1057
                                 - 0.302998686189141 z_k1_s1060
                                 - 0.133999679914483 z_k1_s1061
                                 - 0.0879996867725319 z_k1_s1062
                                 - 0.17999904555847 z_k1_s1063
                                 - 0.0719997437229807 z_k1_s1064
                                 - 0.0819996767540521 z_k1_s1065
                                 - 0.284999534630849 z_k1_s1066
                                 - 0.110999620593711 z_k1_s1067
                                 - 0.233999907859106 z_k1_s1070
                                 - 0.202999254959142 z_k1_s1071
                                 - 0.179999886868554 z_k1_s1072
                                 - 0.393999970053334 z_k1_s1073
                                 - 0.322999301604813 z_k1_s1074
                                 - 0.21100072286847 z_k1_s1075
                                 - 0.134000052087666 z_k1_s1076
                                 - 0.151999875984564 z_k1_s1077
                                 - 0.129999942759961 z_k1_s1100
                                 - 0.322999598409078 z_k1_s1101
                                 - 0.411999266630239 z_k1_s1102
                                 - 0.17499957158085 z_k1_s1103
                                 - 0.108000006109793 z_k1_s1104
                                 - 0.205000542194313 z_k1_s1105
                                 - 0.161999926114721 z_k1_s1106
                                 - 0.146000017487309 z_k1_s1107
                                 - 0.528000412227276 z_k1_s1110
                                 - 0.26199964410023 z_k1_s1111
                                 - 0.377998016644408 z_k1_s1112
                                 - 0.736999855193639 z_k1_s1113
                                 - 0.340999200323853 z_k1_s1114
                                 - 0.368999463126347 z_k1_s1115
                                 - 0.473998720139055 z_k1_s1116
                                 - 0.368000392067292 z_k1_s1117
                                 - 0.271000274206577 z_k1_s1120
                                 - 0.321000776820468 z_k1_s1121
                                 - 0.263000087268323 z_k1_s1122
                                 - 0.269000360622052 z_k1_s1123
                                 - 0.273000187791101 z_k1_s1124
                                 - 0.165000060825375 z_k1_s1125
                                 - 0.598000791780619 z_k1_s1126
                                 - 0.503000902014536 z_k1_s1127
                                 - 0.146999938388992 z_k1_s1130
                                 - 0.134999982775612 z_k1_s1131
                                 - 0.311000984772145 z_k1_s1132
                                 - 0.425000149849183 z_k1_s1133
                                 - 0.347000576112237 z_k1_s1134
                                 - 0.657998805883896 z_k1_s1135
                                 - 0.540999445278476 z_k1_s1136
                                 - 0.191000051138101 z_k1_s1137
                                 - 0.410998821365597 z_k1_s1140
                                 - 0.90599676909493 z_k1_s1141
                                 - 0.312999277709172 z_k1_s1142
                                 - 0.980997396309384 z_k1_s1143
                                 - 0.539998886956738 z_k1_s1144
                                 - 0.321999465554653 z_k1_s1145
                                 - 0.923998270583358 z_k1_s1146
                                 - 1.23199957044026 z_k1_s1147
                                 - 0.864999135730581 z_k1_s1150
                                 - 0.496997591982028 z_k1_s1151
                                 - 0.299999657223384 z_k1_s1152
                                 - 0.538998492147951 z_k1_s1153
                                 - 0.196999893755853 z_k1_s1154
                                 - 0.283999743416707 z_k1_s1155
                                 - 0.386999506884238 z_k1_s1156
                                 - 0.848998046525578 z_k1_s1157
                                 - 0.191999939806465 z_k1_s1160
                                 - 0.607000674757664 z_k1_s1161
                                 - 0.666000799399623 z_k1_s1162
                                 - 0.301999544361544 z_k1_s1163
                                 - 0.389999466985386 z_k1_s1164
                                 - 0.587997799265408 z_k1_s1165
                                 - 0.485001497158042 z_k1_s1166
                                 - 0.752997405247842 z_k1_s1167
                                 - 0.320000694644698 z_k1_s1170
                                 - 0.179000230111541 z_k1_s1171
                                 - 0.294000775375108 z_k1_s1172
                                 - 0.259000592973118 z_k1_s1173
                                 - 0.866000512909309 z_k1_s1174
                                 - 0.539001673788234 z_k1_s1175
                                 - 0.625999802725383 z_k1_s1176
                                 - 0.70100262616339 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.886469 w_k1_ADM_s000
                                 + 1.02961 w_k1_AXP_s000 + 1.1835 w_k1_DIS_s000
                                 + 1.08746 w_k1_FDX_s000
                                 + 1.00283 w_k1_IBM_s000
                                 + 1.26156 w_k1_INTC_s000
                                 + 1.05525 w_k1_KO_s000 + 1.08136 w_k1_MAT_s000
                                 + 1.20491 w_k1_MSFT_s000
                                 + 1.04806 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.08734 w_k1_ADM_s000
                                 + 1.46467 w_k1_AXP_s000
                                 + 1.03816 w_k1_DIS_s000
                                 + 1.13013 w_k1_FDX_s000
                                 + 1.08502 w_k1_IBM_s000
                                 + 1.28487 w_k1_INTC_s000
                                 + 0.967629 w_k1_KO_s000
                                 + 0.840624 w_k1_MAT_s000
                                 + 1.02886 w_k1_MSFT_s000
                                 + 1.07414 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.15193 w_k1_ADM_s000
                                 + 1.39367 w_k1_AXP_s000
                                 + 0.916594 w_k1_DIS_s000
                                 + 0.973258 w_k1_FDX_s000
                                 + 1.13873 w_k1_IBM_s000
                                 + 0.976517 w_k1_INTC_s000
                                 + 0.898208 w_k1_KO_s000
                                 + 1.43112 w_k1_MAT_s000
                                 + 0.918541 w_k1_MSFT_s000
                                 + 1.11321 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.902952 w_k1_ADM_s000
                                 + 0.813589 w_k1_AXP_s000
                                 + 0.786737 w_k1_DIS_s000
                                 + 0.846713 w_k1_FDX_s000
                                 + 0.91475 w_k1_IBM_s000
                                 + 0.941862 w_k1_INTC_s000
                                 + 0.786701 w_k1_KO_s000
                                 + 0.906503 w_k1_MAT_s000
                                 + 0.928367 w_k1_MSFT_s000
                                 + 0.860717 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.887155 w_k1_ADM_s000
                                 + 1.00677 w_k1_AXP_s000
                                 + 0.924577 w_k1_DIS_s000
                                 + 0.933523 w_k1_FDX_s000
                                 + 1.10608 w_k1_IBM_s000
                                 + 1.08782 w_k1_INTC_s000
                                 + 1.02249 w_k1_KO_s000 + 1.26025 w_k1_MAT_s000
                                 + 1.00185 w_k1_MSFT_s000
                                 + 1.09789 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.770372 w_k1_ADM_s000
                                 + 0.610629 w_k1_AXP_s000
                                 + 0.670672 w_k1_DIS_s000
                                 + 0.688413 w_k1_FDX_s000
                                 + 1.04246 w_k1_IBM_s000
                                 + 0.803317 w_k1_INTC_s000
                                 + 0.905217 w_k1_KO_s000
                                 + 1.03453 w_k1_MAT_s000
                                 + 0.864342 w_k1_MSFT_s000
                                 + 0.901657 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.36571 w_k1_ADM_s000
                                 + 1.28584 w_k1_AXP_s000
                                 + 1.26842 w_k1_DIS_s000
                                 + 1.08279 w_k1_FDX_s000
                                 + 1.58344 w_k1_IBM_s000
                                 + 1.18021 w_k1_INTC_s000
                                 + 1.33465 w_k1_KO_s000 + 1.515 w_k1_MAT_s000
                                 + 1.10942 w_k1_MSFT_s000
                                 + 1.26653 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.21714 w_k1_ADM_s000
                                 + 1.28352 w_k1_AXP_s000
                                 + 1.23237 w_k1_DIS_s000
                                 + 1.29256 w_k1_FDX_s000
                                 + 0.877797 w_k1_IBM_s000
                                 + 1.32656 w_k1_INTC_s000 + 1.0988 w_k1_KO_s000
                                 + 1.14817 w_k1_MAT_s000
                                 + 1.2169 w_k1_MSFT_s000 + 1.12427 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.11097 w_k1_ADM_s001
                                 + 0.968177 w_k1_AXP_s001
                                 + 0.961579 w_k1_DIS_s001
                                 + 1.15376 w_k1_FDX_s001
                                 + 0.830239 w_k1_IBM_s001
                                 + 1.00521 w_k1_INTC_s001
                                 + 1.02455 w_k1_KO_s001
                                 + 0.679477 w_k1_MAT_s001
                                 + 0.941095 w_k1_MSFT_s001
                                 + 0.938705 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.15531 w_k1_ADM_s001
                                 + 0.925754 w_k1_AXP_s001
                                 + 1.33696 w_k1_DIS_s001
                                 + 1.17152 w_k1_FDX_s001
                                 + 1.28334 w_k1_IBM_s001
                                 + 1.20279 w_k1_INTC_s001
                                 + 1.42252 w_k1_KO_s001 + 1.01752 w_k1_MAT_s001
                                 + 1.18484 w_k1_MSFT_s001
                                 + 1.03652 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.17957 w_k1_ADM_s001
                                 + 1.09322 w_k1_AXP_s001
                                 + 1.12689 w_k1_DIS_s001
                                 + 1.10267 w_k1_FDX_s001
                                 + 1.22955 w_k1_IBM_s001
                                 + 1.00905 w_k1_INTC_s001
                                 + 1.20774 w_k1_KO_s001 + 1.09632 w_k1_MAT_s001
                                 + 1.01289 w_k1_MSFT_s001
                                 + 1.02616 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.768732 w_k1_ADM_s001
                                 + 0.761932 w_k1_AXP_s001
                                 + 0.794053 w_k1_DIS_s001
                                 + 0.918152 w_k1_FDX_s001
                                 + 0.945971 w_k1_IBM_s001
                                 + 1.0506 w_k1_INTC_s001 + 1.03 w_k1_KO_s001
                                 + 0.917652 w_k1_MAT_s001
                                 + 1.06326 w_k1_MSFT_s001
                                 + 0.951267 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.44645 w_k1_ADM_s001
                                 + 1.27033 w_k1_AXP_s001
                                 + 1.88453 w_k1_DIS_s001
                                 + 1.39288 w_k1_FDX_s001
                                 + 1.39191 w_k1_IBM_s001
                                 + 1.30129 w_k1_INTC_s001
                                 + 1.54283 w_k1_KO_s001 + 1.38883 w_k1_MAT_s001
                                 + 1.33569 w_k1_MSFT_s001
                                 + 1.28648 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.82464 w_k1_ADM_s001
                                 + 0.809398 w_k1_AXP_s001
                                 + 0.762653 w_k1_DIS_s001
                                 + 0.808163 w_k1_FDX_s001
                                 + 0.947633 w_k1_IBM_s001
                                 + 0.868798 w_k1_INTC_s001
                                 + 0.927908 w_k1_KO_s001
                                 + 0.707916 w_k1_MAT_s001
                                 + 0.785479 w_k1_MSFT_s001
                                 + 0.865671 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.32275 w_k1_ADM_s001
                                 + 1.09581 w_k1_AXP_s001
                                 + 1.33379 w_k1_DIS_s001
                                 + 1.26671 w_k1_FDX_s001
                                 + 0.890767 w_k1_IBM_s001
                                 + 1.22825 w_k1_INTC_s001
                                 + 1.13382 w_k1_KO_s001
                                 + 0.874868 w_k1_MAT_s001
                                 + 1.25878 w_k1_MSFT_s001
                                 + 1.08747 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.933088 w_k1_ADM_s001
                                 + 1.15146 w_k1_AXP_s001
                                 + 0.877459 w_k1_DIS_s001
                                 + 1.0338 w_k1_FDX_s001 + 1.14815 w_k1_IBM_s001
                                 + 1.00316 w_k1_INTC_s001
                                 + 1.05686 w_k1_KO_s001 + 1.13089 w_k1_MAT_s001
                                 + 0.871683 w_k1_MSFT_s001
                                 + 1.07256 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.781078 w_k1_ADM_s002
                                 + 0.596232 w_k1_AXP_s002
                                 + 0.729934 w_k1_DIS_s002
                                 + 0.792223 w_k1_FDX_s002
                                 + 0.776981 w_k1_IBM_s002
                                 + 0.822791 w_k1_INTC_s002
                                 + 0.771379 w_k1_KO_s002
                                 + 0.723054 w_k1_MAT_s002
                                 + 0.851499 w_k1_MSFT_s002
                                 + 0.878154 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.707776 w_k1_ADM_s002
                                 + 0.86261 w_k1_AXP_s002
                                 + 0.801606 w_k1_DIS_s002
                                 + 0.823281 w_k1_FDX_s002
                                 + 0.654009 w_k1_IBM_s002
                                 + 0.86974 w_k1_INTC_s002
                                 + 0.731561 w_k1_KO_s002
                                 + 0.779373 w_k1_MAT_s002
                                 + 0.855613 w_k1_MSFT_s002
                                 + 0.901078 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.23546 w_k1_ADM_s002
                                 + 1.60185 w_k1_AXP_s002
                                 + 1.10323 w_k1_DIS_s002
                                 + 1.12243 w_k1_FDX_s002
                                 + 1.69666 w_k1_IBM_s002
                                 + 0.929487 w_k1_INTC_s002
                                 + 1.09199 w_k1_KO_s002 + 2.12224 w_k1_MAT_s002
                                 + 0.923957 w_k1_MSFT_s002
                                 + 1.1246 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.89751 w_k1_ADM_s002
                                 + 1.16974 w_k1_AXP_s002
                                 + 0.963831 w_k1_DIS_s002
                                 + 0.927865 w_k1_FDX_s002
                                 + 1.02985 w_k1_IBM_s002
                                 + 1.05523 w_k1_INTC_s002
                                 + 0.973358 w_k1_KO_s002
                                 + 0.96099 w_k1_MAT_s002
                                 + 0.854949 w_k1_MSFT_s002
                                 + 0.918455 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.51812 w_k1_ADM_s002
                                 + 1.37156 w_k1_AXP_s002
                                 + 1.75684 w_k1_DIS_s002
                                 + 1.15456 w_k1_FDX_s002
                                 + 1.72115 w_k1_IBM_s002
                                 + 0.941578 w_k1_INTC_s002
                                 + 1.60201 w_k1_KO_s002 + 1.44242 w_k1_MAT_s002
                                 + 1.18491 w_k1_MSFT_s002
                                 + 1.19825 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.990407 w_k1_ADM_s002
                                 + 1.12341 w_k1_AXP_s002
                                 + 0.981698 w_k1_DIS_s002
                                 + 0.918841 w_k1_FDX_s002
                                 + 1.29592 w_k1_IBM_s002
                                 + 0.791814 w_k1_INTC_s002
                                 + 1.04694 w_k1_KO_s002 + 1.6722 w_k1_MAT_s002
                                 + 0.898621 w_k1_MSFT_s002
                                 + 1.01914 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.16293 w_k1_ADM_s002
                                 + 1.39994 w_k1_AXP_s002
                                 + 1.35652 w_k1_DIS_s002
                                 + 1.19891 w_k1_FDX_s002
                                 + 0.855066 w_k1_IBM_s002
                                 + 1.25856 w_k1_INTC_s002
                                 + 1.09492 w_k1_KO_s002 + 1.0792 w_k1_MAT_s002
                                 + 1.24919 w_k1_MSFT_s002
                                 + 1.12806 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.880105 w_k1_ADM_s002
                                 + 0.862249 w_k1_AXP_s002
                                 + 0.776017 w_k1_DIS_s002
                                 + 0.706748 w_k1_FDX_s002
                                 + 1.10395 w_k1_IBM_s002
                                 + 0.73331 w_k1_INTC_s002
                                 + 0.843763 w_k1_KO_s002
                                 + 1.10212 w_k1_MAT_s002
                                 + 0.720958 w_k1_MSFT_s002
                                 + 0.862035 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.15907 w_k1_ADM_s003
                                 + 1.36683 w_k1_AXP_s003
                                 + 1.39142 w_k1_DIS_s003
                                 + 1.20528 w_k1_FDX_s003
                                 + 1.07128 w_k1_IBM_s003
                                 + 1.11344 w_k1_INTC_s003
                                 + 1.26864 w_k1_KO_s003 + 1.49243 w_k1_MAT_s003
                                 + 1.15049 w_k1_MSFT_s003
                                 + 1.19192 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.695546 w_k1_ADM_s003
                                 + 0.798512 w_k1_AXP_s003
                                 + 0.764577 w_k1_DIS_s003
                                 + 0.781104 w_k1_FDX_s003
                                 + 1.16308 w_k1_IBM_s003
                                 + 1.00828 w_k1_INTC_s003
                                 + 0.97991 w_k1_KO_s003
                                 + 0.831335 w_k1_MAT_s003
                                 + 0.840072 w_k1_MSFT_s003
                                 + 0.827964 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.988102 w_k1_ADM_s003
                                 + 1.15977 w_k1_AXP_s003
                                 + 1.17614 w_k1_DIS_s003
                                 + 0.985795 w_k1_FDX_s003
                                 + 1.49917 w_k1_IBM_s003
                                 + 1.09378 w_k1_INTC_s003
                                 + 1.52393 w_k1_KO_s003 + 1.46631 w_k1_MAT_s003
                                 + 0.891145 w_k1_MSFT_s003
                                 + 1.04893 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.07403 w_k1_ADM_s003
                                 + 0.935513 w_k1_AXP_s003 + 0.998 w_k1_DIS_s003
                                 + 1.12322 w_k1_FDX_s003
                                 + 0.795549 w_k1_IBM_s003
                                 + 0.946547 w_k1_INTC_s003
                                 + 0.993482 w_k1_KO_s003
                                 + 1.10595 w_k1_MAT_s003
                                 + 1.03245 w_k1_MSFT_s003
                                 + 0.955387 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.4802 w_k1_ADM_s003
                                 + 1.80616 w_k1_AXP_s003
                                 + 1.65492 w_k1_DIS_s003
                                 + 1.34774 w_k1_FDX_s003 + 1.3206 w_k1_IBM_s003
                                 + 1.04982 w_k1_INTC_s003
                                 + 1.44928 w_k1_KO_s003 + 2.40547 w_k1_MAT_s003
                                 + 1.1777 w_k1_MSFT_s003 + 1.19796 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.850555 w_k1_ADM_s003
                                 + 1.06283 w_k1_AXP_s003
                                 + 0.820179 w_k1_DIS_s003
                                 + 0.925055 w_k1_FDX_s003
                                 + 1.02988 w_k1_IBM_s003
                                 + 1.08149 w_k1_INTC_s003
                                 + 1.05324 w_k1_KO_s003 + 1.15433 w_k1_MAT_s003
                                 + 0.860733 w_k1_MSFT_s003
                                 + 0.987896 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.932136 w_k1_ADM_s003
                                 + 0.618207 w_k1_AXP_s003
                                 + 0.546796 w_k1_DIS_s003
                                 + 0.736927 w_k1_FDX_s003
                                 + 0.643844 w_k1_IBM_s003
                                 + 0.912135 w_k1_INTC_s003
                                 + 0.767745 w_k1_KO_s003
                                 + 0.529076 w_k1_MAT_s003
                                 + 0.729393 w_k1_MSFT_s003
                                 + 0.851564 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.995667 w_k1_ADM_s003
                                 + 0.763954 w_k1_AXP_s003
                                 + 0.717674 w_k1_DIS_s003
                                 + 0.886342 w_k1_FDX_s003
                                 + 0.799447 w_k1_IBM_s003
                                 + 0.921151 w_k1_INTC_s003
                                 + 0.947725 w_k1_KO_s003
                                 + 0.929042 w_k1_MAT_s003
                                 + 0.823008 w_k1_MSFT_s003
                                 + 0.797376 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.834727 w_k1_ADM_s004
                                 + 1.10674 w_k1_AXP_s004 + 1.1506 w_k1_DIS_s004
                                 + 1.10112 w_k1_FDX_s004
                                 + 0.988274 w_k1_IBM_s004
                                 + 0.971092 w_k1_INTC_s004
                                 + 0.986401 w_k1_KO_s004
                                 + 1.10492 w_k1_MAT_s004
                                 + 0.954311 w_k1_MSFT_s004
                                 + 1.0061 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.870512 w_k1_ADM_s004
                                 + 0.958055 w_k1_AXP_s004
                                 + 1.0765 w_k1_DIS_s004 + 1.05442 w_k1_FDX_s004
                                 + 1.20342 w_k1_IBM_s004
                                 + 0.988276 w_k1_INTC_s004
                                 + 1.21494 w_k1_KO_s004 + 1.27627 w_k1_MAT_s004
                                 + 0.934979 w_k1_MSFT_s004
                                 + 1.05525 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.17179 w_k1_ADM_s004
                                 + 1.33833 w_k1_AXP_s004
                                 + 1.61796 w_k1_DIS_s004
                                 + 1.33774 w_k1_FDX_s004
                                 + 1.29303 w_k1_IBM_s004
                                 + 1.27557 w_k1_INTC_s004
                                 + 1.36146 w_k1_KO_s004 + 1.38399 w_k1_MAT_s004
                                 + 1.26459 w_k1_MSFT_s004
                                 + 1.15635 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.3276 w_k1_ADM_s004
                                 + 1.15041 w_k1_AXP_s004
                                 + 1.72174 w_k1_DIS_s004 + 1.4188 w_k1_FDX_s004
                                 + 1.05089 w_k1_IBM_s004
                                 + 1.30827 w_k1_INTC_s004
                                 + 1.46643 w_k1_KO_s004 + 1.11211 w_k1_MAT_s004
                                 + 1.45508 w_k1_MSFT_s004
                                 + 1.41054 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.751927 w_k1_ADM_s004
                                 + 0.914044 w_k1_AXP_s004
                                 + 0.911535 w_k1_DIS_s004
                                 + 1.06084 w_k1_FDX_s004
                                 + 0.76712 w_k1_IBM_s004
                                 + 0.923274 w_k1_INTC_s004
                                 + 0.751855 w_k1_KO_s004
                                 + 0.660221 w_k1_MAT_s004
                                 + 0.895518 w_k1_MSFT_s004
                                 + 0.863948 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.688453 w_k1_ADM_s004
                                 + 0.829895 w_k1_AXP_s004
                                 + 0.761349 w_k1_DIS_s004
                                 + 0.842095 w_k1_FDX_s004
                                 + 0.988032 w_k1_IBM_s004
                                 + 0.801 w_k1_INTC_s004 + 0.839779 w_k1_KO_s004
                                 + 0.888423 w_k1_MAT_s004
                                 + 0.741183 w_k1_MSFT_s004
                                 + 0.834595 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.729045 w_k1_ADM_s004
                                 + 0.654119 w_k1_AXP_s004
                                 + 0.74682 w_k1_DIS_s004
                                 + 0.785477 w_k1_FDX_s004
                                 + 0.670668 w_k1_IBM_s004
                                 + 0.837295 w_k1_INTC_s004
                                 + 0.695285 w_k1_KO_s004
                                 + 0.475942 w_k1_MAT_s004
                                 + 0.714786 w_k1_MSFT_s004
                                 + 0.768219 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.13403 w_k1_ADM_s004
                                 + 1.49288 w_k1_AXP_s004
                                 + 1.44594 w_k1_DIS_s004
                                 + 1.12053 w_k1_FDX_s004
                                 + 1.59603 w_k1_IBM_s004
                                 + 0.949093 w_k1_INTC_s004
                                 + 1.33511 w_k1_KO_s004 + 1.85807 w_k1_MAT_s004
                                 + 0.993637 w_k1_MSFT_s004
                                 + 1.20843 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.35233 w_k1_ADM_s005
                                 + 0.899064 w_k1_AXP_s005
                                 + 1.35097 w_k1_DIS_s005
                                 + 1.34819 w_k1_FDX_s005
                                 + 0.698041 w_k1_IBM_s005
                                 + 1.09957 w_k1_INTC_s005
                                 + 1.03428 w_k1_KO_s005
                                 + 0.553709 w_k1_MAT_s005
                                 + 1.4299 w_k1_MSFT_s005 + 1.06961 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.971267 w_k1_ADM_s005
                                 + 0.927623 w_k1_AXP_s005
                                 + 0.914763 w_k1_DIS_s005
                                 + 0.827083 w_k1_FDX_s005
                                 + 1.37709 w_k1_IBM_s005
                                 + 0.738362 w_k1_INTC_s005
                                 + 1.0345 w_k1_KO_s005 + 1.15387 w_k1_MAT_s005
                                 + 0.865747 w_k1_MSFT_s005
                                 + 1.0345 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.27923 w_k1_ADM_s005
                                 + 1.31814 w_k1_AXP_s005
                                 + 1.37378 w_k1_DIS_s005
                                 + 1.28471 w_k1_FDX_s005
                                 + 1.17523 w_k1_IBM_s005
                                 + 1.01094 w_k1_INTC_s005
                                 + 1.19355 w_k1_KO_s005 + 1.28288 w_k1_MAT_s005
                                 + 1.19313 w_k1_MSFT_s005
                                 + 1.17562 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.896183 w_k1_ADM_s005
                                 + 0.618592 w_k1_AXP_s005
                                 + 0.582423 w_k1_DIS_s005
                                 + 0.745912 w_k1_FDX_s005
                                 + 0.761212 w_k1_IBM_s005
                                 + 0.787368 w_k1_INTC_s005
                                 + 0.661356 w_k1_KO_s005
                                 + 0.39919 w_k1_MAT_s005
                                 + 0.799273 w_k1_MSFT_s005
                                 + 0.818517 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.889643 w_k1_ADM_s005
                                 + 0.764603 w_k1_AXP_s005
                                 + 0.810052 w_k1_DIS_s005
                                 + 0.977247 w_k1_FDX_s005
                                 + 0.534445 w_k1_IBM_s005
                                 + 0.914865 w_k1_INTC_s005
                                 + 0.744479 w_k1_KO_s005
                                 + 0.492386 w_k1_MAT_s005
                                 + 1.05722 w_k1_MSFT_s005
                                 + 0.939545 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.12974 w_k1_ADM_s005
                                 + 1.20073 w_k1_AXP_s005
                                 + 1.11803 w_k1_DIS_s005
                                 + 1.01032 w_k1_FDX_s005
                                 + 1.07098 w_k1_IBM_s005
                                 + 0.881291 w_k1_INTC_s005
                                 + 1.00092 w_k1_KO_s005 + 1.20451 w_k1_MAT_s005
                                 + 1.14257 w_k1_MSFT_s005
                                 + 1.04521 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.05834 w_k1_ADM_s005
                                 + 0.716396 w_k1_AXP_s005
                                 + 1.00195 w_k1_DIS_s005
                                 + 0.892173 w_k1_FDX_s005
                                 + 0.896113 w_k1_IBM_s005
                                 + 0.848381 w_k1_INTC_s005
                                 + 1.06625 w_k1_KO_s005
                                 + 0.478087 w_k1_MAT_s005
                                 + 1.12192 w_k1_MSFT_s005
                                 + 0.967494 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.17812 w_k1_ADM_s005
                                 + 1.85099 w_k1_AXP_s005
                                 + 1.65717 w_k1_DIS_s005
                                 + 1.66212 w_k1_FDX_s005
                                 + 0.988712 w_k1_IBM_s005
                                 + 1.28143 w_k1_INTC_s005
                                 + 1.18293 w_k1_KO_s005 + 1.19133 w_k1_MAT_s005
                                 + 1.44475 w_k1_MSFT_s005
                                 + 1.17502 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.99233 w_k1_ADM_s006
                                 + 1.44115 w_k1_AXP_s006
                                 + 1.35342 w_k1_DIS_s006
                                 + 1.31415 w_k1_FDX_s006
                                 + 1.06151 w_k1_IBM_s006
                                 + 1.13239 w_k1_INTC_s006
                                 + 1.27848 w_k1_KO_s006 + 1.96762 w_k1_MAT_s006
                                 + 1.07026 w_k1_MSFT_s006
                                 + 1.08201 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.949974 w_k1_ADM_s006
                                 + 1.02457 w_k1_AXP_s006
                                 + 0.834908 w_k1_DIS_s006
                                 + 0.805486 w_k1_FDX_s006
                                 + 1.13084 w_k1_IBM_s006
                                 + 0.738426 w_k1_INTC_s006
                                 + 1.02475 w_k1_KO_s006 + 1.43632 w_k1_MAT_s006
                                 + 0.681895 w_k1_MSFT_s006
                                 + 0.91755 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.705927 w_k1_ADM_s006
                                 + 0.635187 w_k1_AXP_s006
                                 + 0.53062 w_k1_DIS_s006
                                 + 0.665885 w_k1_FDX_s006
                                 + 0.734628 w_k1_IBM_s006
                                 + 0.795958 w_k1_INTC_s006
                                 + 0.744345 w_k1_KO_s006
                                 + 0.717434 w_k1_MAT_s006
                                 + 0.664912 w_k1_MSFT_s006
                                 + 0.706484 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.00024 w_k1_ADM_s006
                                 + 1.28149 w_k1_AXP_s006
                                 + 2.15596 w_k1_DIS_s006
                                 + 1.43817 w_k1_FDX_s006
                                 + 1.29788 w_k1_IBM_s006
                                 + 1.23474 w_k1_INTC_s006
                                 + 1.35331 w_k1_KO_s006 + 1.07774 w_k1_MAT_s006
                                 + 1.22903 w_k1_MSFT_s006 + 1.1585 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.790906 w_k1_ADM_s006
                                 + 0.870336 w_k1_AXP_s006
                                 + 0.842026 w_k1_DIS_s006
                                 + 1.01285 w_k1_FDX_s006
                                 + 0.585635 w_k1_IBM_s006
                                 + 0.939014 w_k1_INTC_s006
                                 + 0.817988 w_k1_KO_s006
                                 + 0.801868 w_k1_MAT_s006
                                 + 0.870255 w_k1_MSFT_s006
                                 + 0.884111 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.760554 w_k1_ADM_s006
                                 + 0.938157 w_k1_AXP_s006
                                 + 0.957658 w_k1_DIS_s006
                                 + 0.97654 w_k1_FDX_s006
                                 + 0.880688 w_k1_IBM_s006
                                 + 1.19153 w_k1_INTC_s006
                                 + 1.01429 w_k1_KO_s006
                                 + 0.948426 w_k1_MAT_s006
                                 + 0.907479 w_k1_MSFT_s006
                                 + 0.903112 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.775008 w_k1_ADM_s006
                                 + 1.02452 w_k1_AXP_s006
                                 + 1.15572 w_k1_DIS_s006
                                 + 0.949709 w_k1_FDX_s006
                                 + 1.29306 w_k1_IBM_s006
                                 + 1.10301 w_k1_INTC_s006
                                 + 1.23006 w_k1_KO_s006 + 1.44669 w_k1_MAT_s006
                                 + 0.99487 w_k1_MSFT_s006
                                 + 1.09324 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.805807 w_k1_ADM_s006
                                 + 0.871152 w_k1_AXP_s006
                                 + 0.790429 w_k1_DIS_s006
                                 + 0.79385 w_k1_FDX_s006
                                 + 0.775901 w_k1_IBM_s006
                                 + 1.05439 w_k1_INTC_s006
                                 + 0.820805 w_k1_KO_s006
                                 + 0.774681 w_k1_MAT_s006
                                 + 0.795791 w_k1_MSFT_s006
                                 + 0.782399 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.05902 w_k1_ADM_s007
                                 + 1.01786 w_k1_AXP_s007
                                 + 0.721118 w_k1_DIS_s007
                                 + 0.916604 w_k1_FDX_s007
                                 + 1.15621 w_k1_IBM_s007
                                 + 0.719919 w_k1_INTC_s007
                                 + 0.906228 w_k1_KO_s007
                                 + 1.44947 w_k1_MAT_s007
                                 + 0.958462 w_k1_MSFT_s007
                                 + 1.07361 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.11436 w_k1_ADM_s007
                                 + 1.41788 w_k1_AXP_s007
                                 + 1.33195 w_k1_DIS_s007
                                 + 1.17118 w_k1_FDX_s007
                                 + 1.43395 w_k1_IBM_s007
                                 + 1.00695 w_k1_INTC_s007 + 1.3214 w_k1_KO_s007
                                 + 1.319 w_k1_MAT_s007 + 1.13552 w_k1_MSFT_s007
                                 + 1.26787 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.879438 w_k1_ADM_s007
                                 + 0.922625 w_k1_AXP_s007
                                 + 0.800293 w_k1_DIS_s007
                                 + 0.920223 w_k1_FDX_s007
                                 + 0.746786 w_k1_IBM_s007
                                 + 0.900249 w_k1_INTC_s007
                                 + 0.692844 w_k1_KO_s007
                                 + 0.71867 w_k1_MAT_s007
                                 + 0.916228 w_k1_MSFT_s007
                                 + 0.885787 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.31662 w_k1_ADM_s007
                                 + 1.15979 w_k1_AXP_s007
                                 + 1.52351 w_k1_DIS_s007
                                 + 1.36958 w_k1_FDX_s007
                                 + 1.47618 w_k1_IBM_s007
                                 + 1.25014 w_k1_INTC_s007 + 1.4055 w_k1_KO_s007
                                 + 1.10302 w_k1_MAT_s007
                                 + 1.42743 w_k1_MSFT_s007
                                 + 1.33093 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.815451 w_k1_ADM_s007
                                 + 0.706577 w_k1_AXP_s007
                                 + 0.862405 w_k1_DIS_s007
                                 + 0.831994 w_k1_FDX_s007
                                 + 1.13384 w_k1_IBM_s007
                                 + 0.921886 w_k1_INTC_s007
                                 + 1.06675 w_k1_KO_s007 + 1.24265 w_k1_MAT_s007
                                 + 1.00746 w_k1_MSFT_s007
                                 + 0.854764 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.04524 w_k1_ADM_s007
                                 + 0.887224 w_k1_AXP_s007
                                 + 0.898997 w_k1_DIS_s007
                                 + 1.06036 w_k1_FDX_s007
                                 + 0.835131 w_k1_IBM_s007
                                 + 1.2982 w_k1_INTC_s007
                                 + 0.919116 w_k1_KO_s007
                                 + 0.658852 w_k1_MAT_s007
                                 + 1.10841 w_k1_MSFT_s007
                                 + 0.930144 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.0832 w_k1_ADM_s007
                                 + 1.02059 w_k1_AXP_s007 + 1.2011 w_k1_DIS_s007
                                 + 1.12284 w_k1_FDX_s007
                                 + 1.17221 w_k1_IBM_s007
                                 + 0.935782 w_k1_INTC_s007
                                 + 1.11686 w_k1_KO_s007 + 1.26248 w_k1_MAT_s007
                                 + 1.18066 w_k1_MSFT_s007
                                 + 0.99237 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.736075 w_k1_ADM_s007
                                 + 0.657686 w_k1_AXP_s007
                                 + 0.618415 w_k1_DIS_s007
                                 + 0.69931 w_k1_FDX_s007
                                 + 0.807887 w_k1_IBM_s007
                                 + 0.805426 w_k1_INTC_s007
                                 + 0.727153 w_k1_KO_s007
                                 + 0.626381 w_k1_MAT_s007
                                 + 0.773949 w_k1_MSFT_s007
                                 + 0.784767 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.11445 w_k1_ADM_s010
                                 + 1.22322 w_k1_AXP_s010
                                 + 1.29738 w_k1_DIS_s010
                                 + 1.16271 w_k1_FDX_s010
                                 + 1.37392 w_k1_IBM_s010
                                 + 1.0393 w_k1_INTC_s010 + 1.32275 w_k1_KO_s010
                                 + 1.18747 w_k1_MAT_s010
                                 + 1.06923 w_k1_MSFT_s010
                                 + 1.07927 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.957653 w_k1_ADM_s010
                                 + 1.07218 w_k1_AXP_s010
                                 + 1.88933 w_k1_DIS_s010
                                 + 1.21458 w_k1_FDX_s010
                                 + 1.31725 w_k1_IBM_s010
                                 + 1.53222 w_k1_INTC_s010
                                 + 1.50787 w_k1_KO_s010 + 1.24351 w_k1_MAT_s010
                                 + 1.36896 w_k1_MSFT_s010
                                 + 1.15202 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.00844 w_k1_ADM_s010
                                 + 1.22067 w_k1_AXP_s010
                                 + 1.17509 w_k1_DIS_s010
                                 + 1.03769 w_k1_FDX_s010
                                 + 1.63671 w_k1_IBM_s010
                                 + 1.10175 w_k1_INTC_s010
                                 + 1.43199 w_k1_KO_s010 + 1.74651 w_k1_MAT_s010
                                 + 1.10446 w_k1_MSFT_s010
                                 + 1.08804 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.42986 w_k1_ADM_s010
                                 + 1.49798 w_k1_AXP_s010
                                 + 1.47733 w_k1_DIS_s010
                                 + 1.35476 w_k1_FDX_s010
                                 + 1.04393 w_k1_IBM_s010
                                 + 1.06243 w_k1_INTC_s010
                                 + 1.30715 w_k1_KO_s010 + 1.19048 w_k1_MAT_s010
                                 + 1.13809 w_k1_MSFT_s010
                                 + 1.28788 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.749403 w_k1_ADM_s010
                                 + 0.861113 w_k1_AXP_s010
                                 + 0.66195 w_k1_DIS_s010
                                 + 0.752016 w_k1_FDX_s010
                                 + 0.880028 w_k1_IBM_s010
                                 + 0.842379 w_k1_INTC_s010
                                 + 0.738733 w_k1_KO_s010
                                 + 0.819101 w_k1_MAT_s010
                                 + 0.778238 w_k1_MSFT_s010
                                 + 0.76952 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.870211 w_k1_ADM_s010
                                 + 0.754449 w_k1_AXP_s010
                                 + 0.793879 w_k1_DIS_s010
                                 + 0.896785 w_k1_FDX_s010
                                 + 0.704724 w_k1_IBM_s010
                                 + 0.902267 w_k1_INTC_s010
                                 + 0.815052 w_k1_KO_s010
                                 + 0.652742 w_k1_MAT_s010
                                 + 0.923928 w_k1_MSFT_s010
                                 + 0.88034 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.00939 w_k1_ADM_s010
                                 + 1.01889 w_k1_AXP_s010
                                 + 1.11113 w_k1_DIS_s010
                                 + 1.07104 w_k1_FDX_s010
                                 + 0.921615 w_k1_IBM_s010
                                 + 0.920578 w_k1_INTC_s010
                                 + 0.975594 w_k1_KO_s010 + 1.2943 w_k1_MAT_s010
                                 + 1.13041 w_k1_MSFT_s010
                                 + 0.999121 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.784941 w_k1_ADM_s010
                                 + 1.30175 w_k1_AXP_s010
                                 + 1.00099 w_k1_DIS_s010
                                 + 1.01415 w_k1_FDX_s010
                                 + 1.06098 w_k1_IBM_s010
                                 + 1.01243 w_k1_INTC_s010 + 1.0282 w_k1_KO_s010
                                 + 1.35822 w_k1_MAT_s010
                                 + 0.956796 w_k1_MSFT_s010
                                 + 1.03713 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.6198 w_k1_ADM_s011
                                 + 1.50276 w_k1_AXP_s011
                                 + 1.73059 w_k1_DIS_s011
                                 + 1.39309 w_k1_FDX_s011
                                 + 1.30071 w_k1_IBM_s011
                                 + 1.14853 w_k1_INTC_s011 + 1.4634 w_k1_KO_s011
                                 + 1.18878 w_k1_MAT_s011
                                 + 1.40251 w_k1_MSFT_s011
                                 + 1.22302 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.44315 w_k1_ADM_s011
                                 + 1.28189 w_k1_AXP_s011
                                 + 1.32081 w_k1_DIS_s011
                                 + 1.12042 w_k1_FDX_s011
                                 + 1.66376 w_k1_IBM_s011
                                 + 1.33981 w_k1_INTC_s011
                                 + 1.45983 w_k1_KO_s011 + 1.29418 w_k1_MAT_s011
                                 + 1.2685 w_k1_MSFT_s011 + 1.03574 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.859689 w_k1_ADM_s011
                                 + 0.623331 w_k1_AXP_s011
                                 + 0.69093 w_k1_DIS_s011
                                 + 0.834803 w_k1_FDX_s011
                                 + 0.673701 w_k1_IBM_s011
                                 + 1.03792 w_k1_INTC_s011
                                 + 0.734723 w_k1_KO_s011
                                 + 0.462508 w_k1_MAT_s011
                                 + 0.991638 w_k1_MSFT_s011
                                 + 0.812279 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.952363 w_k1_ADM_s011
                                 + 0.750944 w_k1_AXP_s011
                                 + 0.801567 w_k1_DIS_s011
                                 + 0.879995 w_k1_FDX_s011
                                 + 0.887162 w_k1_IBM_s011
                                 + 1.06164 w_k1_INTC_s011
                                 + 0.888232 w_k1_KO_s011
                                 + 0.690884 w_k1_MAT_s011
                                 + 0.949736 w_k1_MSFT_s011
                                 + 0.86382 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.24793 w_k1_ADM_s011
                                 + 1.13437 w_k1_AXP_s011
                                 + 1.01841 w_k1_DIS_s011
                                 + 1.09054 w_k1_FDX_s011
                                 + 0.892337 w_k1_IBM_s011
                                 + 1.26213 w_k1_INTC_s011
                                 + 1.09263 w_k1_KO_s011
                                 + 0.990759 w_k1_MAT_s011
                                 + 1.23199 w_k1_MSFT_s011
                                 + 1.16992 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.31143 w_k1_ADM_s011
                                 + 1.11825 w_k1_AXP_s011
                                 + 1.49645 w_k1_DIS_s011
                                 + 1.24328 w_k1_FDX_s011
                                 + 0.962619 w_k1_IBM_s011
                                 + 1.49905 w_k1_INTC_s011
                                 + 1.25997 w_k1_KO_s011
                                 + 0.912466 w_k1_MAT_s011
                                 + 1.2915 w_k1_MSFT_s011 + 1.07898 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.01091 w_k1_ADM_s011
                                 + 1.02768 w_k1_AXP_s011
                                 + 0.852349 w_k1_DIS_s011
                                 + 0.885251 w_k1_FDX_s011
                                 + 1.44997 w_k1_IBM_s011
                                 + 1.23394 w_k1_INTC_s011
                                 + 1.14064 w_k1_KO_s011 + 1.19582 w_k1_MAT_s011
                                 + 1.14977 w_k1_MSFT_s011
                                 + 1.09606 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.946706 w_k1_ADM_s011
                                 + 0.791019 w_k1_AXP_s011
                                 + 0.6694 w_k1_DIS_s011
                                 + 0.755566 w_k1_FDX_s011
                                 + 1.1141 w_k1_IBM_s011
                                 + 0.786315 w_k1_INTC_s011
                                 + 0.946238 w_k1_KO_s011
                                 + 1.00575 w_k1_MAT_s011
                                 + 0.880296 w_k1_MSFT_s011
                                 + 0.932774 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.30919 w_k1_ADM_s012
                                 + 1.17404 w_k1_AXP_s012
                                 + 0.813096 w_k1_DIS_s012
                                 + 0.968253 w_k1_FDX_s012
                                 + 0.938884 w_k1_IBM_s012
                                 + 0.859613 w_k1_INTC_s012
                                 + 0.915476 w_k1_KO_s012
                                 + 1.11713 w_k1_MAT_s012
                                 + 0.846446 w_k1_MSFT_s012
                                 + 0.95101 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.45436 w_k1_ADM_s012
                                 + 1.14533 w_k1_AXP_s012
                                 + 1.23335 w_k1_DIS_s012
                                 + 1.24883 w_k1_FDX_s012
                                 + 0.938708 w_k1_IBM_s012
                                 + 0.961804 w_k1_INTC_s012
                                 + 1.13251 w_k1_KO_s012
                                 + 0.848231 w_k1_MAT_s012
                                 + 1.04975 w_k1_MSFT_s012
                                 + 1.05171 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.29842 w_k1_ADM_s012
                                 + 1.31991 w_k1_AXP_s012
                                 + 1.11834 w_k1_DIS_s012
                                 + 0.987973 w_k1_FDX_s012
                                 + 1.56742 w_k1_IBM_s012
                                 + 1.25639 w_k1_INTC_s012
                                 + 1.30589 w_k1_KO_s012 + 1.22257 w_k1_MAT_s012
                                 + 0.944315 w_k1_MSFT_s012
                                 + 1.14264 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.6936 w_k1_ADM_s012
                                 + 1.22358 w_k1_AXP_s012
                                 + 1.57295 w_k1_DIS_s012
                                 + 1.33211 w_k1_FDX_s012
                                 + 1.00706 w_k1_IBM_s012
                                 + 1.10407 w_k1_INTC_s012
                                 + 1.21164 w_k1_KO_s012 + 1.117 w_k1_MAT_s012
                                 + 1.24248 w_k1_MSFT_s012
                                 + 1.11085 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.976095 w_k1_ADM_s012
                                 + 0.752673 w_k1_AXP_s012
                                 + 0.882224 w_k1_DIS_s012
                                 + 0.807237 w_k1_FDX_s012
                                 + 0.942097 w_k1_IBM_s012
                                 + 0.808739 w_k1_INTC_s012
                                 + 0.895091 w_k1_KO_s012
                                 + 0.731473 w_k1_MAT_s012
                                 + 0.83476 w_k1_MSFT_s012
                                 + 0.868327 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.0053 w_k1_ADM_s012
                                 + 0.825164 w_k1_AXP_s012
                                 + 0.725542 w_k1_DIS_s012
                                 + 0.885542 w_k1_FDX_s012
                                 + 0.6531 w_k1_IBM_s012
                                 + 0.900884 w_k1_INTC_s012
                                 + 0.613849 w_k1_KO_s012
                                 + 0.601221 w_k1_MAT_s012
                                 + 0.822238 w_k1_MSFT_s012
                                 + 0.851089 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.49285 w_k1_ADM_s012
                                 + 1.07862 w_k1_AXP_s012
                                 + 1.03282 w_k1_DIS_s012
                                 + 1.13633 w_k1_FDX_s012
                                 + 0.693476 w_k1_IBM_s012
                                 + 0.76756 w_k1_INTC_s012 + 0.9543 w_k1_KO_s012
                                 + 0.779171 w_k1_MAT_s012
                                 + 0.931077 w_k1_MSFT_s012
                                 + 0.997944 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.51159 w_k1_ADM_s012
                                 + 1.14866 w_k1_AXP_s012
                                 + 1.22956 w_k1_DIS_s012
                                 + 1.14218 w_k1_FDX_s012
                                 + 1.21596 w_k1_IBM_s012
                                 + 1.07606 w_k1_INTC_s012
                                 + 1.40211 w_k1_KO_s012 + 1.12305 w_k1_MAT_s012
                                 + 1.14377 w_k1_MSFT_s012
                                 + 1.19932 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.08544 w_k1_ADM_s013
                                 + 1.1725 w_k1_AXP_s013 + 1.1295 w_k1_DIS_s013
                                 + 1.0083 w_k1_FDX_s013 + 1.35948 w_k1_IBM_s013
                                 + 0.966579 w_k1_INTC_s013
                                 + 1.1416 w_k1_KO_s013 + 1.04434 w_k1_MAT_s013
                                 + 1.08355 w_k1_MSFT_s013
                                 + 1.01974 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.533122 w_k1_ADM_s013
                                 + 0.886031 w_k1_AXP_s013
                                 + 0.818067 w_k1_DIS_s013
                                 + 0.848349 w_k1_FDX_s013
                                 + 1.10096 w_k1_IBM_s013
                                 + 0.842829 w_k1_INTC_s013
                                 + 0.866052 w_k1_KO_s013
                                 + 1.09946 w_k1_MAT_s013
                                 + 0.908773 w_k1_MSFT_s013
                                 + 0.858074 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.901875 w_k1_ADM_s013
                                 + 1.01901 w_k1_AXP_s013
                                 + 0.868935 w_k1_DIS_s013
                                 + 0.846271 w_k1_FDX_s013
                                 + 1.11045 w_k1_IBM_s013
                                 + 1.04008 w_k1_INTC_s013
                                 + 0.883268 w_k1_KO_s013
                                 + 1.06611 w_k1_MAT_s013
                                 + 1.03279 w_k1_MSFT_s013
                                 + 0.863682 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.676803 w_k1_ADM_s013
                                 + 0.756028 w_k1_AXP_s013
                                 + 0.573018 w_k1_DIS_s013
                                 + 0.667038 w_k1_FDX_s013
                                 + 0.917832 w_k1_IBM_s013
                                 + 0.871547 w_k1_INTC_s013
                                 + 0.652291 w_k1_KO_s013
                                 + 0.853817 w_k1_MAT_s013
                                 + 0.763483 w_k1_MSFT_s013
                                 + 0.708673 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.919467 w_k1_ADM_s013
                                 + 1.01962 w_k1_AXP_s013 + 1.0441 w_k1_DIS_s013
                                 + 1.08103 w_k1_FDX_s013
                                 + 0.810556 w_k1_IBM_s013
                                 + 0.978912 w_k1_INTC_s013
                                 + 0.943127 w_k1_KO_s013
                                 + 0.851818 w_k1_MAT_s013
                                 + 1.04648 w_k1_MSFT_s013
                                 + 0.947216 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.23555 w_k1_ADM_s013
                                 + 1.49707 w_k1_AXP_s013
                                 + 1.58207 w_k1_DIS_s013
                                 + 1.29447 w_k1_FDX_s013
                                 + 1.28623 w_k1_IBM_s013
                                 + 1.37908 w_k1_INTC_s013
                                 + 1.31112 w_k1_KO_s013 + 1.09074 w_k1_MAT_s013
                                 + 1.43951 w_k1_MSFT_s013
                                 + 1.16286 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.896158 w_k1_ADM_s013
                                 + 0.78396 w_k1_AXP_s013
                                 + 0.846445 w_k1_DIS_s013
                                 + 0.84684 w_k1_FDX_s013
                                 + 0.692347 w_k1_IBM_s013
                                 + 0.940085 w_k1_INTC_s013
                                 + 0.784076 w_k1_KO_s013
                                 + 0.696408 w_k1_MAT_s013
                                 + 0.933719 w_k1_MSFT_s013
                                 + 0.837409 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.05368 w_k1_ADM_s013
                                 + 1.02038 w_k1_AXP_s013
                                 + 1.40726 w_k1_DIS_s013
                                 + 1.03104 w_k1_FDX_s013
                                 + 1.01247 w_k1_IBM_s013
                                 + 1.06103 w_k1_INTC_s013
                                 + 1.21255 w_k1_KO_s013
                                 + 0.813548 w_k1_MAT_s013
                                 + 1.35161 w_k1_MSFT_s013
                                 + 0.993412 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.897953 w_k1_ADM_s014
                                 + 0.836984 w_k1_AXP_s014
                                 + 0.906346 w_k1_DIS_s014
                                 + 0.943038 w_k1_FDX_s014
                                 + 0.758263 w_k1_IBM_s014
                                 + 0.951134 w_k1_INTC_s014
                                 + 0.880072 w_k1_KO_s014
                                 + 0.739651 w_k1_MAT_s014
                                 + 0.93747 w_k1_MSFT_s014
                                 + 0.962723 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.92477 w_k1_ADM_s014
                                 + 1.02672 w_k1_AXP_s014
                                 + 1.05244 w_k1_DIS_s014
                                 + 0.900703 w_k1_FDX_s014
                                 + 1.84103 w_k1_IBM_s014
                                 + 0.83498 w_k1_INTC_s014
                                 + 1.26362 w_k1_KO_s014 + 2.08273 w_k1_MAT_s014
                                 + 0.868155 w_k1_MSFT_s014
                                 + 0.931218 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.657933 w_k1_ADM_s014
                                 + 0.565419 w_k1_AXP_s014
                                 + 0.611062 w_k1_DIS_s014
                                 + 0.723578 w_k1_FDX_s014
                                 + 0.75436 w_k1_IBM_s014
                                 + 0.982101 w_k1_INTC_s014
                                 + 0.799112 w_k1_KO_s014
                                 + 1.03506 w_k1_MAT_s014
                                 + 0.890575 w_k1_MSFT_s014
                                 + 0.878065 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.800319 w_k1_ADM_s014
                                 + 0.909344 w_k1_AXP_s014
                                 + 0.853681 w_k1_DIS_s014
                                 + 0.861236 w_k1_FDX_s014
                                 + 1.16111 w_k1_IBM_s014
                                 + 0.846402 w_k1_INTC_s014
                                 + 1.00217 w_k1_KO_s014 + 1.5264 w_k1_MAT_s014
                                 + 0.941784 w_k1_MSFT_s014
                                 + 0.940614 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.12426 w_k1_ADM_s014
                                 + 1.31364 w_k1_AXP_s014
                                 + 1.61006 w_k1_DIS_s014
                                 + 1.42727 w_k1_FDX_s014
                                 + 0.892321 w_k1_IBM_s014
                                 + 1.14348 w_k1_INTC_s014
                                 + 1.14086 w_k1_KO_s014 + 1.12546 w_k1_MAT_s014
                                 + 1.379 w_k1_MSFT_s014 + 1.10051 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.832592 w_k1_ADM_s014
                                 + 0.917907 w_k1_AXP_s014
                                 + 1.11625 w_k1_DIS_s014
                                 + 0.972387 w_k1_FDX_s014
                                 + 1.34779 w_k1_IBM_s014
                                 + 0.949565 w_k1_INTC_s014
                                 + 1.1829 w_k1_KO_s014 + 1.72134 w_k1_MAT_s014
                                 + 1.11036 w_k1_MSFT_s014
                                 + 1.05498 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.16783 w_k1_ADM_s014
                                 + 1.03631 w_k1_AXP_s014
                                 + 1.06144 w_k1_DIS_s014
                                 + 1.12302 w_k1_FDX_s014
                                 + 0.822616 w_k1_IBM_s014
                                 + 1.05485 w_k1_INTC_s014
                                 + 0.883929 w_k1_KO_s014
                                 + 0.854461 w_k1_MAT_s014
                                 + 1.10489 w_k1_MSFT_s014
                                 + 1.00024 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.983554 w_k1_ADM_s014
                                 + 1.26538 w_k1_AXP_s014
                                 + 1.60672 w_k1_DIS_s014
                                 + 1.19989 w_k1_FDX_s014
                                 + 1.73947 w_k1_IBM_s014
                                 + 1.26289 w_k1_INTC_s014
                                 + 1.73371 w_k1_KO_s014 + 2.07386 w_k1_MAT_s014
                                 + 1.50215 w_k1_MSFT_s014
                                 + 1.09997 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.941702 w_k1_ADM_s015
                                 + 0.870011 w_k1_AXP_s015
                                 + 0.836728 w_k1_DIS_s015
                                 + 0.865799 w_k1_FDX_s015
                                 + 0.91436 w_k1_IBM_s015
                                 + 0.952749 w_k1_INTC_s015
                                 + 0.837954 w_k1_KO_s015
                                 + 0.979701 w_k1_MAT_s015
                                 + 0.957189 w_k1_MSFT_s015
                                 + 0.921262 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.05495 w_k1_ADM_s015
                                 + 0.930146 w_k1_AXP_s015
                                 + 1.08321 w_k1_DIS_s015
                                 + 1.15484 w_k1_FDX_s015
                                 + 0.647271 w_k1_IBM_s015
                                 + 1.12688 w_k1_INTC_s015
                                 + 0.931269 w_k1_KO_s015
                                 + 0.727572 w_k1_MAT_s015
                                 + 1.06508 w_k1_MSFT_s015
                                 + 0.928327 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.1653 w_k1_ADM_s015
                                 + 0.878188 w_k1_AXP_s015
                                 + 0.954831 w_k1_DIS_s015
                                 + 0.968808 w_k1_FDX_s015
                                 + 0.911562 w_k1_IBM_s015
                                 + 0.822349 w_k1_INTC_s015
                                 + 0.953224 w_k1_KO_s015 + 1.0491 w_k1_MAT_s015
                                 + 1.00112 w_k1_MSFT_s015
                                 + 0.945585 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.725411 w_k1_ADM_s015
                                 + 0.723765 w_k1_AXP_s015
                                 + 0.577427 w_k1_DIS_s015
                                 + 0.773797 w_k1_FDX_s015
                                 + 0.945598 w_k1_IBM_s015
                                 + 0.792702 w_k1_INTC_s015
                                 + 0.776736 w_k1_KO_s015
                                 + 1.21563 w_k1_MAT_s015
                                 + 0.747662 w_k1_MSFT_s015
                                 + 0.818069 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.34082 w_k1_ADM_s015
                                 + 1.30117 w_k1_AXP_s015
                                 + 1.85029 w_k1_DIS_s015
                                 + 1.38109 w_k1_FDX_s015
                                 + 1.28465 w_k1_IBM_s015
                                 + 1.2867 w_k1_INTC_s015 + 1.32511 w_k1_KO_s015
                                 + 1.20826 w_k1_MAT_s015
                                 + 1.49361 w_k1_MSFT_s015
                                 + 1.20295 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.25366 w_k1_ADM_s015
                                 + 1.51005 w_k1_AXP_s015
                                 + 1.01147 w_k1_DIS_s015
                                 + 1.09034 w_k1_FDX_s015
                                 + 1.32589 w_k1_IBM_s015
                                 + 1.0281 w_k1_INTC_s015 + 1.1588 w_k1_KO_s015
                                 + 2.24768 w_k1_MAT_s015
                                 + 1.13437 w_k1_MSFT_s015
                                 + 1.16379 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.952034 w_k1_ADM_s015
                                 + 1.13638 w_k1_AXP_s015
                                 + 1.00871 w_k1_DIS_s015
                                 + 1.00931 w_k1_FDX_s015
                                 + 1.19956 w_k1_IBM_s015
                                 + 1.06847 w_k1_INTC_s015
                                 + 0.91596 w_k1_KO_s015 + 1.27902 w_k1_MAT_s015
                                 + 1.06118 w_k1_MSFT_s015
                                 + 0.984042 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.1938 w_k1_ADM_s015
                                 + 1.31948 w_k1_AXP_s015
                                 + 1.30947 w_k1_DIS_s015
                                 + 1.21255 w_k1_FDX_s015
                                 + 1.30465 w_k1_IBM_s015
                                 + 1.10653 w_k1_INTC_s015
                                 + 1.06792 w_k1_KO_s015 + 1.40782 w_k1_MAT_s015
                                 + 1.12809 w_k1_MSFT_s015
                                 + 1.02677 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.766833 w_k1_ADM_s016
                                 + 0.768015 w_k1_AXP_s016
                                 + 1.02268 w_k1_DIS_s016
                                 + 0.887931 w_k1_FDX_s016
                                 + 0.944859 w_k1_IBM_s016
                                 + 1.04829 w_k1_INTC_s016
                                 + 1.12011 w_k1_KO_s016
                                 + 0.589153 w_k1_MAT_s016
                                 + 0.906106 w_k1_MSFT_s016
                                 + 0.788457 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.02814 w_k1_ADM_s016
                                 + 0.897738 w_k1_AXP_s016
                                 + 0.85159 w_k1_DIS_s016
                                 + 1.06824 w_k1_FDX_s016
                                 + 0.729765 w_k1_IBM_s016
                                 + 0.905193 w_k1_INTC_s016
                                 + 0.896756 w_k1_KO_s016
                                 + 0.665327 w_k1_MAT_s016
                                 + 0.865093 w_k1_MSFT_s016
                                 + 0.97314 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.14578 w_k1_ADM_s016
                                 + 1.31887 w_k1_AXP_s016
                                 + 0.970718 w_k1_DIS_s016
                                 + 1.08241 w_k1_FDX_s016
                                 + 0.756487 w_k1_IBM_s016
                                 + 0.895803 w_k1_INTC_s016
                                 + 0.896476 w_k1_KO_s016
                                 + 0.82845 w_k1_MAT_s016
                                 + 0.85661 w_k1_MSFT_s016
                                 + 0.990311 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.990253 w_k1_ADM_s016
                                 + 1.13129 w_k1_AXP_s016
                                 + 0.779781 w_k1_DIS_s016
                                 + 0.880719 w_k1_FDX_s016
                                 + 1.03925 w_k1_IBM_s016
                                 + 0.794678 w_k1_INTC_s016
                                 + 1.0522 w_k1_KO_s016 + 1.10749 w_k1_MAT_s016
                                 + 0.764348 w_k1_MSFT_s016
                                 + 0.847737 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.03107 w_k1_ADM_s016
                                 + 1.21324 w_k1_AXP_s016
                                 + 1.10684 w_k1_DIS_s016
                                 + 1.04772 w_k1_FDX_s016
                                 + 1.35017 w_k1_IBM_s016
                                 + 0.905005 w_k1_INTC_s016
                                 + 1.3478 w_k1_KO_s016 + 1.23507 w_k1_MAT_s016
                                 + 0.888143 w_k1_MSFT_s016
                                 + 0.93233 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.36685 w_k1_ADM_s016
                                 + 1.43221 w_k1_AXP_s016
                                 + 1.70116 w_k1_DIS_s016
                                 + 1.39457 w_k1_FDX_s016
                                 + 1.18106 w_k1_IBM_s016
                                 + 1.38343 w_k1_INTC_s016 + 1.3604 w_k1_KO_s016
                                 + 1.44441 w_k1_MAT_s016
                                 + 1.28393 w_k1_MSFT_s016
                                 + 1.07606 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.21736 w_k1_ADM_s016
                                 + 0.968807 w_k1_AXP_s016
                                 + 1.23454 w_k1_DIS_s016
                                 + 1.11236 w_k1_FDX_s016
                                 + 1.03707 w_k1_IBM_s016
                                 + 0.991742 w_k1_INTC_s016
                                 + 1.19504 w_k1_KO_s016
                                 + 0.899302 w_k1_MAT_s016
                                 + 1.01037 w_k1_MSFT_s016
                                 + 0.996077 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.696416 w_k1_ADM_s016
                                 + 0.701656 w_k1_AXP_s016
                                 + 0.617191 w_k1_DIS_s016
                                 + 0.723894 w_k1_FDX_s016
                                 + 0.824939 w_k1_IBM_s016
                                 + 0.826854 w_k1_INTC_s016
                                 + 0.87649 w_k1_KO_s016
                                 + 0.744488 w_k1_MAT_s016
                                 + 0.770239 w_k1_MSFT_s016
                                 + 0.827143 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.901698 w_k1_ADM_s017
                                 + 0.746669 w_k1_AXP_s017
                                 + 0.913312 w_k1_DIS_s017
                                 + 0.895028 w_k1_FDX_s017
                                 + 0.999987 w_k1_IBM_s017
                                 + 0.970877 w_k1_INTC_s017
                                 + 1.10239 w_k1_KO_s017 + 1.04021 w_k1_MAT_s017
                                 + 1.04775 w_k1_MSFT_s017
                                 + 0.907461 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.931656 w_k1_ADM_s017
                                 + 1.07333 w_k1_AXP_s017
                                 + 0.750411 w_k1_DIS_s017
                                 + 0.811083 w_k1_FDX_s017
                                 + 1.34416 w_k1_IBM_s017
                                 + 0.863626 w_k1_INTC_s017
                                 + 0.987162 w_k1_KO_s017
                                 + 1.39215 w_k1_MAT_s017
                                 + 0.892067 w_k1_MSFT_s017
                                 + 0.894872 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.03553 w_k1_ADM_s017
                                 + 1.11507 w_k1_AXP_s017
                                 + 1.01187 w_k1_DIS_s017
                                 + 1.12969 w_k1_FDX_s017
                                 + 0.834019 w_k1_IBM_s017
                                 + 1.04388 w_k1_INTC_s017
                                 + 0.892428 w_k1_KO_s017
                                 + 0.862182 w_k1_MAT_s017
                                 + 1.05472 w_k1_MSFT_s017
                                 + 0.931031 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.887483 w_k1_ADM_s017
                                 + 0.831333 w_k1_AXP_s017
                                 + 0.813725 w_k1_DIS_s017
                                 + 0.944703 w_k1_FDX_s017
                                 + 0.697315 w_k1_IBM_s017
                                 + 0.826804 w_k1_INTC_s017
                                 + 0.739815 w_k1_KO_s017
                                 + 0.710234 w_k1_MAT_s017
                                 + 0.921346 w_k1_MSFT_s017
                                 + 0.903608 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.1956 w_k1_ADM_s017
                                 + 1.10548 w_k1_AXP_s017 + 1.4384 w_k1_DIS_s017
                                 + 1.21054 w_k1_FDX_s017
                                 + 1.15286 w_k1_IBM_s017
                                 + 1.44134 w_k1_INTC_s017
                                 + 1.14463 w_k1_KO_s017
                                 + 0.782687 w_k1_MAT_s017
                                 + 1.30162 w_k1_MSFT_s017
                                 + 1.02377 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.1869 w_k1_ADM_s017
                                 + 1.33685 w_k1_AXP_s017
                                 + 1.02074 w_k1_DIS_s017
                                 + 1.18386 w_k1_FDX_s017 + 1.142 w_k1_IBM_s017
                                 + 1.11049 w_k1_INTC_s017
                                 + 1.03544 w_k1_KO_s017 + 1.30009 w_k1_MAT_s017
                                 + 1.09805 w_k1_MSFT_s017
                                 + 1.07101 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.760653 w_k1_ADM_s017
                                 + 0.634743 w_k1_AXP_s017
                                 + 0.562211 w_k1_DIS_s017
                                 + 0.698405 w_k1_FDX_s017
                                 + 0.818556 w_k1_IBM_s017
                                 + 0.577937 w_k1_INTC_s017
                                 + 0.632927 w_k1_KO_s017
                                 + 0.878206 w_k1_MAT_s017
                                 + 0.675338 w_k1_MSFT_s017
                                 + 0.727623 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.18859 w_k1_ADM_s017
                                 + 1.56229 w_k1_AXP_s017 + 1.5043 w_k1_DIS_s017
                                 + 1.42503 w_k1_FDX_s017 + 1.4333 w_k1_IBM_s017
                                 + 1.06943 w_k1_INTC_s017
                                 + 1.42893 w_k1_KO_s017 + 1.86141 w_k1_MAT_s017
                                 + 1.26329 w_k1_MSFT_s017
                                 + 1.10877 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.60888 w_k1_ADM_s100
                                 + 1.40669 w_k1_AXP_s100
                                 + 1.76131 w_k1_DIS_s100
                                 + 1.24905 w_k1_FDX_s100
                                 + 1.94717 w_k1_IBM_s100
                                 + 1.12613 w_k1_INTC_s100
                                 + 1.81641 w_k1_KO_s100 + 1.65823 w_k1_MAT_s100
                                 + 1.53019 w_k1_MSFT_s100
                                 + 1.43625 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.01451 w_k1_ADM_s100
                                 + 0.996632 w_k1_AXP_s100
                                 + 1.28416 w_k1_DIS_s100
                                 + 1.08136 w_k1_FDX_s100
                                 + 1.21643 w_k1_IBM_s100
                                 + 1.01773 w_k1_INTC_s100
                                 + 1.16085 w_k1_KO_s100
                                 + 0.851826 w_k1_MAT_s100
                                 + 1.21076 w_k1_MSFT_s100
                                 + 1.04454 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.14668 w_k1_ADM_s100
                                 + 1.07393 w_k1_AXP_s100 + 1.0275 w_k1_DIS_s100
                                 + 1.13189 w_k1_FDX_s100
                                 + 1.05807 w_k1_IBM_s100
                                 + 0.767288 w_k1_INTC_s100
                                 + 1.03264 w_k1_KO_s100 + 1.04475 w_k1_MAT_s100
                                 + 1.06395 w_k1_MSFT_s100
                                 + 1.14643 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.21911 w_k1_ADM_s100
                                 + 1.08892 w_k1_AXP_s100
                                 + 1.13867 w_k1_DIS_s100
                                 + 0.925344 w_k1_FDX_s100
                                 + 1.41558 w_k1_IBM_s100
                                 + 0.73781 w_k1_INTC_s100
                                 + 1.24524 w_k1_KO_s100 + 1.46889 w_k1_MAT_s100
                                 + 0.966543 w_k1_MSFT_s100
                                 + 1.22581 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.25174 w_k1_ADM_s100
                                 + 1.13261 w_k1_AXP_s100 + 1.2113 w_k1_DIS_s100
                                 + 1.23117 w_k1_FDX_s100
                                 + 0.704165 w_k1_IBM_s100
                                 + 1.08621 w_k1_INTC_s100
                                 + 0.920625 w_k1_KO_s100
                                 + 0.634956 w_k1_MAT_s100
                                 + 1.27369 w_k1_MSFT_s100
                                 + 0.990001 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.849306 w_k1_ADM_s100
                                 + 0.623127 w_k1_AXP_s100
                                 + 0.599436 w_k1_DIS_s100
                                 + 0.699686 w_k1_FDX_s100
                                 + 0.790104 w_k1_IBM_s100
                                 + 0.585476 w_k1_INTC_s100
                                 + 0.662098 w_k1_KO_s100
                                 + 0.733441 w_k1_MAT_s100
                                 + 0.645298 w_k1_MSFT_s100
                                 + 0.795787 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.01809 w_k1_ADM_s100
                                 + 0.966815 w_k1_AXP_s100
                                 + 0.928447 w_k1_DIS_s100
                                 + 0.961692 w_k1_FDX_s100
                                 + 0.814841 w_k1_IBM_s100
                                 + 0.992897 w_k1_INTC_s100
                                 + 0.899738 w_k1_KO_s100
                                 + 0.844007 w_k1_MAT_s100
                                 + 1.07158 w_k1_MSFT_s100
                                 + 0.974247 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.30831 w_k1_ADM_s100
                                 + 1.79616 w_k1_AXP_s100
                                 + 1.61216 w_k1_DIS_s100 + 1.4827 w_k1_FDX_s100
                                 + 1.38368 w_k1_IBM_s100
                                 + 1.15626 w_k1_INTC_s100
                                 + 1.30252 w_k1_KO_s100 + 1.3783 w_k1_MAT_s100
                                 + 1.40993 w_k1_MSFT_s100
                                 + 1.26111 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.826612 w_k1_ADM_s101
                                 + 0.820436 w_k1_AXP_s101
                                 + 0.66218 w_k1_DIS_s101
                                 + 0.798994 w_k1_FDX_s101
                                 + 1.21854 w_k1_IBM_s101
                                 + 1.19022 w_k1_INTC_s101
                                 + 0.899242 w_k1_KO_s101 + 1.2382 w_k1_MAT_s101
                                 + 0.921626 w_k1_MSFT_s101
                                 + 0.876312 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.14804 w_k1_ADM_s101
                                 + 0.952375 w_k1_AXP_s101
                                 + 0.975291 w_k1_DIS_s101
                                 + 1.09591 w_k1_FDX_s101
                                 + 0.945444 w_k1_IBM_s101
                                 + 1.09474 w_k1_INTC_s101
                                 + 0.922154 w_k1_KO_s101
                                 + 0.797053 w_k1_MAT_s101
                                 + 0.978083 w_k1_MSFT_s101
                                 + 1.07662 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.836682 w_k1_ADM_s101
                                 + 0.791135 w_k1_AXP_s101
                                 + 0.841338 w_k1_DIS_s101
                                 + 0.963309 w_k1_FDX_s101
                                 + 0.74702 w_k1_IBM_s101
                                 + 0.869813 w_k1_INTC_s101
                                 + 0.818079 w_k1_KO_s101
                                 + 0.757625 w_k1_MAT_s101
                                 + 0.958297 w_k1_MSFT_s101
                                 + 0.956374 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.924648 w_k1_ADM_s101
                                 + 1.38608 w_k1_AXP_s101
                                 + 1.19696 w_k1_DIS_s101
                                 + 1.05135 w_k1_FDX_s101
                                 + 1.61576 w_k1_IBM_s101
                                 + 1.17689 w_k1_INTC_s101
                                 + 1.15986 w_k1_KO_s101 + 1.16703 w_k1_MAT_s101
                                 + 0.913519 w_k1_MSFT_s101
                                 + 1.16178 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.18498 w_k1_ADM_s101
                                 + 1.17369 w_k1_AXP_s101
                                 + 1.17001 w_k1_DIS_s101
                                 + 1.16814 w_k1_FDX_s101
                                 + 1.21043 w_k1_IBM_s101
                                 + 0.987105 w_k1_INTC_s101
                                 + 0.927693 w_k1_KO_s101
                                 + 1.04265 w_k1_MAT_s101
                                 + 1.1014 w_k1_MSFT_s101 + 1.10027 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.725311 w_k1_ADM_s101
                                 + 0.744111 w_k1_AXP_s101
                                 + 0.567599 w_k1_DIS_s101
                                 + 0.689111 w_k1_FDX_s101
                                 + 0.66079 w_k1_IBM_s101
                                 + 0.786846 w_k1_INTC_s101
                                 + 0.703916 w_k1_KO_s101
                                 + 0.755833 w_k1_MAT_s101
                                 + 0.752639 w_k1_MSFT_s101
                                 + 0.758908 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.978922 w_k1_ADM_s101
                                 + 0.720467 w_k1_AXP_s101
                                 + 0.636843 w_k1_DIS_s101
                                 + 0.79314 w_k1_FDX_s101
                                 + 0.956935 w_k1_IBM_s101
                                 + 0.79556 w_k1_INTC_s101
                                 + 0.877264 w_k1_KO_s101
                                 + 0.791093 w_k1_MAT_s101
                                 + 0.826554 w_k1_MSFT_s101
                                 + 0.96462 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.51404 w_k1_ADM_s101
                                 + 1.28217 w_k1_AXP_s101
                                 + 1.22338 w_k1_DIS_s101
                                 + 1.30599 w_k1_FDX_s101
                                 + 0.880092 w_k1_IBM_s101
                                 + 1.14421 w_k1_INTC_s101
                                 + 0.978192 w_k1_KO_s101
                                 + 0.870261 w_k1_MAT_s101
                                 + 1.07116 w_k1_MSFT_s101
                                 + 1.01321 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.14936 w_k1_ADM_s102
                                 + 1.47169 w_k1_AXP_s102
                                 + 1.39875 w_k1_DIS_s102 + 1.3043 w_k1_FDX_s102
                                 + 1.32997 w_k1_IBM_s102
                                 + 1.3011 w_k1_INTC_s102 + 1.26097 w_k1_KO_s102
                                 + 1.36684 w_k1_MAT_s102
                                 + 1.21552 w_k1_MSFT_s102
                                 + 1.21329 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.790799 w_k1_ADM_s102
                                 + 0.699062 w_k1_AXP_s102
                                 + 0.618414 w_k1_DIS_s102
                                 + 0.78036 w_k1_FDX_s102
                                 + 0.673171 w_k1_IBM_s102
                                 + 0.832167 w_k1_INTC_s102
                                 + 0.639605 w_k1_KO_s102
                                 + 0.356547 w_k1_MAT_s102
                                 + 0.748288 w_k1_MSFT_s102
                                 + 0.790419 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.860712 w_k1_ADM_s102
                                 + 1.07863 w_k1_AXP_s102
                                 + 0.945969 w_k1_DIS_s102
                                 + 1.01163 w_k1_FDX_s102
                                 + 0.979968 w_k1_IBM_s102
                                 + 0.868061 w_k1_INTC_s102
                                 + 0.933994 w_k1_KO_s102
                                 + 0.773599 w_k1_MAT_s102
                                 + 0.934371 w_k1_MSFT_s102
                                 + 1.04386 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.686034 w_k1_ADM_s102
                                 + 0.788247 w_k1_AXP_s102
                                 + 0.7874 w_k1_DIS_s102
                                 + 0.927591 w_k1_FDX_s102
                                 + 0.821006 w_k1_IBM_s102
                                 + 0.928494 w_k1_INTC_s102
                                 + 0.896832 w_k1_KO_s102
                                 + 0.759844 w_k1_MAT_s102
                                 + 0.98308 w_k1_MSFT_s102
                                 + 0.965421 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.08869 w_k1_ADM_s102
                                 + 1.03853 w_k1_AXP_s102 + 1.1485 w_k1_DIS_s102
                                 + 1.25627 w_k1_FDX_s102
                                 + 0.991677 w_k1_IBM_s102
                                 + 1.37627 w_k1_INTC_s102
                                 + 1.03563 w_k1_KO_s102
                                 + 0.638599 w_k1_MAT_s102
                                 + 1.24469 w_k1_MSFT_s102
                                 + 1.03453 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.739761 w_k1_ADM_s102
                                 + 0.667496 w_k1_AXP_s102
                                 + 0.935329 w_k1_DIS_s102
                                 + 0.925855 w_k1_FDX_s102
                                 + 0.845109 w_k1_IBM_s102
                                 + 0.977128 w_k1_INTC_s102
                                 + 1.07021 w_k1_KO_s102
                                 + 0.666426 w_k1_MAT_s102
                                 + 1.13117 w_k1_MSFT_s102
                                 + 0.966326 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.810506 w_k1_ADM_s102
                                 + 0.93987 w_k1_AXP_s102
                                 + 1.15645 w_k1_DIS_s102
                                 + 0.985045 w_k1_FDX_s102
                                 + 1.22997 w_k1_IBM_s102
                                 + 1.31819 w_k1_INTC_s102
                                 + 1.17776 w_k1_KO_s102
                                 + 0.734098 w_k1_MAT_s102
                                 + 1.10368 w_k1_MSFT_s102
                                 + 1.06173 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.17997 w_k1_ADM_s102
                                 + 1.01278 w_k1_AXP_s102
                                 + 1.23031 w_k1_DIS_s102
                                 + 1.25772 w_k1_FDX_s102
                                 + 0.724532 w_k1_IBM_s102
                                 + 1.08929 w_k1_INTC_s102
                                 + 1.10322 w_k1_KO_s102 + 0.53864 w_k1_MAT_s102
                                 + 1.2617 w_k1_MSFT_s102 + 1.16881 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.75686 w_k1_ADM_s103
                                 + 0.906699 w_k1_AXP_s103
                                 + 0.790891 w_k1_DIS_s103
                                 + 0.864669 w_k1_FDX_s103
                                 + 1.37779 w_k1_IBM_s103
                                 + 1.11797 w_k1_INTC_s103
                                 + 1.15011 w_k1_KO_s103 + 1.31761 w_k1_MAT_s103
                                 + 0.930917 w_k1_MSFT_s103
                                 + 1.10333 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.995965 w_k1_ADM_s103
                                 + 1.47732 w_k1_AXP_s103
                                 + 1.63165 w_k1_DIS_s103
                                 + 1.32626 w_k1_FDX_s103
                                 + 1.46489 w_k1_IBM_s103
                                 + 1.43146 w_k1_INTC_s103
                                 + 1.63022 w_k1_KO_s103 + 1.32055 w_k1_MAT_s103
                                 + 1.32883 w_k1_MSFT_s103
                                 + 1.24655 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.833046 w_k1_ADM_s103
                                 + 0.880463 w_k1_AXP_s103
                                 + 0.81762 w_k1_DIS_s103
                                 + 0.946148 w_k1_FDX_s103
                                 + 0.821578 w_k1_IBM_s103
                                 + 1.10926 w_k1_INTC_s103
                                 + 0.884837 w_k1_KO_s103
                                 + 0.568836 w_k1_MAT_s103
                                 + 0.973248 w_k1_MSFT_s103
                                 + 0.894415 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.11589 w_k1_ADM_s103
                                 + 1.26195 w_k1_AXP_s103
                                 + 1.30641 w_k1_DIS_s103
                                 + 1.26419 w_k1_FDX_s103
                                 + 0.798551 w_k1_IBM_s103
                                 + 1.28072 w_k1_INTC_s103
                                 + 1.06556 w_k1_KO_s103
                                 + 0.801946 w_k1_MAT_s103
                                 + 1.31591 w_k1_MSFT_s103
                                 + 1.11003 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.847973 w_k1_ADM_s103
                                 + 0.629593 w_k1_AXP_s103
                                 + 0.551226 w_k1_DIS_s103
                                 + 0.821815 w_k1_FDX_s103
                                 + 0.651932 w_k1_IBM_s103
                                 + 0.871946 w_k1_INTC_s103
                                 + 0.789506 w_k1_KO_s103
                                 + 0.576086 w_k1_MAT_s103
                                 + 0.788107 w_k1_MSFT_s103
                                 + 0.788449 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.949235 w_k1_ADM_s103
                                 + 1.09936 w_k1_AXP_s103
                                 + 1.23377 w_k1_DIS_s103
                                 + 1.19462 w_k1_FDX_s103
                                 + 0.982166 w_k1_IBM_s103
                                 + 1.36208 w_k1_INTC_s103
                                 + 1.29856 w_k1_KO_s103
                                 + 0.936882 w_k1_MAT_s103
                                 + 1.28889 w_k1_MSFT_s103
                                 + 1.20835 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.757802 w_k1_ADM_s103
                                 + 0.739255 w_k1_AXP_s103
                                 + 0.848443 w_k1_DIS_s103
                                 + 0.828733 w_k1_FDX_s103
                                 + 1.14287 w_k1_IBM_s103
                                 + 0.89896 w_k1_INTC_s103
                                 + 0.908045 w_k1_KO_s103
                                 + 0.869888 w_k1_MAT_s103
                                 + 0.826713 w_k1_MSFT_s103
                                 + 0.958346 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.937368 w_k1_ADM_s103
                                 + 1.0516 w_k1_AXP_s103
                                 + 0.856604 w_k1_DIS_s103
                                 + 1.02854 w_k1_FDX_s103
                                 + 1.04231 w_k1_IBM_s103
                                 + 1.29704 w_k1_INTC_s103
                                 + 1.03211 w_k1_KO_s103
                                 + 0.783561 w_k1_MAT_s103
                                 + 1.05457 w_k1_MSFT_s103
                                 + 1.03031 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.799408 w_k1_ADM_s104
                                 + 0.83602 w_k1_AXP_s104 + 0.6721 w_k1_DIS_s104
                                 + 0.875966 w_k1_FDX_s104
                                 + 0.709812 w_k1_IBM_s104
                                 + 0.860258 w_k1_INTC_s104
                                 + 0.764302 w_k1_KO_s104
                                 + 0.865057 w_k1_MAT_s104
                                 + 0.80423 w_k1_MSFT_s104
                                 + 0.958061 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.957358 w_k1_ADM_s104
                                 + 1.05656 w_k1_AXP_s104
                                 + 1.22479 w_k1_DIS_s104
                                 + 1.15667 w_k1_FDX_s104
                                 + 1.23757 w_k1_IBM_s104
                                 + 0.887824 w_k1_INTC_s104
                                 + 0.994985 w_k1_KO_s104
                                 + 1.29944 w_k1_MAT_s104
                                 + 0.973404 w_k1_MSFT_s104
                                 + 1.24459 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.1095 w_k1_ADM_s104
                                 + 1.47362 w_k1_AXP_s104 + 1.2009 w_k1_DIS_s104
                                 + 1.23816 w_k1_FDX_s104
                                 + 1.53132 w_k1_IBM_s104
                                 + 1.05154 w_k1_INTC_s104
                                 + 1.13502 w_k1_KO_s104 + 1.58364 w_k1_MAT_s104
                                 + 1.1233 w_k1_MSFT_s104 + 1.02707 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.30545 w_k1_ADM_s104
                                 + 1.5228 w_k1_AXP_s104 + 1.54657 w_k1_DIS_s104
                                 + 1.33969 w_k1_FDX_s104
                                 + 1.13759 w_k1_IBM_s104
                                 + 1.22649 w_k1_INTC_s104
                                 + 1.10773 w_k1_KO_s104 + 1.41229 w_k1_MAT_s104
                                 + 1.16901 w_k1_MSFT_s104
                                 + 1.25649 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.969949 w_k1_ADM_s104
                                 + 1.24835 w_k1_AXP_s104
                                 + 0.863098 w_k1_DIS_s104
                                 + 1.08295 w_k1_FDX_s104
                                 + 1.01113 w_k1_IBM_s104
                                 + 0.897895 w_k1_INTC_s104
                                 + 0.885028 w_k1_KO_s104
                                 + 0.996108 w_k1_MAT_s104
                                 + 0.856433 w_k1_MSFT_s104
                                 + 0.937468 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.543599 w_k1_ADM_s104
                                 + 0.967128 w_k1_AXP_s104
                                 + 0.789539 w_k1_DIS_s104
                                 + 0.860164 w_k1_FDX_s104
                                 + 1.12059 w_k1_IBM_s104
                                 + 0.687754 w_k1_INTC_s104
                                 + 0.902069 w_k1_KO_s104
                                 + 1.33616 w_k1_MAT_s104
                                 + 0.785345 w_k1_MSFT_s104
                                 + 0.96465 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.651937 w_k1_ADM_s104
                                 + 0.631597 w_k1_AXP_s104
                                 + 0.867033 w_k1_DIS_s104
                                 + 0.883471 w_k1_FDX_s104
                                 + 0.90689 w_k1_IBM_s104
                                 + 0.996452 w_k1_INTC_s104
                                 + 0.790727 w_k1_KO_s104
                                 + 0.691064 w_k1_MAT_s104
                                 + 1.0386 w_k1_MSFT_s104
                                 + 0.894804 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.22692 w_k1_ADM_s104
                                 + 1.15264 w_k1_AXP_s104
                                 + 1.12474 w_k1_DIS_s104
                                 + 1.23959 w_k1_FDX_s104
                                 + 0.764695 w_k1_IBM_s104
                                 + 1.0784 w_k1_INTC_s104
                                 + 0.828411 w_k1_KO_s104
                                 + 0.621833 w_k1_MAT_s104
                                 + 0.968463 w_k1_MSFT_s104
                                 + 1.02758 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.4669 w_k1_ADM_s105
                                 + 1.04512 w_k1_AXP_s105 + 1.3246 w_k1_DIS_s105
                                 + 1.13095 w_k1_FDX_s105
                                 + 0.887653 w_k1_IBM_s105
                                 + 1.01475 w_k1_INTC_s105
                                 + 1.21057 w_k1_KO_s105
                                 + 0.582992 w_k1_MAT_s105
                                 + 0.991435 w_k1_MSFT_s105
                                 + 1.18797 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.03098 w_k1_ADM_s105
                                 + 1.16219 w_k1_AXP_s105 + 1.0469 w_k1_DIS_s105
                                 + 1.11621 w_k1_FDX_s105
                                 + 0.866838 w_k1_IBM_s105
                                 + 0.966897 w_k1_INTC_s105
                                 + 1.12755 w_k1_KO_s105 + 1.17095 w_k1_MAT_s105
                                 + 0.987742 w_k1_MSFT_s105
                                 + 1.15964 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.14264 w_k1_ADM_s105
                                 + 1.39987 w_k1_AXP_s105
                                 + 1.05354 w_k1_DIS_s105
                                 + 1.01389 w_k1_FDX_s105
                                 + 1.13144 w_k1_IBM_s105
                                 + 0.916661 w_k1_INTC_s105
                                 + 1.25062 w_k1_KO_s105 + 1.5401 w_k1_MAT_s105
                                 + 0.922107 w_k1_MSFT_s105
                                 + 1.09807 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.906707 w_k1_ADM_s105
                                 + 0.70039 w_k1_AXP_s105
                                 + 0.842165 w_k1_DIS_s105
                                 + 0.803028 w_k1_FDX_s105
                                 + 0.874978 w_k1_IBM_s105
                                 + 1.04608 w_k1_INTC_s105
                                 + 1.04645 w_k1_KO_s105
                                 + 0.669106 w_k1_MAT_s105
                                 + 0.956504 w_k1_MSFT_s105
                                 + 0.855499 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.35358 w_k1_ADM_s105
                                 + 1.6826 w_k1_AXP_s105 + 1.57973 w_k1_DIS_s105
                                 + 1.48693 w_k1_FDX_s105
                                 + 1.35815 w_k1_IBM_s105
                                 + 1.19238 w_k1_INTC_s105
                                 + 1.15394 w_k1_KO_s105 + 1.24527 w_k1_MAT_s105
                                 + 1.20841 w_k1_MSFT_s105
                                 + 1.13164 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.101 w_k1_ADM_s105
                                 + 1.02218 w_k1_AXP_s105
                                 + 0.70526 w_k1_DIS_s105
                                 + 0.920494 w_k1_FDX_s105
                                 + 0.837143 w_k1_IBM_s105
                                 + 0.801628 w_k1_INTC_s105
                                 + 0.841499 w_k1_KO_s105
                                 + 0.773719 w_k1_MAT_s105
                                 + 0.786816 w_k1_MSFT_s105
                                 + 0.854802 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.874889 w_k1_ADM_s105
                                 + 0.937884 w_k1_AXP_s105
                                 + 0.783408 w_k1_DIS_s105
                                 + 0.779832 w_k1_FDX_s105
                                 + 1.35451 w_k1_IBM_s105
                                 + 0.738147 w_k1_INTC_s105
                                 + 1.00947 w_k1_KO_s105 + 1.54519 w_k1_MAT_s105
                                 + 0.758506 w_k1_MSFT_s105
                                 + 0.957039 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.11552 w_k1_ADM_s105
                                 + 1.12034 w_k1_AXP_s105
                                 + 1.45425 w_k1_DIS_s105
                                 + 1.00421 w_k1_FDX_s105
                                 + 1.28285 w_k1_IBM_s105
                                 + 0.935401 w_k1_INTC_s105
                                 + 1.25152 w_k1_KO_s105 + 1.07301 w_k1_MAT_s105
                                 + 1.10223 w_k1_MSFT_s105
                                 + 1.17759 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.0119 w_k1_ADM_s106
                                 + 0.925214 w_k1_AXP_s106
                                 + 0.809644 w_k1_DIS_s106
                                 + 0.933196 w_k1_FDX_s106
                                 + 1.00277 w_k1_IBM_s106
                                 + 0.816107 w_k1_INTC_s106
                                 + 0.990968 w_k1_KO_s106
                                 + 1.12066 w_k1_MAT_s106
                                 + 0.904507 w_k1_MSFT_s106
                                 + 1.11813 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06935 w_k1_ADM_s106
                                 + 1.32568 w_k1_AXP_s106
                                 + 1.53807 w_k1_DIS_s106
                                 + 1.29401 w_k1_FDX_s106
                                 + 1.03202 w_k1_IBM_s106
                                 + 1.11425 w_k1_INTC_s106
                                 + 1.11239 w_k1_KO_s106 + 1.46078 w_k1_MAT_s106
                                 + 1.15712 w_k1_MSFT_s106
                                 + 1.07838 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.00598 w_k1_ADM_s106
                                 + 1.00283 w_k1_AXP_s106
                                 + 0.854035 w_k1_DIS_s106
                                 + 0.918891 w_k1_FDX_s106
                                 + 1.26607 w_k1_IBM_s106
                                 + 0.951124 w_k1_INTC_s106
                                 + 1.04288 w_k1_KO_s106 + 1.14686 w_k1_MAT_s106
                                 + 0.919347 w_k1_MSFT_s106
                                 + 1.01002 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.2521 w_k1_ADM_s106
                                 + 1.19824 w_k1_AXP_s106
                                 + 1.30354 w_k1_DIS_s106
                                 + 1.16095 w_k1_FDX_s106
                                 + 1.31668 w_k1_IBM_s106
                                 + 0.996548 w_k1_INTC_s106
                                 + 1.20454 w_k1_KO_s106 + 1.68786 w_k1_MAT_s106
                                 + 1.12126 w_k1_MSFT_s106
                                 + 1.36531 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.922665 w_k1_ADM_s106
                                 + 1.1807 w_k1_AXP_s106 + 1.15786 w_k1_DIS_s106
                                 + 1.0417 w_k1_FDX_s106 + 1.7003 w_k1_IBM_s106
                                 + 0.995273 w_k1_INTC_s106
                                 + 1.39986 w_k1_KO_s106 + 1.86345 w_k1_MAT_s106
                                 + 0.973316 w_k1_MSFT_s106
                                 + 1.19253 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.72183 w_k1_ADM_s106
                                 + 0.646603 w_k1_AXP_s106
                                 + 0.710198 w_k1_DIS_s106
                                 + 0.75707 w_k1_FDX_s106 + 1.1494 w_k1_IBM_s106
                                 + 0.806439 w_k1_INTC_s106
                                 + 0.928218 w_k1_KO_s106 + 1.2529 w_k1_MAT_s106
                                 + 0.845187 w_k1_MSFT_s106
                                 + 0.899094 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.921335 w_k1_ADM_s106
                                 + 0.814902 w_k1_AXP_s106
                                 + 1.00208 w_k1_DIS_s106 + 1.0342 w_k1_FDX_s106
                                 + 0.735316 w_k1_IBM_s106
                                 + 1.04752 w_k1_INTC_s106 + 1.018 w_k1_KO_s106
                                 + 0.783356 w_k1_MAT_s106
                                 + 1.15737 w_k1_MSFT_s106
                                 + 1.07788 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.729715 w_k1_ADM_s106
                                 + 0.708938 w_k1_AXP_s106
                                 + 0.713648 w_k1_DIS_s106
                                 + 0.753403 w_k1_FDX_s106
                                 + 0.701457 w_k1_IBM_s106
                                 + 0.823432 w_k1_INTC_s106
                                 + 0.731483 w_k1_KO_s106
                                 + 0.866282 w_k1_MAT_s106
                                 + 0.91333 w_k1_MSFT_s106
                                 + 0.956411 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.732043 w_k1_ADM_s107
                                 + 0.742962 w_k1_AXP_s107
                                 + 0.813632 w_k1_DIS_s107
                                 + 0.8005 w_k1_FDX_s107 + 1.16327 w_k1_IBM_s107
                                 + 0.851038 w_k1_INTC_s107
                                 + 0.843892 w_k1_KO_s107
                                 + 1.21055 w_k1_MAT_s107
                                 + 0.728144 w_k1_MSFT_s107
                                 + 0.949173 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.810336 w_k1_ADM_s107
                                 + 0.851759 w_k1_AXP_s107
                                 + 1.13461 w_k1_DIS_s107
                                 + 1.08002 w_k1_FDX_s107
                                 + 0.674181 w_k1_IBM_s107
                                 + 1.21244 w_k1_INTC_s107
                                 + 0.817866 w_k1_KO_s107
                                 + 0.661296 w_k1_MAT_s107
                                 + 0.951435 w_k1_MSFT_s107
                                 + 0.86097 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.85372 w_k1_ADM_s107
                                 + 1.02592 w_k1_AXP_s107
                                 + 0.773117 w_k1_DIS_s107
                                 + 0.98058 w_k1_FDX_s107
                                 + 0.609592 w_k1_IBM_s107
                                 + 1.1292 w_k1_INTC_s107
                                 + 0.589157 w_k1_KO_s107
                                 + 0.58826 w_k1_MAT_s107
                                 + 0.753227 w_k1_MSFT_s107
                                 + 0.852562 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.840297 w_k1_ADM_s107
                                 + 1.13962 w_k1_AXP_s107
                                 + 1.29326 w_k1_DIS_s107
                                 + 1.06265 w_k1_FDX_s107
                                 + 1.40914 w_k1_IBM_s107
                                 + 1.29391 w_k1_INTC_s107
                                 + 1.09903 w_k1_KO_s107 + 1.15097 w_k1_MAT_s107
                                 + 1.137 w_k1_MSFT_s107 + 1.16019 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.921645 w_k1_ADM_s107
                                 + 1.22007 w_k1_AXP_s107
                                 + 0.865571 w_k1_DIS_s107
                                 + 1.07867 w_k1_FDX_s107
                                 + 0.894712 w_k1_IBM_s107
                                 + 1.03072 w_k1_INTC_s107
                                 + 0.720121 w_k1_KO_s107
                                 + 1.12757 w_k1_MAT_s107
                                 + 0.876169 w_k1_MSFT_s107
                                 + 1.01553 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.727828 w_k1_ADM_s107
                                 + 0.745471 w_k1_AXP_s107
                                 + 0.731581 w_k1_DIS_s107
                                 + 0.803739 w_k1_FDX_s107
                                 + 0.894934 w_k1_IBM_s107
                                 + 0.99787 w_k1_INTC_s107
                                 + 0.821027 w_k1_KO_s107 + 0.7992 w_k1_MAT_s107
                                 + 0.874603 w_k1_MSFT_s107
                                 + 0.900063 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.45183 w_k1_ADM_s107
                                 + 1.82038 w_k1_AXP_s107
                                 + 1.55359 w_k1_DIS_s107
                                 + 1.33574 w_k1_FDX_s107
                                 + 1.71195 w_k1_IBM_s107
                                 + 1.85696 w_k1_INTC_s107
                                 + 1.41825 w_k1_KO_s107 + 1.3288 w_k1_MAT_s107
                                 + 1.15917 w_k1_MSFT_s107
                                 + 1.25228 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.61082 w_k1_ADM_s107
                                 + 0.531579 w_k1_AXP_s107
                                 + 0.621222 w_k1_DIS_s107
                                 + 0.689754 w_k1_FDX_s107
                                 + 0.636721 w_k1_IBM_s107
                                 + 0.775895 w_k1_INTC_s107
                                 + 0.545082 w_k1_KO_s107
                                 + 0.435339 w_k1_MAT_s107
                                 + 0.651519 w_k1_MSFT_s107
                                 + 0.671696 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.80752 w_k1_ADM_s110
                                 + 0.664617 w_k1_AXP_s110
                                 + 0.688218 w_k1_DIS_s110
                                 + 0.67398 w_k1_FDX_s110
                                 + 0.957082 w_k1_IBM_s110
                                 + 0.921006 w_k1_INTC_s110
                                 + 0.850039 w_k1_KO_s110
                                 + 0.791398 w_k1_MAT_s110
                                 + 0.824473 w_k1_MSFT_s110
                                 + 0.834665 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.752924 w_k1_ADM_s110
                                 + 0.825785 w_k1_AXP_s110
                                 + 0.918937 w_k1_DIS_s110
                                 + 0.85857 w_k1_FDX_s110
                                 + 1.11043 w_k1_IBM_s110
                                 + 0.912839 w_k1_INTC_s110
                                 + 1.07258 w_k1_KO_s110 + 1.36253 w_k1_MAT_s110
                                 + 0.959778 w_k1_MSFT_s110
                                 + 1.01923 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.17884 w_k1_ADM_s110
                                 + 1.20254 w_k1_AXP_s110
                                 + 1.21247 w_k1_DIS_s110
                                 + 1.09047 w_k1_FDX_s110
                                 + 1.14924 w_k1_IBM_s110
                                 + 1.03824 w_k1_INTC_s110
                                 + 1.07953 w_k1_KO_s110 + 1.27034 w_k1_MAT_s110
                                 + 1.1664 w_k1_MSFT_s110 + 1.06403 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.08304 w_k1_ADM_s110
                                 + 0.944279 w_k1_AXP_s110
                                 + 0.882395 w_k1_DIS_s110
                                 + 1.03275 w_k1_FDX_s110
                                 + 0.713002 w_k1_IBM_s110
                                 + 1.17965 w_k1_INTC_s110
                                 + 0.849676 w_k1_KO_s110
                                 + 0.710603 w_k1_MAT_s110
                                 + 0.978326 w_k1_MSFT_s110
                                 + 0.970001 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.766086 w_k1_ADM_s110
                                 + 0.657284 w_k1_AXP_s110
                                 + 0.522536 w_k1_DIS_s110
                                 + 0.681006 w_k1_FDX_s110
                                 + 0.601434 w_k1_IBM_s110
                                 + 0.862888 w_k1_INTC_s110
                                 + 0.621026 w_k1_KO_s110
                                 + 0.728728 w_k1_MAT_s110
                                 + 0.685137 w_k1_MSFT_s110
                                 + 0.781191 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.852909 w_k1_ADM_s110
                                 + 0.646963 w_k1_AXP_s110
                                 + 0.839413 w_k1_DIS_s110
                                 + 0.867385 w_k1_FDX_s110
                                 + 0.700444 w_k1_IBM_s110
                                 + 0.95585 w_k1_INTC_s110
                                 + 0.799697 w_k1_KO_s110
                                 + 0.612174 w_k1_MAT_s110
                                 + 0.911829 w_k1_MSFT_s110
                                 + 0.784584 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.29796 w_k1_ADM_s110
                                 + 1.43789 w_k1_AXP_s110
                                 + 1.75671 w_k1_DIS_s110 + 1.3177 w_k1_FDX_s110
                                 + 1.60365 w_k1_IBM_s110
                                 + 1.28562 w_k1_INTC_s110
                                 + 1.69742 w_k1_KO_s110 + 2.99868 w_k1_MAT_s110
                                 + 1.36738 w_k1_MSFT_s110
                                 + 1.49309 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.990476 w_k1_ADM_s110
                                 + 1.20014 w_k1_AXP_s110
                                 + 0.963917 w_k1_DIS_s110
                                 + 1.00095 w_k1_FDX_s110
                                 + 1.07551 w_k1_IBM_s110
                                 + 0.8278 w_k1_INTC_s110
                                 + 0.875325 w_k1_KO_s110 + 1.4299 w_k1_MAT_s110
                                 + 0.862011 w_k1_MSFT_s110
                                 + 1.01843 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.831362 w_k1_ADM_s111
                                 + 1.04215 w_k1_AXP_s111
                                 + 1.06244 w_k1_DIS_s111
                                 + 0.914892 w_k1_FDX_s111
                                 + 1.31606 w_k1_IBM_s111
                                 + 0.934972 w_k1_INTC_s111
                                 + 1.1709 w_k1_KO_s111 + 1.36282 w_k1_MAT_s111
                                 + 0.967052 w_k1_MSFT_s111
                                 + 1.03794 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.782069 w_k1_ADM_s111
                                 + 0.824955 w_k1_AXP_s111
                                 + 0.75947 w_k1_DIS_s111
                                 + 0.675412 w_k1_FDX_s111
                                 + 1.06078 w_k1_IBM_s111
                                 + 0.853797 w_k1_INTC_s111
                                 + 0.955971 w_k1_KO_s111
                                 + 0.880435 w_k1_MAT_s111
                                 + 0.818473 w_k1_MSFT_s111
                                 + 0.802054 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.898655 w_k1_ADM_s111
                                 + 0.813622 w_k1_AXP_s111
                                 + 0.955059 w_k1_DIS_s111
                                 + 0.968625 w_k1_FDX_s111
                                 + 0.677009 w_k1_IBM_s111
                                 + 0.896904 w_k1_INTC_s111
                                 + 0.798056 w_k1_KO_s111
                                 + 0.651065 w_k1_MAT_s111
                                 + 0.919839 w_k1_MSFT_s111
                                 + 0.889679 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.06229 w_k1_ADM_s111
                                 + 0.841885 w_k1_AXP_s111
                                 + 1.21578 w_k1_DIS_s111
                                 + 0.977309 w_k1_FDX_s111
                                 + 0.888524 w_k1_IBM_s111
                                 + 0.918879 w_k1_INTC_s111
                                 + 1.16071 w_k1_KO_s111
                                 + 0.653759 w_k1_MAT_s111
                                 + 1.01243 w_k1_MSFT_s111
                                 + 0.939217 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.17125 w_k1_ADM_s111
                                 + 1.1537 w_k1_AXP_s111 + 1.07982 w_k1_DIS_s111
                                 + 1.11115 w_k1_FDX_s111
                                 + 1.02719 w_k1_IBM_s111
                                 + 1.20405 w_k1_INTC_s111
                                 + 1.15061 w_k1_KO_s111
                                 + 0.824592 w_k1_MAT_s111
                                 + 1.01291 w_k1_MSFT_s111 + 1.0602 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.945502 w_k1_ADM_s111
                                 + 1.53478 w_k1_AXP_s111
                                 + 1.89074 w_k1_DIS_s111
                                 + 1.28659 w_k1_FDX_s111
                                 + 1.59366 w_k1_IBM_s111
                                 + 1.26071 w_k1_INTC_s111
                                 + 1.75282 w_k1_KO_s111 + 1.32229 w_k1_MAT_s111
                                 + 1.38405 w_k1_MSFT_s111
                                 + 1.21661 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.0717 w_k1_ADM_s111
                                 + 1.42071 w_k1_AXP_s111
                                 + 1.22102 w_k1_DIS_s111
                                 + 1.11617 w_k1_FDX_s111
                                 + 0.930557 w_k1_IBM_s111
                                 + 1.20275 w_k1_INTC_s111
                                 + 1.16005 w_k1_KO_s111 + 1.19613 w_k1_MAT_s111
                                 + 1.18214 w_k1_MSFT_s111
                                 + 1.10392 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.761748 w_k1_ADM_s111
                                 + 0.597261 w_k1_AXP_s111
                                 + 0.659808 w_k1_DIS_s111
                                 + 0.721896 w_k1_FDX_s111
                                 + 0.66981 w_k1_IBM_s111
                                 + 0.654391 w_k1_INTC_s111
                                 + 0.827187 w_k1_KO_s111
                                 + 0.551878 w_k1_MAT_s111
                                 + 0.766954 w_k1_MSFT_s111
                                 + 0.803437 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.867361 w_k1_ADM_s112
                                 + 0.984714 w_k1_AXP_s112
                                 + 0.765049 w_k1_DIS_s112
                                 + 0.929913 w_k1_FDX_s112
                                 + 0.632806 w_k1_IBM_s112
                                 + 0.813162 w_k1_INTC_s112
                                 + 0.716887 w_k1_KO_s112
                                 + 0.853739 w_k1_MAT_s112
                                 + 0.89174 w_k1_MSFT_s112
                                 + 0.810151 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.0094 w_k1_ADM_s112
                                 + 1.15852 w_k1_AXP_s112
                                 + 0.823049 w_k1_DIS_s112
                                 + 0.97557 w_k1_FDX_s112
                                 + 1.01093 w_k1_IBM_s112
                                 + 0.794607 w_k1_INTC_s112
                                 + 0.869774 w_k1_KO_s112
                                 + 1.11296 w_k1_MAT_s112
                                 + 0.852561 w_k1_MSFT_s112
                                 + 0.968441 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.771977 w_k1_ADM_s112
                                 + 0.975967 w_k1_AXP_s112
                                 + 0.813845 w_k1_DIS_s112
                                 + 0.816371 w_k1_FDX_s112
                                 + 1.43034 w_k1_IBM_s112
                                 + 0.826413 w_k1_INTC_s112
                                 + 1.09411 w_k1_KO_s112 + 1.36996 w_k1_MAT_s112
                                 + 0.859822 w_k1_MSFT_s112
                                 + 0.927263 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.70398 w_k1_ADM_s112
                                 + 0.635552 w_k1_AXP_s112
                                 + 0.555798 w_k1_DIS_s112
                                 + 0.641743 w_k1_FDX_s112
                                 + 0.774787 w_k1_IBM_s112
                                 + 0.613477 w_k1_INTC_s112
                                 + 0.799856 w_k1_KO_s112
                                 + 0.854872 w_k1_MAT_s112
                                 + 0.689292 w_k1_MSFT_s112
                                 + 0.738967 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.805186 w_k1_ADM_s112
                                 + 0.916578 w_k1_AXP_s112
                                 + 0.746294 w_k1_DIS_s112
                                 + 0.740852 w_k1_FDX_s112
                                 + 0.924147 w_k1_IBM_s112
                                 + 0.864956 w_k1_INTC_s112
                                 + 1.00193 w_k1_KO_s112 + 1.10801 w_k1_MAT_s112
                                 + 0.843109 w_k1_MSFT_s112
                                 + 0.844103 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.12462 w_k1_ADM_s112
                                 + 1.00008 w_k1_AXP_s112
                                 + 1.03166 w_k1_DIS_s112
                                 + 1.04167 w_k1_FDX_s112
                                 + 0.827623 w_k1_IBM_s112
                                 + 1.09378 w_k1_INTC_s112
                                 + 1.07623 w_k1_KO_s112
                                 + 0.957995 w_k1_MAT_s112
                                 + 1.16563 w_k1_MSFT_s112
                                 + 1.02117 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.888666 w_k1_ADM_s112
                                 + 1.09415 w_k1_AXP_s112
                                 + 1.20115 w_k1_DIS_s112
                                 + 1.10286 w_k1_FDX_s112 + 1.1182 w_k1_IBM_s112
                                 + 0.903973 w_k1_INTC_s112
                                 + 1.16728 w_k1_KO_s112 + 1.73967 w_k1_MAT_s112
                                 + 1.07437 w_k1_MSFT_s112
                                 + 1.03608 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.57054 w_k1_ADM_s112
                                 + 1.72016 w_k1_AXP_s112
                                 + 1.52109 w_k1_DIS_s112
                                 + 1.35608 w_k1_FDX_s112
                                 + 1.05457 w_k1_IBM_s112
                                 + 1.03471 w_k1_INTC_s112
                                 + 1.33119 w_k1_KO_s112 + 1.59398 w_k1_MAT_s112
                                 + 1.25731 w_k1_MSFT_s112
                                 + 1.17677 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.862222 w_k1_ADM_s113
                                 + 0.68523 w_k1_AXP_s113
                                 + 0.790764 w_k1_DIS_s113
                                 + 0.688542 w_k1_FDX_s113
                                 + 0.988468 w_k1_IBM_s113
                                 + 0.632147 w_k1_INTC_s113
                                 + 0.744234 w_k1_KO_s113
                                 + 0.879335 w_k1_MAT_s113
                                 + 0.615131 w_k1_MSFT_s113
                                 + 0.783357 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.10766 w_k1_ADM_s113
                                 + 0.646307 w_k1_AXP_s113
                                 + 1.33686 w_k1_DIS_s113
                                 + 1.04151 w_k1_FDX_s113
                                 + 1.12063 w_k1_IBM_s113
                                 + 1.02954 w_k1_INTC_s113
                                 + 1.11033 w_k1_KO_s113 + 1.01696 w_k1_MAT_s113
                                 + 1.14145 w_k1_MSFT_s113
                                 + 1.01015 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.760334 w_k1_ADM_s113
                                 + 0.965044 w_k1_AXP_s113
                                 + 1.01614 w_k1_DIS_s113
                                 + 0.850803 w_k1_FDX_s113
                                 + 1.3372 w_k1_IBM_s113
                                 + 0.958697 w_k1_INTC_s113
                                 + 1.0425 w_k1_KO_s113 + 1.192 w_k1_MAT_s113
                                 + 0.806402 w_k1_MSFT_s113
                                 + 0.936921 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.949969 w_k1_ADM_s113
                                 + 1.09055 w_k1_AXP_s113 + 1.7129 w_k1_DIS_s113
                                 + 1.26343 w_k1_FDX_s113
                                 + 1.38623 w_k1_IBM_s113
                                 + 1.10845 w_k1_INTC_s113
                                 + 1.46381 w_k1_KO_s113 + 1.61514 w_k1_MAT_s113
                                 + 1.14103 w_k1_MSFT_s113
                                 + 1.05675 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.797517 w_k1_ADM_s113
                                 + 0.758401 w_k1_AXP_s113
                                 + 0.700546 w_k1_DIS_s113
                                 + 0.840872 w_k1_FDX_s113
                                 + 0.648223 w_k1_IBM_s113
                                 + 0.843048 w_k1_INTC_s113
                                 + 0.588673 w_k1_KO_s113
                                 + 0.625218 w_k1_MAT_s113
                                 + 0.736499 w_k1_MSFT_s113
                                 + 0.809658 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.03112 w_k1_ADM_s113
                                 + 0.910096 w_k1_AXP_s113
                                 + 1.02487 w_k1_DIS_s113
                                 + 0.950816 w_k1_FDX_s113
                                 + 0.843073 w_k1_IBM_s113
                                 + 0.957386 w_k1_INTC_s113
                                 + 0.820654 w_k1_KO_s113
                                 + 0.711091 w_k1_MAT_s113
                                 + 0.981288 w_k1_MSFT_s113
                                 + 0.81204 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.20468 w_k1_ADM_s113
                                 + 1.16767 w_k1_AXP_s113
                                 + 1.18393 w_k1_DIS_s113
                                 + 1.23236 w_k1_FDX_s113
                                 + 0.856854 w_k1_IBM_s113
                                 + 1.21997 w_k1_INTC_s113
                                 + 0.859209 w_k1_KO_s113
                                 + 0.901969 w_k1_MAT_s113
                                 + 1.07881 w_k1_MSFT_s113
                                 + 0.877372 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.38371 w_k1_ADM_s113
                                 + 1.34991 w_k1_AXP_s113
                                 + 1.66705 w_k1_DIS_s113
                                 + 1.53783 w_k1_FDX_s113
                                 + 0.949721 w_k1_IBM_s113
                                 + 1.19799 w_k1_INTC_s113
                                 + 1.01833 w_k1_KO_s113 + 1.11473 w_k1_MAT_s113
                                 + 1.29905 w_k1_MSFT_s113
                                 + 1.16502 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.47493 w_k1_ADM_s114
                                 + 1.66327 w_k1_AXP_s114
                                 + 1.71448 w_k1_DIS_s114
                                 + 1.34048 w_k1_FDX_s114
                                 + 1.53993 w_k1_IBM_s114
                                 + 1.38792 w_k1_INTC_s114 + 1.6702 w_k1_KO_s114
                                 + 1.8135 w_k1_MAT_s114
                                 + 1.32168 w_k1_MSFT_s114
                                 + 1.14512 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.972617 w_k1_ADM_s114
                                 + 1.17577 w_k1_AXP_s114
                                 + 1.06079 w_k1_DIS_s114
                                 + 0.92853 w_k1_FDX_s114
                                 + 1.35005 w_k1_IBM_s114
                                 + 1.09944 w_k1_INTC_s114 + 1.2099 w_k1_KO_s114
                                 + 1.38951 w_k1_MAT_s114
                                 + 0.987004 w_k1_MSFT_s114
                                 + 1.00158 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.17654 w_k1_ADM_s114
                                 + 1.5643 w_k1_AXP_s114 + 1.29105 w_k1_DIS_s114
                                 + 1.15682 w_k1_FDX_s114
                                 + 1.55169 w_k1_IBM_s114
                                 + 1.21107 w_k1_INTC_s114
                                 + 1.38192 w_k1_KO_s114 + 1.78472 w_k1_MAT_s114
                                 + 1.02487 w_k1_MSFT_s114
                                 + 1.15934 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.886723 w_k1_ADM_s114
                                 + 0.710547 w_k1_AXP_s114
                                 + 0.889335 w_k1_DIS_s114
                                 + 0.991771 w_k1_FDX_s114
                                 + 0.750117 w_k1_IBM_s114
                                 + 0.991497 w_k1_INTC_s114
                                 + 0.932671 w_k1_KO_s114
                                 + 0.571635 w_k1_MAT_s114
                                 + 1.01106 w_k1_MSFT_s114
                                 + 0.864133 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.691813 w_k1_ADM_s114
                                 + 0.664145 w_k1_AXP_s114
                                 + 0.640195 w_k1_DIS_s114
                                 + 0.73035 w_k1_FDX_s114
                                 + 0.805257 w_k1_IBM_s114
                                 + 0.691401 w_k1_INTC_s114
                                 + 0.801593 w_k1_KO_s114
                                 + 1.02167 w_k1_MAT_s114
                                 + 0.657541 w_k1_MSFT_s114
                                 + 0.789038 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.999988 w_k1_ADM_s114
                                 + 1.10062 w_k1_AXP_s114
                                 + 0.761138 w_k1_DIS_s114
                                 + 1.04239 w_k1_FDX_s114
                                 + 0.666463 w_k1_IBM_s114
                                 + 0.845214 w_k1_INTC_s114
                                 + 0.768248 w_k1_KO_s114
                                 + 1.02501 w_k1_MAT_s114
                                 + 0.951029 w_k1_MSFT_s114
                                 + 0.899856 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.2381 w_k1_ADM_s114
                                 + 1.17049 w_k1_AXP_s114
                                 + 1.54323 w_k1_DIS_s114
                                 + 1.33957 w_k1_FDX_s114
                                 + 0.970348 w_k1_IBM_s114
                                 + 1.35124 w_k1_INTC_s114
                                 + 1.36054 w_k1_KO_s114 + 1.12294 w_k1_MAT_s114
                                 + 1.25463 w_k1_MSFT_s114
                                 + 1.13957 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.907167 w_k1_ADM_s114
                                 + 0.999471 w_k1_AXP_s114
                                 + 0.984584 w_k1_DIS_s114
                                 + 1.02812 w_k1_FDX_s114
                                 + 0.927413 w_k1_IBM_s114
                                 + 1.00255 w_k1_INTC_s114 + 1.0985 w_k1_KO_s114
                                 + 1.19063 w_k1_MAT_s114
                                 + 0.895039 w_k1_MSFT_s114
                                 + 0.872443 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.979645 w_k1_ADM_s115
                                 + 0.944976 w_k1_AXP_s115
                                 + 0.660413 w_k1_DIS_s115
                                 + 0.843851 w_k1_FDX_s115
                                 + 0.794712 w_k1_IBM_s115
                                 + 0.885968 w_k1_INTC_s115
                                 + 0.753555 w_k1_KO_s115
                                 + 0.923715 w_k1_MAT_s115
                                 + 0.938819 w_k1_MSFT_s115
                                 + 0.788863 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.11555 w_k1_ADM_s115
                                 + 1.20345 w_k1_AXP_s115
                                 + 1.17229 w_k1_DIS_s115
                                 + 1.10053 w_k1_FDX_s115 + 1.1429 w_k1_IBM_s115
                                 + 1.0983 w_k1_INTC_s115 + 1.09639 w_k1_KO_s115
                                 + 1.05425 w_k1_MAT_s115
                                 + 1.1502 w_k1_MSFT_s115 + 1.04997 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.30728 w_k1_ADM_s115
                                 + 1.09174 w_k1_AXP_s115
                                 + 1.04826 w_k1_DIS_s115
                                 + 1.21436 w_k1_FDX_s115
                                 + 0.743816 w_k1_IBM_s115
                                 + 1.24221 w_k1_INTC_s115
                                 + 0.96801 w_k1_KO_s115 + 0.71972 w_k1_MAT_s115
                                 + 1.11721 w_k1_MSFT_s115 + 1.0854 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.12161 w_k1_ADM_s115
                                 + 0.900996 w_k1_AXP_s115
                                 + 0.669535 w_k1_DIS_s115
                                 + 0.854776 w_k1_FDX_s115
                                 + 1.12219 w_k1_IBM_s115
                                 + 0.839311 w_k1_INTC_s115
                                 + 0.827597 w_k1_KO_s115
                                 + 0.960947 w_k1_MAT_s115
                                 + 0.77146 w_k1_MSFT_s115
                                 + 0.903072 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.20748 w_k1_ADM_s115
                                 + 0.814117 w_k1_AXP_s115
                                 + 0.760998 w_k1_DIS_s115
                                 + 0.928568 w_k1_FDX_s115
                                 + 0.601886 w_k1_IBM_s115
                                 + 0.740872 w_k1_INTC_s115
                                 + 0.726999 w_k1_KO_s115
                                 + 0.620323 w_k1_MAT_s115
                                 + 0.808335 w_k1_MSFT_s115
                                 + 0.874892 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.35405 w_k1_ADM_s115
                                 + 1.74419 w_k1_AXP_s115
                                 + 1.20935 w_k1_DIS_s115
                                 + 1.25866 w_k1_FDX_s115
                                 + 1.43454 w_k1_IBM_s115
                                 + 1.16028 w_k1_INTC_s115
                                 + 1.35486 w_k1_KO_s115 + 1.77279 w_k1_MAT_s115
                                 + 1.19037 w_k1_MSFT_s115 + 1.194 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.16788 w_k1_ADM_s115
                                 + 1.16258 w_k1_AXP_s115 + 1.0482 w_k1_DIS_s115
                                 + 0.984517 w_k1_FDX_s115
                                 + 1.56924 w_k1_IBM_s115
                                 + 1.0088 w_k1_INTC_s115 + 1.1701 w_k1_KO_s115
                                 + 1.67914 w_k1_MAT_s115
                                 + 0.996798 w_k1_MSFT_s115
                                 + 1.14842 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.42173 w_k1_ADM_s115
                                 + 1.34251 w_k1_AXP_s115
                                 + 1.08517 w_k1_DIS_s115
                                 + 1.18991 w_k1_FDX_s115
                                 + 0.937483 w_k1_IBM_s115
                                 + 1.28187 w_k1_INTC_s115
                                 + 1.01078 w_k1_KO_s115 + 1.11534 w_k1_MAT_s115
                                 + 1.22282 w_k1_MSFT_s115 + 1.0575 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.19549 w_k1_ADM_s116
                                 + 1.31377 w_k1_AXP_s116
                                 + 1.76616 w_k1_DIS_s116
                                 + 1.25279 w_k1_FDX_s116
                                 + 1.58745 w_k1_IBM_s116
                                 + 1.15673 w_k1_INTC_s116
                                 + 1.48006 w_k1_KO_s116 + 1.75801 w_k1_MAT_s116
                                 + 1.32985 w_k1_MSFT_s116
                                 + 1.24841 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.789745 w_k1_ADM_s116
                                 + 0.983785 w_k1_AXP_s116
                                 + 0.823699 w_k1_DIS_s116
                                 + 0.821258 w_k1_FDX_s116
                                 + 0.988949 w_k1_IBM_s116
                                 + 0.951992 w_k1_INTC_s116
                                 + 0.92071 w_k1_KO_s116 + 1.09783 w_k1_MAT_s116
                                 + 0.822845 w_k1_MSFT_s116
                                 + 0.905445 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.800962 w_k1_ADM_s116
                                 + 0.951454 w_k1_AXP_s116
                                 + 0.905544 w_k1_DIS_s116
                                 + 0.941727 w_k1_FDX_s116
                                 + 0.776771 w_k1_IBM_s116
                                 + 0.821285 w_k1_INTC_s116
                                 + 0.790441 w_k1_KO_s116
                                 + 0.748033 w_k1_MAT_s116
                                 + 0.89333 w_k1_MSFT_s116
                                 + 0.891829 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.06322 w_k1_ADM_s116
                                 + 1.9124 w_k1_AXP_s116 + 1.17596 w_k1_DIS_s116
                                 + 1.18918 w_k1_FDX_s116
                                 + 1.36103 w_k1_IBM_s116
                                 + 1.32286 w_k1_INTC_s116
                                 + 1.15537 w_k1_KO_s116 + 1.886 w_k1_MAT_s116
                                 + 1.14153 w_k1_MSFT_s116
                                 + 1.13396 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.843487 w_k1_ADM_s116
                                 + 1.01272 w_k1_AXP_s116 + 0.8539 w_k1_DIS_s116
                                 + 0.888729 w_k1_FDX_s116
                                 + 1.29733 w_k1_IBM_s116
                                 + 0.857218 w_k1_INTC_s116
                                 + 1.07464 w_k1_KO_s116 + 1.52628 w_k1_MAT_s116
                                 + 0.961637 w_k1_MSFT_s116
                                 + 0.982858 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.01139 w_k1_ADM_s116
                                 + 1.29015 w_k1_AXP_s116
                                 + 1.24609 w_k1_DIS_s116 + 1.1689 w_k1_FDX_s116
                                 + 0.885243 w_k1_IBM_s116
                                 + 1.07119 w_k1_INTC_s116
                                 + 1.03493 w_k1_KO_s116 + 1.04188 w_k1_MAT_s116
                                 + 1.25736 w_k1_MSFT_s116
                                 + 1.14085 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.798384 w_k1_ADM_s116
                                 + 0.693664 w_k1_AXP_s116
                                 + 0.723244 w_k1_DIS_s116
                                 + 0.757301 w_k1_FDX_s116
                                 + 0.731455 w_k1_IBM_s116
                                 + 0.895693 w_k1_INTC_s116
                                 + 0.746667 w_k1_KO_s116
                                 + 0.564287 w_k1_MAT_s116
                                 + 0.727367 w_k1_MSFT_s116
                                 + 0.834383 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.07118 w_k1_ADM_s116
                                 + 1.01279 w_k1_AXP_s116 + 1.3488 w_k1_DIS_s116
                                 + 1.08563 w_k1_FDX_s116
                                 + 1.11616 w_k1_IBM_s116
                                 + 1.06845 w_k1_INTC_s116
                                 + 1.18031 w_k1_KO_s116 + 1.05724 w_k1_MAT_s116
                                 + 1.12111 w_k1_MSFT_s116 + 1.1513 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.36941 w_k1_ADM_s117
                                 + 1.40686 w_k1_AXP_s117
                                 + 1.46815 w_k1_DIS_s117
                                 + 1.39085 w_k1_FDX_s117
                                 + 1.43309 w_k1_IBM_s117
                                 + 1.29249 w_k1_INTC_s117
                                 + 1.43561 w_k1_KO_s117 + 1.55666 w_k1_MAT_s117
                                 + 1.40977 w_k1_MSFT_s117
                                 + 1.22341 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.47648 w_k1_ADM_s117
                                 + 1.16162 w_k1_AXP_s117
                                 + 1.33953 w_k1_DIS_s117
                                 + 1.34292 w_k1_FDX_s117
                                 + 0.859382 w_k1_IBM_s117
                                 + 1.00507 w_k1_INTC_s117
                                 + 1.06986 w_k1_KO_s117
                                 + 0.863852 w_k1_MAT_s117
                                 + 1.15701 w_k1_MSFT_s117
                                 + 1.04119 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.928635 w_k1_ADM_s117
                                 + 0.570546 w_k1_AXP_s117
                                 + 0.677518 w_k1_DIS_s117
                                 + 0.76623 w_k1_FDX_s117
                                 + 0.727589 w_k1_IBM_s117
                                 + 0.629778 w_k1_INTC_s117
                                 + 0.72048 w_k1_KO_s117
                                 + 0.615989 w_k1_MAT_s117
                                 + 0.828839 w_k1_MSFT_s117
                                 + 0.884374 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.22796 w_k1_ADM_s117
                                 + 0.842765 w_k1_AXP_s117
                                 + 0.787666 w_k1_DIS_s117
                                 + 1.02651 w_k1_FDX_s117
                                 + 0.614373 w_k1_IBM_s117
                                 + 0.714454 w_k1_INTC_s117
                                 + 0.681682 w_k1_KO_s117
                                 + 0.595507 w_k1_MAT_s117
                                 + 1.04379 w_k1_MSFT_s117
                                 + 0.881159 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.903488 w_k1_ADM_s117
                                 + 0.940329 w_k1_AXP_s117
                                 + 0.864515 w_k1_DIS_s117
                                 + 0.921705 w_k1_FDX_s117
                                 + 1.23395 w_k1_IBM_s117
                                 + 0.761357 w_k1_INTC_s117
                                 + 0.894113 w_k1_KO_s117
                                 + 1.31709 w_k1_MAT_s117
                                 + 0.989889 w_k1_MSFT_s117
                                 + 0.974361 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.937919 w_k1_ADM_s117
                                 + 0.714921 w_k1_AXP_s117
                                 + 0.767305 w_k1_DIS_s117
                                 + 0.818304 w_k1_FDX_s117
                                 + 0.860328 w_k1_IBM_s117
                                 + 0.723788 w_k1_INTC_s117
                                 + 0.720275 w_k1_KO_s117
                                 + 0.680437 w_k1_MAT_s117
                                 + 0.832835 w_k1_MSFT_s117
                                 + 0.864892 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.25497 w_k1_ADM_s117
                                 + 0.980293 w_k1_AXP_s117
                                 + 0.948215 w_k1_DIS_s117
                                 + 1.09334 w_k1_FDX_s117
                                 + 0.834132 w_k1_IBM_s117
                                 + 0.9173 w_k1_INTC_s117
                                 + 0.828026 w_k1_KO_s117
                                 + 0.858102 w_k1_MAT_s117
                                 + 1.02244 w_k1_MSFT_s117
                                 + 0.957358 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.968397 w_k1_ADM_s117
                                 + 1.18893 w_k1_AXP_s117
                                 + 1.17677 w_k1_DIS_s117
                                 + 1.12007 w_k1_FDX_s117 + 1.275 w_k1_IBM_s117
                                 + 1.04561 w_k1_INTC_s117
                                 + 1.10409 w_k1_KO_s117 + 1.67351 w_k1_MAT_s117
                                 + 1.1106 w_k1_MSFT_s117 + 0.99662 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0267222143019665 w_k1_ADM_s000
                                 + 0.0167228396753116 w_k1_ADM_s001
                                 + 0.0143121439597632 w_k1_ADM_s002
                                 + 0.0189921931442053 w_k1_ADM_s003
                                 + 0.0123838116964649 w_k1_ADM_s004
                                 + 0.0518822356116117 w_k1_ADM_s005
                                 + 0.0109720358225108 w_k1_ADM_s006
                                 + 0.0118277237005024 w_k1_ADM_s007
                                 + 0.0597408219146533 w_k1_ADM_s010
                                 + 0.0401184257269108 w_k1_ADM_s011
                                 + 0.0823770244002346 w_k1_ADM_s012
                                 + 0.0489867205834309 w_k1_ADM_s013
                                 + 0.0357854483447996 w_k1_ADM_s014
                                 + 0.037497118530722 w_k1_ADM_s015
                                 + 0.0690501475452436 w_k1_ADM_s016
                                 + 0.0949321399303685 w_k1_ADM_s017
                                 + 0.0265250156363272 w_k1_ADM_s100
                                 + 0.00794317653949407 w_k1_ADM_s101
                                 + 0.00605949211065976 w_k1_ADM_s102
                                 + 0.0101057607117735 w_k1_ADM_s103
                                 + 0.00959599246246448 w_k1_ADM_s104
                                 + 0.0228907042155001 w_k1_ADM_s105
                                 + 0.0123299926270873 w_k1_ADM_s106
                                 + 0.0155919488039161 w_k1_ADM_s107
                                 + 0.0163584749243306 w_k1_ADM_s110
                                 + 0.0330305169698685 w_k1_ADM_s111
                                 + 0.0267825845108915 w_k1_ADM_s112
                                 + 0.0278771673158737 w_k1_ADM_s113
                                 + 0.0568271694027726 w_k1_ADM_s114
                                 + 0.0467998565380387 w_k1_ADM_s115
                                 + 0.0368091081423908 w_k1_ADM_s116
                                 + 0.0404598498407644 w_k1_ADM_s117
                                 + 0.0288312268511517 w_k1_AXP_s000
                                 + 0.0154883896501435 w_k1_AXP_s001
                                 + 0.0159982857587394 w_k1_AXP_s002
                                 + 0.019841891058416 w_k1_AXP_s003
                                 + 0.0145030353663063 w_k1_AXP_s004
                                 + 0.0496025116712406 w_k1_AXP_s005
                                 + 0.0135199811791513 w_k1_AXP_s006
                                 + 0.0112709006687049 w_k1_AXP_s007
                                 + 0.0679456654690486 w_k1_AXP_s010
                                 + 0.0351771147918985 w_k1_AXP_s011
                                 + 0.0672871001266575 w_k1_AXP_s012
                                 + 0.0529489024784417 w_k1_AXP_s013
                                 + 0.0371522793237404 w_k1_AXP_s014
                                 + 0.0383278426000896 w_k1_AXP_s015
                                 + 0.0693149636738078 w_k1_AXP_s016
                                 + 0.0972767640406041 w_k1_AXP_s017
                                 + 0.0246938884559865 w_k1_AXP_s100
                                 + 0.00756475770253517 w_k1_AXP_s101
                                 + 0.00672759881385154 w_k1_AXP_s102
                                 + 0.0111356074687447 w_k1_AXP_s103
                                 + 0.0115963743623265 w_k1_AXP_s104
                                 + 0.0228125793820304 w_k1_AXP_s105
                                 + 0.0121087997854505 w_k1_AXP_s106
                                 + 0.0185654284643774 w_k1_AXP_s107
                                 + 0.016255953904197 w_k1_AXP_s110
                                 + 0.0354736094867526 w_k1_AXP_s111
                                 + 0.030011705322337 w_k1_AXP_s112
                                 + 0.0270314311588048 w_k1_AXP_s113
                                 + 0.0596143054884666 w_k1_AXP_s114
                                 + 0.0447410901569466 w_k1_AXP_s115
                                 + 0.043132369144783 w_k1_AXP_s116
                                 + 0.0369091931456154 w_k1_AXP_s117
                                 + 0.0260176193470822 w_k1_DIS_s000
                                 + 0.0171507335562016 w_k1_DIS_s001
                                 + 0.0152910360795729 w_k1_DIS_s002
                                 + 0.018672749911027 w_k1_DIS_s003
                                 + 0.0156349636845445 w_k1_DIS_s004
                                 + 0.0508722924139719 w_k1_DIS_s005
                                 + 0.0139031784733074 w_k1_DIS_s006
                                 + 0.0117973676173404 w_k1_DIS_s007
                                 + 0.0677434402172545 w_k1_DIS_s010
                                 + 0.0354880618212398 w_k1_DIS_s011
                                 + 0.0675896097889723 w_k1_DIS_s012
                                 + 0.0566286929832377 w_k1_DIS_s013
                                 + 0.041783674830184 w_k1_DIS_s014
                                 + 0.0380115984935005 w_k1_DIS_s015
                                 + 0.0714425257820968 w_k1_DIS_s016
                                 + 0.0929703831061987 w_k1_DIS_s017
                                 + 0.0253551435067693 w_k1_DIS_s100
                                 + 0.00710055243570443 w_k1_DIS_s101
                                 + 0.00681772363376105 w_k1_DIS_s102
                                 + 0.0106347049939593 w_k1_DIS_s103
                                 + 0.0110415840964893 w_k1_DIS_s104
                                 + 0.0214228549056172 w_k1_DIS_s105
                                 + 0.0126762134490586 w_k1_DIS_s106
                                 + 0.0180596992771572 w_k1_DIS_s107
                                 + 0.0169407288072472 w_k1_DIS_s110
                                 + 0.039044610407558 w_k1_DIS_s111
                                 + 0.0269243968358974 w_k1_DIS_s112
                                 + 0.0321708684387434 w_k1_DIS_s113
                                 + 0.0617199414169965 w_k1_DIS_s114
                                 + 0.0364956968556475 w_k1_DIS_s115
                                 + 0.0422945692011905 w_k1_DIS_s116
                                 + 0.036935323525347 w_k1_DIS_s117
                                 + 0.0261960465601819 w_k1_FDX_s000
                                 + 0.0170683093053874 w_k1_FDX_s001
                                 + 0.0136650237873254 w_k1_FDX_s002
                                 + 0.0186072714609319 w_k1_FDX_s003
                                 + 0.0150538201121015 w_k1_FDX_s004
                                 + 0.0497338053761345 w_k1_FDX_s005
                                 + 0.01307621576291 w_k1_FDX_s006
                                 + 0.0119005770866395 w_k1_FDX_s007
                                 + 0.0646568080338238 w_k1_FDX_s010
                                 + 0.0352784387715509 w_k1_FDX_s011
                                 + 0.0651778956496359 w_k1_FDX_s012
                                 + 0.0505086159809343 w_k1_FDX_s013
                                 + 0.0394512127596021 w_k1_FDX_s014
                                 + 0.0372583551609791 w_k1_FDX_s015
                                 + 0.0688324089975648 w_k1_FDX_s016
                                 + 0.0972209474956494 w_k1_FDX_s017
                                 + 0.0245182213357178 w_k1_FDX_s100
                                 + 0.00766696510428483 w_k1_FDX_s101
                                 + 0.00700075007570838 w_k1_FDX_s102
                                 + 0.0114384100568017 w_k1_FDX_s103
                                 + 0.0114212317470856 w_k1_FDX_s104
                                 + 0.0211204177620916 w_k1_FDX_s105
                                 + 0.0126144169406811 w_k1_FDX_s106
                                 + 0.0180357945752001 w_k1_FDX_s107
                                 + 0.0160590784205715 w_k1_FDX_s110
                                 + 0.0339480942030909 w_k1_FDX_s111
                                 + 0.0266824911341344 w_k1_FDX_s112
                                 + 0.0292122851988695 w_k1_FDX_s113
                                 + 0.0596163044492131 w_k1_FDX_s114
                                 + 0.0403354747791377 w_k1_FDX_s115
                                 + 0.0394404567343463 w_k1_FDX_s116
                                 + 0.0388546173219376 w_k1_FDX_s117
                                 + 0.0281108318293026 w_k1_IBM_s000
                                 + 0.0174635382623271 w_k1_IBM_s001
                                 + 0.01508207735754 w_k1_IBM_s002
                                 + 0.0192646560515846 w_k1_IBM_s003
                                 + 0.0145536790341856 w_k1_IBM_s004
                                 + 0.0474453066474238 w_k1_IBM_s005
                                 + 0.0130795574238432 w_k1_IBM_s006
                                 + 0.0128203872147751 w_k1_IBM_s007
                                 + 0.0639534605602056 w_k1_IBM_s010
                                 + 0.0357397941443959 w_k1_IBM_s011
                                 + 0.0659488235524556 w_k1_IBM_s012
                                 + 0.0514991957895851 w_k1_IBM_s013
                                 + 0.0409367463809752 w_k1_IBM_s014
                                 + 0.0388368444774981 w_k1_IBM_s015
                                 + 0.0694117660131802 w_k1_IBM_s016
                                 + 0.0967685983625268 w_k1_IBM_s017
                                 + 0.0257166108311774 w_k1_IBM_s100
                                 + 0.00812370744222563 w_k1_IBM_s101
                                 + 0.00641360530376696 w_k1_IBM_s102
                                 + 0.0116862792918131 w_k1_IBM_s103
                                 + 0.0118161591147826 w_k1_IBM_s104
                                 + 0.0209398362976975 w_k1_IBM_s105
                                 + 0.0129464102033597 w_k1_IBM_s106
                                 + 0.0187799689662507 w_k1_IBM_s107
                                 + 0.0170671206676945 w_k1_IBM_s110
                                 + 0.0350945814618458 w_k1_IBM_s111
                                 + 0.0266858173855296 w_k1_IBM_s112
                                 + 0.027262373203338 w_k1_IBM_s113
                                 + 0.0576620193126889 w_k1_IBM_s114
                                 + 0.0401264381577986 w_k1_IBM_s115
                                 + 0.0405514485982396 w_k1_IBM_s116
                                 + 0.0393371845341974 w_k1_IBM_s117
                                 + 0.0290985900008995 w_k1_INTC_s000
                                 + 0.017257145027158 w_k1_INTC_s001
                                 + 0.0135278805774209 w_k1_INTC_s002
                                 + 0.0188176802590063 w_k1_INTC_s003
                                 + 0.0137911409026214 w_k1_INTC_s004
                                 + 0.0431422478165483 w_k1_INTC_s005
                                 + 0.0137299771848505 w_k1_INTC_s006
                                 + 0.0112662663280103 w_k1_INTC_s007
                                 + 0.0628845558270398 w_k1_INTC_s010
                                 + 0.0405677687395919 w_k1_INTC_s011
                                 + 0.0617269975953013 w_k1_INTC_s012
                                 + 0.0522574605687807 w_k1_INTC_s013
                                 + 0.0387798190243876 w_k1_INTC_s014
                                 + 0.0368791317487167 w_k1_INTC_s015
                                 + 0.0661082514270702 w_k1_INTC_s016
                                 + 0.0935519024961911 w_k1_INTC_s017
                                 + 0.0207917067964703 w_k1_INTC_s100
                                 + 0.00759782392130672 w_k1_INTC_s101
                                 + 0.00695484720027531 w_k1_INTC_s102
                                 + 0.0131803424104336 w_k1_INTC_s103
                                 + 0.0102803554256038 w_k1_INTC_s104
                                 + 0.0197862626378025 w_k1_INTC_s105
                                 + 0.0118739528542701 w_k1_INTC_s106
                                 + 0.0206856541575157 w_k1_INTC_s107
                                 + 0.0166704055791136 w_k1_INTC_s110
                                 + 0.0342030124013623 w_k1_INTC_s111
                                 + 0.0233545236462674 w_k1_INTC_s112
                                 + 0.0281247681049756 w_k1_INTC_s113
                                 + 0.0604744140187531 w_k1_INTC_s114
                                 + 0.0409144510322425 w_k1_INTC_s115
                                 + 0.0394955590984169 w_k1_INTC_s116
                                 + 0.0332036816002336 w_k1_INTC_s117
                                 + 0.0258817085852016 w_k1_KO_s000
                                 + 0.0185961283046922 w_k1_KO_s001
                                 + 0.0143737776133139 w_k1_KO_s002
                                 + 0.0210255497541023 w_k1_KO_s003
                                 + 0.0144590820855242 w_k1_KO_s004
                                 + 0.0468208070844772 w_k1_KO_s005
                                 + 0.0139796602178244 w_k1_KO_s006
                                 + 0.0119575802617523 w_k1_KO_s007
                                 + 0.0659503112194694 w_k1_KO_s010
                                 + 0.0374344001879134 w_k1_KO_s011
                                 + 0.0674053612747225 w_k1_KO_s012
                                 + 0.0517686598703334 w_k1_KO_s013
                                 + 0.0409076990001254 w_k1_KO_s014
                                 + 0.0348259551716016 w_k1_KO_s015
                                 + 0.0762750205301076 w_k1_KO_s016
                                 + 0.0929288645959769 w_k1_KO_s017
                                 + 0.0248004187866998 w_k1_KO_s100
                                 + 0.00707814101086093 w_k1_KO_s101
                                 + 0.00665380252615996 w_k1_KO_s102
                                 + 0.0121863171590089 w_k1_KO_s103
                                 + 0.0101228427352528 w_k1_KO_s104
                                 + 0.0224342144359565 w_k1_KO_s105
                                 + 0.0130613806860419 w_k1_KO_s106
                                 + 0.0158129666017522 w_k1_KO_s107
                                 + 0.0168419494517145 w_k1_KO_s110
                                 + 0.0391922051599207 w_k1_KO_s111
                                 + 0.0279511650975828 w_k1_KO_s112
                                 + 0.0260922677563723 w_k1_KO_s113
                                 + 0.0642082641523176 w_k1_KO_s114
                                 + 0.0377218080349215 w_k1_KO_s115
                                 + 0.0399694914347483 w_k1_KO_s116
                                 + 0.034941249536463 w_k1_KO_s117
                                 + 0.0287667779758434 w_k1_MAT_s000
                                 + 0.0150787783482105 w_k1_MAT_s001
                                 + 0.0165244749464818 w_k1_MAT_s002
                                 + 0.0233054876205524 w_k1_MAT_s003
                                 + 0.0148838108611704 w_k1_MAT_s004
                                 + 0.0448124963687682 w_k1_MAT_s005
                                 + 0.0161171669866172 w_k1_MAT_s006
                                 + 0.0125985723253255 w_k1_MAT_s007
                                 + 0.0714558790946398 w_k1_MAT_s010
                                 + 0.0322336090758038 w_k1_MAT_s011
                                 + 0.0604156917236524 w_k1_MAT_s012
                                 + 0.0458934936012128 w_k1_MAT_s013
                                 + 0.0478403366955788 w_k1_MAT_s014
                                 + 0.0439250802343176 w_k1_MAT_s015
                                 + 0.0653453370001666 w_k1_MAT_s016
                                 + 0.100634662854745 w_k1_MAT_s017
                                 + 0.0242708252117239 w_k1_MAT_s100
                                 + 0.0071797817797736 w_k1_MAT_s101
                                 + 0.00520301701080356 w_k1_MAT_s102
                                 + 0.00996658961584066 w_k1_MAT_s103
                                 + 0.0123870644838618 w_k1_MAT_s104
                                 + 0.0209514509184255 w_k1_MAT_s105
                                 + 0.0149635745612165 w_k1_MAT_s106
                                 + 0.0175394690620024 w_k1_MAT_s107
                                 + 0.0208946270027223 w_k1_MAT_s110
                                 + 0.0321732966766712 w_k1_MAT_s111
                                 + 0.0348153859008571 w_k1_MAT_s112
                                 + 0.0270942593991666 w_k1_MAT_s113
                                 + 0.0650983005227146 w_k1_MAT_s114
                                 + 0.0427926042876679 w_k1_MAT_s115
                                 + 0.0434933363171169 w_k1_MAT_s116
                                 + 0.0420576378293423 w_k1_MAT_s117
                                 + 0.0268420265746781 w_k1_MSFT_s000
                                 + 0.0166454764896526 w_k1_MSFT_s001
                                 + 0.013668825137635 w_k1_MSFT_s002
                                 + 0.0172809623541929 w_k1_MSFT_s003
                                 + 0.013438492090254 w_k1_MSFT_s004
                                 + 0.0516815457735483 w_k1_MSFT_s005
                                 + 0.0119224587630175 w_k1_MSFT_s006
                                 + 0.0125415785556619 w_k1_MSFT_s007
                                 + 0.0644148432587342 w_k1_MSFT_s010
                                 + 0.0395241106417007 w_k1_MSFT_s011
                                 + 0.0612412615153269 w_k1_MSFT_s012
                                 + 0.0567155813533383 w_k1_MSFT_s013
                                 + 0.0414549881262099 w_k1_MSFT_s014
                                 + 0.0381340915987711 w_k1_MSFT_s015
                                 + 0.0630574122462908 w_k1_MSFT_s016
                                 + 0.097444458280116 w_k1_MSFT_s017
                                 + 0.0252828440984973 w_k1_MSFT_s100
                                 + 0.00738989395066531 w_k1_MSFT_s101
                                 + 0.00694299872805305 w_k1_MSFT_s102
                                 + 0.0116725088405622 w_k1_MSFT_s103
                                 + 0.0105660278128108 w_k1_MSFT_s104
                                 + 0.0198373442219965 w_k1_MSFT_s105
                                 + 0.0128245887836045 w_k1_MSFT_s106
                                 + 0.0166898845367796 w_k1_MSFT_s107
                                 + 0.0167724124645246 w_k1_MSFT_s110
                                 + 0.0351758899703955 w_k1_MSFT_s111
                                 + 0.0262428769642242 w_k1_MSFT_s112
                                 + 0.0271325582407113 w_k1_MSFT_s113
                                 + 0.056733280061693 w_k1_MSFT_s114
                                 + 0.0405594145710986 w_k1_MSFT_s115
                                 + 0.0400582845947384 w_k1_MSFT_s116
                                 + 0.0389697856875946 w_k1_MSFT_s117
                                 + 0.0272305313654895 w_k1_PG_s000
                                 + 0.0163209867254054 w_k1_PG_s001
                                 + 0.014097792629345 w_k1_PG_s002
                                 + 0.0182568315321763 w_k1_PG_s003
                                 + 0.0141546795799834 w_k1_PG_s004
                                 + 0.0489040859243286 w_k1_PG_s005
                                 + 0.0125738398908926 w_k1_PG_s006
                                 + 0.0117103973030675 w_k1_PG_s007
                                 + 0.0632620221536793 w_k1_PG_s010
                                 + 0.0358372051243939 w_k1_PG_s011
                                 + 0.0643595613888283 w_k1_PG_s012
                                 + 0.0482436686672079 w_k1_PG_s013
                                 + 0.0383575675391138 w_k1_PG_s014
                                 + 0.0354582306368684 w_k1_PG_s015
                                 + 0.0629197210192607 w_k1_PG_s016
                                 + 0.0892495011138581 w_k1_PG_s017
                                 + 0.0254850299728509 w_k1_PG_s100
                                 + 0.00782199905397461 w_k1_PG_s101
                                 + 0.00686419076729406 w_k1_PG_s102
                                 + 0.0115277397674295 w_k1_PG_s103
                                 + 0.0111821226713333 w_k1_PG_s104
                                 + 0.0213263243486711 w_k1_PG_s105
                                 + 0.013908742646133 w_k1_PG_s106
                                 + 0.017952912702027 w_k1_PG_s107
                                 + 0.0168162666767793 w_k1_PG_s110
                                 + 0.034160478880167 w_k1_PG_s111
                                 + 0.025835005173018 w_k1_PG_s112
                                 + 0.0250447571300782 w_k1_PG_s113
                                 + 0.0543229892328165 w_k1_PG_s114
                                 + 0.0387027518169869 w_k1_PG_s115
                                 + 0.0405144890334105 w_k1_PG_s116
                                 + 0.036740071624797 w_k1_PG_s117 >= 1.17
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
