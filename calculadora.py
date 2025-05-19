import streamlit as st
import numpy as np
from scipy.stats import norm
import math as m
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import sample_colorscale
import pandas as pd

class OptionCalculator:
    def __init__(self):
        self.calculator_type = "Européia (BSM)"  # Inicialização padrão
        self.setup_interface()
        
    def setup_interface(self):
        st.title("Calculadora de Derivativos Financeiros")
        st.header("Ferramenta de Precificação de Opções")
        #st.subheader("Enzo Moreira, Gustavo Liang, Gabriel Fedele, Andre Aoki")

        # Move the calculator type selection to the main area
        self.calculator_type = st.radio(
            "Selecione o Método de Precificação:",
            ["Européia (BSM)",
             "Opção Binária (Monte Carlo)",
             "Opção Asiática (Aritmética)",
             "Opção Asiática (Geométrica)",
             "Européia/Americana (Árvore)",
             "Volatilidade Implícita"]
        )
        
        # Sidebar for inputs
        with st.sidebar:
            st.header("Parâmetros")
            self.underlying_price = st.number_input("Preço do Ativo Subjacente", min_value=0.0, value=100.0, step=1.0)
            self.strike = st.number_input("Preço de Exercício", min_value=0.0, value=100.0, step=1.0)
            self.interest = st.number_input("Taxa de Juros Anual (%)", min_value=0.0, value=5.0, step=0.1) / 100
            is_volatility_calc = self.calculator_type == "Volatilidade Implícita"
            self.volatility = st.number_input("Volatilidade Anual (%)", min_value=0.0, value=20.0, step=1.0, disabled=is_volatility_calc) / 100
            self.expiry = st.number_input("Tempo até o Vencimento (anos)", min_value=0.0, value=1.0, step=1.0)
            
            # Adiciona opção para mostrar visualizações
            self.show_viz = st.checkbox("Mostrar Visualizações", value=True)

    def price_european_bs(self, option_style='call'):
        """Calcula o preço de uma opção europeia usando Black-Scholes"""
        d1 = (np.log(self.underlying_price / self.strike) + 
              (self.interest + 0.5 * self.volatility**2) * self.expiry) / (self.volatility * np.sqrt(self.expiry))
        d2 = d1 - self.volatility * np.sqrt(self.expiry)
        
        if option_style == 'call':
            price = self.underlying_price * norm.cdf(d1) - self.strike * np.exp(-self.interest * self.expiry) * norm.cdf(d2)
        else:
            price = self.strike * np.exp(-self.interest * self.expiry) * norm.cdf(-d2) - self.underlying_price * norm.cdf(-d1)
        return price

    def price_binary_mc(self, n_simulations=10000):
        """Calcula o preço de uma opção binária usando Monte Carlo"""
        n_days = int(self.expiry * 252)
        dt = self.expiry / n_days
        
        drift = (self.interest - 0.5 * self.volatility**2) * dt
        diffusion = self.volatility * np.sqrt(dt)
        
        Z = np.random.randn(n_days, n_simulations)
        daily_returns = drift + diffusion * Z
        cumulative_returns = np.cumsum(daily_returns, axis=0)
        price_paths = self.underlying_price * np.exp(cumulative_returns)
        
        payoff = np.where(price_paths[-1] > self.strike, 1, 0)
        return np.exp(-self.interest * self.expiry) * np.mean(payoff)

    def price_arithmetic_asian(self, option_style='call', n_simulations=10000):
        """Calcula o preço de uma opção asiática aritmética"""
        steps = 252
        sim_steps = int(steps * self.expiry)
        dt = 1/steps
        
        mu = (self.interest - 0.5 * self.volatility**2) * dt
        sigma_dt = self.volatility * np.sqrt(dt)
        
        St = np.zeros((sim_steps, n_simulations))
        St[0] = self.underlying_price
        
        for i in range(1, sim_steps):
            Z = np.random.randn(n_simulations)
            St[i] = St[i-1] * np.exp(mu + sigma_dt * Z)
        
        path_means = np.mean(St, axis=0)
        
        if option_style == 'call':
            payoffs = np.maximum(0, path_means - self.strike)
        else:
            payoffs = np.maximum(0, self.strike - path_means)
        
        return np.exp(-self.interest * self.expiry) * np.mean(payoffs)
    
    def price_geometric_asian(self, option_style='call'):
        """Calcula o preço de uma opção asiática geométrica"""
        Nt = int(self.expiry * 252)
        adj_sigma = self.volatility * m.sqrt((2*Nt+1)/(6*(Nt+1)))
        rho = 0.5 * (self.interest - (self.volatility**2)*0.5 + adj_sigma**2)
        
        d1 = (m.log(self.underlying_price/self.strike) + 
              (rho + 0.5*adj_sigma**2)*self.expiry) / (adj_sigma*m.sqrt(self.expiry))
        d2 = (m.log(self.underlying_price/self.strike) + 
              (rho - 0.5*adj_sigma**2)*self.expiry) / (adj_sigma*m.sqrt(self.expiry))
        
        if option_style == 'call':
            price = m.exp(-self.interest*self.expiry) * (
                self.underlying_price * m.exp(rho*self.expiry) * norm.cdf(d1) - 
                self.strike * norm.cdf(d2)
            )
        else:
            price = m.exp(-self.interest*self.expiry) * (
                self.strike * norm.cdf(-d2) - 
                self.underlying_price * m.exp(rho*self.expiry) * norm.cdf(-d1)
            )
        return price

    def price_tree(self, option_style='call', exercise_type='Européia', num_steps=100):
        """Calcula o preço usando árvore binomial"""
        dt = self.expiry / num_steps
        u = np.exp(self.volatility * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.interest * dt) - d) / (u - d)
        
        # Preços finais
        stock_prices = np.array([
            self.underlying_price * u**(num_steps-j) * d**j 
            for j in range(num_steps + 1)
        ])
        
        # Payoffs no vencimento
        if option_style == 'call':
            option_values = np.maximum(0, stock_prices - self.strike)
        else:
            option_values = np.maximum(0, self.strike - stock_prices)
            
        # Backward induction
        for i in range(num_steps-1, -1, -1):
            stock_prices = [
                self.underlying_price * u**(i-j) * d**j 
                for j in range(i + 1)
            ]
            option_values = np.exp(-self.interest * dt) * (
                p * option_values[:-1] + (1-p) * option_values[1:]
            )
            
            # Tratamento para opções americanas
            if exercise_type == 'Americana':
                for j in range(i + 1):
                    intrinsic = (
                        np.maximum(0, stock_prices[j] - self.strike) 
                        if option_style == 'call' 
                        else np.maximum(0, self.strike - stock_prices[j])
                    )
                    option_values[j] = np.maximum(option_values[j], intrinsic)
                    
        return option_values[0]


    def vega(self, S, K, T, r, sigma):
        """Calcula o vega da opção"""
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)

    def calc_implied_volatility(self, market_price, option_style='call', tol=1e-5, max_iter=100):
        """Calcula a volatilidade implícita"""
        sigma = 0.2  # Estimativa inicial
        
        for i in range(max_iter):
            price = self.price_european_bs(option_style)
            diff = price - market_price
            
            if abs(diff) < tol:
                return sigma
                
            vega = self.vega(self.underlying_price, self.strike, 
                            self.expiry, self.interest, sigma)
            if vega == 0:
                return np.nan
                
            sigma = sigma - diff / vega
            self.volatility = sigma
            
        return sigma
    
    def plot_price_sensitivity(self):
        """Plota a sensibilidade do preço"""
        prices = np.linspace(self.underlying_price * 0.5, self.underlying_price * 1.5, 100)
        call_prices = []
        put_prices = []
        orig_price = self.underlying_price
        
        for price in prices:
            self.underlying_price = price
            call_prices.append(self.price_european_bs('call'))
            put_prices.append(self.price_european_bs('put'))
        
        self.underlying_price = orig_price
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=call_prices, name="Call Option"))
        fig.add_trace(go.Scatter(x=prices, y=put_prices, name="Put Option"))
        
        fig.update_layout(
            title="Sensibilidade do Preço da Opção ao Preço do Ativo",
            xaxis_title="Preço do Ativo",
            yaxis_title="Preço da Opção",
            height=500
        )
        
        st.plotly_chart(fig)

    def plot_geometric_sensitivity(self):
        """Plota a sensibilidade do preço para opção asiática geométrica"""
        prices = np.linspace(self.underlying_price * 0.5, self.underlying_price * 1.5, 100)
        call_prices = []
        put_prices = []
        orig_price = self.underlying_price
        
        for price in prices:
            self.underlying_price = price
            call_prices.append(self.price_geometric_asian('call'))
            put_prices.append(self.price_geometric_asian('put'))
        
        self.underlying_price = orig_price
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=call_prices, name="Call Option"))
        fig.add_trace(go.Scatter(x=prices, y=put_prices, name="Put Option"))
        
        fig.update_layout(
            title="Sensibilidade do Preço da Opção ao Preço do Ativo",
            xaxis_title="Preço do Ativo",
            yaxis_title="Preço da Opção",
            height=500
        )
        
        st.plotly_chart(fig)

    def plot_monte_carlo_paths(self, n_paths=50):
        """Plota caminhos da simulação Monte Carlo com estatísticas"""
        dt = self.expiry / 252
        paths = np.zeros((n_paths, 253))
        paths[:, 0] = self.underlying_price
        
        for i in range(1, 253):
            z = np.random.standard_normal(n_paths)
            paths[:, i] = paths[:, i-1] * np.exp(
                (self.interest - 0.5 * self.volatility**2) * dt +
                self.volatility * np.sqrt(dt) * z
            )
        
        fig = go.Figure()

        colors = sample_colorscale('Blues', [i / (n_paths - 1) for i in range(n_paths)])

        for i in range(n_paths):
            fig.add_trace(go.Scatter(y=paths[i], mode='lines', 
                                   showlegend=False,
                                   line=dict(width=1.2, color=colors[i])))
        
        # Adicionar média
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(y=mean_path, mode='lines',
                                name='Média',
                                line=dict(color='red', width=2)))
        
        # Adicionar estatísticas
        final_prices = paths[:,-1]
        mean_price = np.mean(final_prices)
        std_price = np.std(final_prices)
        
        stats_text = f"""
        Preço Médio Final: R${mean_price:.2f}
        Desvio Padrão: R${std_price:.2f}
        IC 95%: [R${mean_price - 1.96*std_price:.2f}, R${mean_price + 1.96*std_price:.2f}]
        """
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=1.0, y=1.0,
            showarrow=False,
            bordercolor="black",
            borderwidth=0.5,
            borderpad=4,
            bgcolor="rgba(255,255,255,0.1)",
            align="left"
        )
        
        fig.update_layout(
            title="Caminhos de Preço Monte Carlo",
            xaxis_title="Passos Temporais",
            yaxis_title="Preço do Ativo",
            height=500,
            #plot_bgcolor='rgba(0,0,0,0)',
            #paper_bgcolor='rgba(255, 75, 75, 0.1)'
        )
        
        st.plotly_chart(fig)

    def plot_asian_paths(self, n_paths=50, geometric=False):
        """Plota caminhos da simulação Monte Carlo com média para opções asiáticas"""
        dt = self.expiry / 252
        paths = np.zeros((n_paths, 253))
        paths[:, 0] = self.underlying_price
        
        for i in range(1, 253):
            z = np.random.standard_normal(n_paths)
            paths[:, i] = paths[:, i-1] * np.exp(
                (self.interest - 0.5 * self.volatility**2) * dt +
                self.volatility * np.sqrt(dt) * z
            )
        
        fig = go.Figure()

        colors = sample_colorscale('Blues', [i / (n_paths - 1) for i in range(n_paths)])

        # Plotar caminhos
        for i in range(n_paths):
            fig.add_trace(go.Scatter(y=paths[i], mode='lines', 
                               showlegend=False,
                               line=dict(width=1.2, color=colors[i])))
        
        # Adicionar média
        if geometric:
            mean_path = np.exp(np.mean(np.log(paths), axis=0))
            title = "Caminhos de Preço Monte Carlo (com Média Geométrica)"
        else:
            mean_path = np.mean(paths, axis=0)
            title = "Caminhos de Preço Monte Carlo (com Média Aritmética)"
            
        fig.add_trace(go.Scatter(y=mean_path, mode='lines',
                                name='Média',
                                line=dict(color='red', width=2)))
        
        # Adicionar estatísticas
        final_means = mean_path[-1]
        std_price = np.std(paths[:,-1])
        
        stats_text = f"""
        {'Média Geométrica' if geometric else 'Média Aritmética'} Final: R${final_means:.2f}
        Desvio Padrão: R${std_price:.2f}
        IC 95%: [R${final_means - 1.96*std_price:.2f}, R${final_means + 1.96*std_price:.2f}]
        """
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=1.0, y=1.0,
            showarrow=False,
            bordercolor="black",
            borderwidth=0.5,
            borderpad=4,
            bgcolor="rgba(255,255,255,0.1)",
            align="left"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Passos Temporais",
            yaxis_title="Preço do Ativo",
            height=500
        )
        
        st.plotly_chart(fig)

    def plot_implied_vol_convergence(self, market_price, option_style='call'):
        """Plota convergência da volatilidade implícita"""
        vols = np.linspace(0.01, 1, 100)
        prices = []
        orig_vol = self.volatility
        
        for vol in vols:
            self.volatility = vol
            prices.append(self.price_european_bs(option_style))
        
        self.volatility = orig_vol
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols*100, y=prices,
                                name="Preço BS"))
        fig.add_hline(y=market_price, name="Preço de Mercado",
                     line=dict(color='red', dash='dash'))
        
        fig.update_layout(
            title="Convergência da Volatilidade Implícita",
            xaxis_title="Volatilidade (%)",
            yaxis_title="Preço da Opção",
            height=500
        )
        
        st.plotly_chart(fig)

    def plot_tree_diagram(self, num_steps=5):
        """Plota diagrama simplificado da árvore binomial"""
        dt = self.expiry / num_steps
        u = np.exp(self.volatility * np.sqrt(dt))
        d = 1 / u
        
        # Criar coordenadas para os nós
        x_coords = []
        y_coords = []
        prices = []
        
        for i in range(num_steps + 1):
            for j in range(i + 1):
                x_coords.append(i)
                y_coords.append(j - i/2)
                prices.append(self.underlying_price * (u**(i-j)) * (d**j))
        
        fig = go.Figure()
        
        # Adicionar nós
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            text=[f'R${p:.1f}' for p in prices],
            textposition='top center',
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Diagrama da Árvore Binomial (Primeiros 5 Passos)",
            xaxis_title="Passos",
            yaxis_title="Níveis",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig)

    def run(self):
        if self.calculator_type == "Européia (BSM)":
            option_style = st.selectbox("Tipo de Opção", ["Call", "Put"])
            price = self.price_european_bs(option_style.lower())
            st.write(f"Valor da Opção Black-Scholes: R${price:.2f}")
            
            if self.show_viz:
                st.subheader("Visualizações")
                self.plot_price_sensitivity()
            
        elif self.calculator_type == "Opção Binária (Monte Carlo)":
            num_sims = st.number_input("Número de Simulações", min_value=1000, max_value=100000, value=10000, step=1000)
            price = self.price_binary_mc(num_sims)
            st.write(f"Valor da Opção Binária: R${price:.2f}")
            
            if self.show_viz:
                st.subheader("Simulações Monte Carlo")
                self.plot_monte_carlo_paths()
            
        elif self.calculator_type == "Opção Asiática (Aritmética)":
            option_style = st.selectbox("Tipo de Opção", ["Call", "Put"])
            num_sims = st.number_input("Número de Simulações", min_value=1000, max_value=100000, value=10000, step=1000)
            price = self.price_arithmetic_asian(option_style.lower(), num_sims)
            st.write(f"Valor da Opção Asiática Aritmética: R${price:.2f}")
            
            if self.show_viz:
                st.subheader("Simulações Monte Carlo")
                self.plot_asian_paths(geometric=False)
            
        elif self.calculator_type == "Opção Asiática (Geométrica)":
            option_style = st.selectbox("Tipo de Opção", ["Call", "Put"])
            price = self.price_geometric_asian(option_style.lower())
            st.write(f"Valor da Opção Asiática Geométrica: R${price:.2f}")
            
            if self.show_viz:
                st.subheader("Análise do Preço da Opção")
                self.plot_geometric_sensitivity()  # Novo método para sensibilidade específica da asiática geométrica
                    
        elif self.calculator_type == "Européia/Americana (Árvore)":
            option_style = st.selectbox("Tipo de Opção", ["Call", "Put"])
            exercise_type = st.selectbox("Estilo de Exercício", ["Européia", "Americana"])
            steps = st.number_input("Número de Passos", min_value=10, value=100)
            price = self.price_tree(option_style.lower(), exercise_type, steps)
            st.write(f"Valor da Opção (Método da Árvore): R${price:.2f}")
            
            if self.show_viz:
                st.subheader("Visualizações")
                self.plot_price_sensitivity()
                if steps <= 5:  # Só mostra o diagrama da árvore para poucos passos
                    self.plot_tree_diagram(steps)
                else:
                    if st.button("Mostrar Diagrama Simplificado da Árvore (5 passos)"):
                        self.plot_tree_diagram(5)
            
        elif self.calculator_type == "Volatilidade Implícita":
            option_style = st.selectbox("Tipo de Opção", ["Call", "Put"])
            market_price = st.number_input("Preço de Mercado", min_value=0.0, value=10.0)
            impl_vol = self.calc_implied_volatility(market_price, option_style.lower())
            
            if np.isnan(impl_vol):
                st.write("Não foi possível calcular a volatilidade implícita com os parâmetros fornecidos")
            else:
                st.write(f"Volatilidade Implícita: {impl_vol*100:.2f}%")
                
            if self.show_viz:
                st.subheader("Análise de Convergência")
                self.plot_implied_vol_convergence(market_price, option_style.lower())

if __name__ == "__main__":
    calculator = OptionCalculator()
    calculator.run()
