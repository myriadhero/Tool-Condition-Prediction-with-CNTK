mkdir engine;
mkdir console;
dotnet new sln;

cd ./console;
dotnet new console;

cd..;

cd ./engine;
dotnet new classlib;

cd..;

dotnet sln add ./engine/engine.csproj;

dotnet sln add ./console/console.csproj;

cd ./console;

dotnet add reference ../engine/engine.csproj;

cd..;

code .